import sys, os
sys.path.append('./')
dir_path = os.getcwd()

import jax
import jax.numpy as jnp
import jax_cfd.base as base
import jax_cfd.collocated as collocated
import jax_cfd.continuity as continuity
import jax_cfd.momentum as momentum
import jax_cfd.energy as energy
import jax_cfd.data.xarray_utils as xru
import numpy as np
import seaborn
import xarray
import scipy

# TODO CHECK fd for base , is it valid for collocated
from jax_cfd.base import finite_differences as fd
from jax_cfd.collocated import boundaries
from jax_cfd.base import grids

from jax_cfd.collocated import advection
from jax_cfd.collocated import momentum_stress

import functools
funcutils = base.funcutils

import matplotlib.pyplot as plt

def heat_2d_analytical(x, y, t, Lx, Ly, alpha, f, M=20, N=20):
    """
    Analytical solution for 2D transient heat diffusion with zero Dirichlet BCs.
    
    Parameters:
    - x, y: 2D meshgrid arrays
    - t: time
    - Lx, Ly: domain lengths in x and y
    - alpha: thermal diffusivity
    - f: function f(x, y), initial condition
    - M, N: number of terms in x and y directions
    
    Returns:
    - u: 2D array of solution values at each (x, y)
    """
    u = np.zeros_like(x, dtype=float)
    
    for m in range(1, M+1):
        for n in range(1, N+1):
            # Compute A_mn
            integrand = lambda y_, x_: f(x_, y_) * np.sin(m*np.pi*x_/Lx) * np.sin(n*np.pi*y_/Ly)
            A_mn, _ = scipy.integrate.dblquad(integrand, 0, Lx, lambda _: 0, lambda _: Ly)
            A_mn *= (4 / (Lx * Ly))
            
            # Compute term contribution
            term = (
                A_mn *
                np.sin(m * np.pi * x / Lx) *
                np.sin(n * np.pi * y / Ly) *
                np.exp(-alpha * np.pi**2 * ((m / Lx)**2 + (n / Ly)**2) * t)
            )
            u += term
            
    return u

thermal_cond = 1e-1
T_fu = lambda x, y: np.abs(0.5 * np.sin(x) * np.sin(y))
Lx, Ly = 2 * np.pi, 2 * np.pi


def run_collocated(size, seed=0, inner_steps=100, outer_steps=100):
    # density = 1000.
    max_velocity = 2.0
    cfl_safety_factor = 0.5

    properties = dict(
        m_ion=1.0,
        diffusivity=1e-1,
        thermal_cond=thermal_cond,
        dyna_viscosity=1e-1,
        Cv=1.0,
    )

    # Define the physical dimensions of the simulation.
    grid = base.grids.Grid((size, size),
                            domain=((0, 2 * jnp.pi), (0, 2 * jnp.pi)))

    # Choose a time step.
    dt = 0.001
    #   dt = base.equations.stable_time_step(
    #       max_velocity, cfl_safety_factor, viscosity, grid)

    # constant density
    n_fu = lambda x, y: 0 * jnp.sin(x/2) * jnp.sin(y/2) + 1
    n_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((1,1),(1,1)))
    density = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, n_fu, n_bc)
    
    # zero velocity
    v0 = collocated.initial_conditions.initial_v_field(
        jax.random.PRNGKey(seed), grid, (0*np.pi*jnp.ones(grid.shape), 0*np.pi*jnp.ones(grid.shape)), )
        # [boundaries.neumann_boundary_conditions( ndim=grid.ndim, bc_vals=((0,0),(0,0)))]*2 )

    # Define the initial temperature conditions.
    T_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((0,0),(0,0)))
    T0 = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, T_fu, T_bc)

    # Constant pressure
    p_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((1,1),(1,1)))
    p0 = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, 1*jnp.ones(grid.shape), p_bc)
    
    state = dict(
        rho=density,
        v=v0,
        T=T0,
        p=p0,
    )

    ############################## density equation ##############################
    def convect(rho, state):
        return advection.advect_linear(rho, state['v'])
    
    # Define a step function and use it to compute a trajectory.
    density_step_fu = continuity.equations.explicit_continuity_eq(state_var='rho',dt=dt, grid=grid,
                                                                  convect=convect, properties=properties)
    
    ############################## Momentum equation (NS) ##############################
    
    # TODO CHECK if this is correct
    def convect(rhov, v, state):
        return tuple(
            advection.advect_linear(rhou, v, dt) + fd.central_difference(state['p'], axis) 
        for axis, rhou in enumerate(rhov))
    
    # TODO(deepak): add div(\tau.v) term
    def diff_stress(v):
        tau = momentum_stress.stress(v)
        return tuple(-properties['dyna_viscosity']*fd.centered_divergence(taui) for taui in tau)

    # TODO CHECK if this is correct
    def diffuse(rhov, v, state):
        # return tuple(properties['dyna_viscosity'] * fd.laplacian(u) for u in v)
        return diff_stress(v)

    # # TODO CHECK if this is correct
    # def forcing(rhov, v, state):
    #     return collocated.finite_differences.forward_difference(state['p'])

    momentum_step_fu = momentum.equations.explicit_momentum_eq(state_var='v',dt=dt, grid=grid,
                                                               convect=convect, diffuse=diffuse, properties=properties)


    ############################## energy equation ##############################
    def convect(rhoE, T, state):  # pylint: disable=function-redefined
        return advection.advect_linear(rhoE, state['v']) + advection.advect_linear(state['p'], state['v'])

    def diffuse(rhoE, T, state):
        tau = momentum_stress.stress(state['v'])
        tauv = tuple(
            grids.GridVariable(properties['dyna_viscosity'] * sum(tauij.array*u.array for tauij, u in zip(taui, state['v'])),
                               # TODO CHECK if bc is correct
                                bc=boundaries.dirichlet_boundary_conditions( ndim=len(state['v']), bc_vals=((0,0),(0,0))) )
            for taui in tau)
        q_diff = properties['thermal_cond'] * fd.laplacian(T)
        return q_diff + fd.centered_divergence(tauv)
    
    # Define a step function and use it to compute a trajectory.
    energy_step_fu = energy.equations.explicit_energy_eq(state_var='T', dt=dt, grid=grid, 
                                                         convect=convect, diffuse=diffuse, properties=properties)

    def _step_fu(state):
        state = density_step_fu (state)
        state = momentum_step_fu(state)
        state = energy_step_fu  (state)
        return state
    
    step_fn = funcutils.repeated( _step_fu, steps=inner_steps)
    rollout_fn = jax.jit(funcutils.trajectory(step_fn, outer_steps))
    _, trajectory = jax.device_get(rollout_fn(state))

    # load into xarray for visualization and analysis
    ds = xarray.Dataset(
        {
            'rho': (('time', 'x', 'y'), trajectory['rho'].data),
            'u': (('time', 'x', 'y'), trajectory['v'][0].data),
            'v': (('time', 'x', 'y'), trajectory['v'][1].data),
            'T': (('time', 'x', 'y'), trajectory['T'].data),
        },
        coords={
            'x': grid.axes()[0],
            'y': grid.axes()[1],
            'time': dt * inner_steps * np.arange(outer_steps)
        }
    )
    return ds

ds = run_collocated(size=64)

ana_sol = functools.partial(heat_2d_analytical, f=T_fu, Lx=Lx, Ly=Ly, alpha=thermal_cond)

# Choose n evenly spaced time indices
n = 5  # or any number you want
time_indices = np.linspace(0, 
                           len(ds['time']) - 1, 
                           n, dtype=int)

plt.close()
# Create subplots
fig, ax = plt.subplots(3, n, figsize=(4*n, 4*3), constrained_layout=True)

vmin = 0#ds['T'][1:].min().item()
vmax = 0.5#ds['T'][1:].max().item()
for i, t_idx in enumerate(time_indices):
    ana_sol_ = ana_sol(*np.meshgrid(ds['x'], ds['y']), ds['time'].values[t_idx])
    ax[0,i].contourf(ds['x'], ds['y'], ana_sol_,
                   levels=np.linspace(vmin, vmax, 20), 
                   cmap='viridis')
    im1 = ax[1,i].contourf(ds['x'], ds['y'], ds['T'][t_idx], 
                   levels=np.linspace(vmin, vmax, 20), 
                   cmap='viridis')
    im2 = ax[2,i].contourf(ds['x'], ds['y'], np.abs(ana_sol_-ds['T'][t_idx]),
                   cmap='viridis')
    ax[0,i].set_title(f"T at t={ds['time'].values[t_idx]:.2f}")
    fig.colorbar(im2, ax=ax[2,i], orientation='vertical', label='Error')

# Add shared colorbar
fig.colorbar(im1, ax=ax[1,i], orientation='vertical', label='Temperature')

plt.savefig('/home/deepak/Project/trial_short_project/jax-plasma/examples/solvers/heat_eq/heat_diffusion_val_2d.png', dpi=300)