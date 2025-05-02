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

# TODO CHECK fd for base , is it valid for collocated
from jax_cfd.base import finite_differences as fd
from jax_cfd.collocated import boundaries
from jax_cfd.base import grids

from jax_cfd.collocated import advection
from jax_cfd.collocated import momentum_stress

import functools
funcutils = base.funcutils

def run_collocated(size, seed=0, inner_steps=100, outer_steps=10):
    # density = 1000.
    max_velocity = 2.0
    cfl_safety_factor = 0.5

    properties = dict(
        m_ion=1.0,
        diffusivity=1e-1,
        thermal_cond=1e-0,
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

    # TODO: add a function to generate a initalize density field
    n_fu = lambda x, y: 0 * jnp.sin(x/2) * jnp.sin(y/2) + 1
    n_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((1,1),(1,1)))
    density = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, n_fu, n_bc)
    
    # Define the initial velocity conditions.
    v0 = collocated.initial_conditions.initial_v_field(
        jax.random.PRNGKey(seed), grid, (2*np.pi*jnp.ones(grid.shape), 2*np.pi*jnp.ones(grid.shape)), )
        # [boundaries.neumann_boundary_conditions( ndim=grid.ndim, bc_vals=((0,0),(0,0)))]*2 )

    # Define the initial temperature conditions.
    T_fu = lambda x, y: 0.5 * jnp.sin(x/2) * jnp.sin(y/2)
    T_bc = None#boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((0,0),(0,0)))
    T0 = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, T_fu, T_bc)

    # TODO: add a function to generate a initalize pressure field
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

ds_collocated = run_collocated(size=64)
ds_collocated.to_netcdf(dir_path+"/examples/solvers/compressible/my_dataset.nc")