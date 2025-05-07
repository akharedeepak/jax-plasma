import sys, os
sys.path.append('./')
dir_path = os.getcwd()

import jax
import jax.numpy as jnp
import jax_cfd.base as base
import jax_cfd.collocated as col
import jax_cfd.continuity as continuity
import jax_cfd.momentum as momentum
import jax_cfd.energy as energy
import jax_cfd.data.xarray_utils as xru

import tree_math
import numpy as np
import seaborn
import xarray

# # TODO CHECK fd for base , is it valid for collocated
from jax_cfd.base import finite_differences as fd
# from jax_cfd.collocated import boundaries
# from jax_cfd.base import grids

# from jax_cfd.collocated import advection
# from jax_cfd.collocated import momentum_stress

import functools
funcutils = base.funcutils

seed = 0
key = jax.random.PRNGKey(seed)

nx = 10
ny = 100
Lx = 10
Ly = 1

size = (nx, ny)
domain = ((0, Lx), (0, Ly))
dx = Lx/nx
dy = Ly/ny
density = 1.
# Re_in=1000 #Reynolds number based on the inlet width
# viscosity = 1e-3

# density = 1000.
max_velocity = 2.0
cfl_safety_factor = 0.5

# number of computations in each repeated
inner_steps = 1000
# number of outputs in the main loop
outer_steps = 500

# Choose a time step.
dt = 1e-5
#   dt = base.equations.stable_time_step(
#       max_velocity, cfl_safety_factor, viscosity, grid)


properties = dict(
    gamma = 1.4,
    R = 287.0,       # J/kg/K
    m_ion=1.0,
    diffusivity=1e-1,
    thermal_cond=0.026, # W/mK
    dyna_viscosity=1e-1,
)
properties['Cv'] = properties['R'] / (properties['gamma'] - 1)

T0 = 300.0      # K
_p0 = 1e5
# rho0 = _p0/( properties['R'] *T0)
dpdx = 1.
# Define the physical dimensions of the simulation.
grid = base.grids.Grid(size, domain=domain)

# Define the initial velocity conditions.
subkey, key = jax.random.split(key)
v_bc=(col.boundaries.ConstantBoundaryConditions((('periodic', 'periodic'),('dirichlet','dirichlet')),((0,0) ,(0,0))),
      col.boundaries.ConstantBoundaryConditions((('periodic', 'periodic'),('dirichlet','dirichlet')),((0,0) ,(0,0))))
vx = lambda x, y: 0*y*y
vy = lambda x, y: 0.*x
v0 = col.initial_conditions.initial_v_field(subkey, grid, 
                                            (vx, vy), #(0.1*np.pi*jnp.ones(grid.shape), 0.*np.pi*jnp.ones(grid.shape)), 
                                            v_bc)

# TODO: add a function to generate a initalize pressure field
subkey, key = jax.random.split(key)
p_bc = col.boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('neumann','neumann')),((_p0,_p0-0*Lx),(0,0)))
p0 = energy.initial_conditions.initial_T_field(subkey, grid, lambda x, y: _p0-0*x, p_bc)

# TODO: add a function to generate a initalize density field
subkey, key = jax.random.split(key)
rho_bc = col.boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('neumann','neumann')),
                                                   ((_p0/(properties['R']*T0),(_p0-0*Lx)/(properties['R']*T0)),(0,0)))
rho0 = energy.initial_conditions.initial_T_field(subkey, grid, lambda x, y: (_p0-0*x)/(properties['R']*T0), rho_bc)

# Define the initial temperature conditions.
subkey, key = jax.random.split(key)
T_bc = col.boundaries.ConstantBoundaryConditions((('neumann', 'neumann'),('dirichlet','dirichlet')),((0,0),(1*T0,T0)))
T0 = energy.initial_conditions.initial_T_field(subkey, grid, T0*jnp.ones(grid.shape), T_bc)

state = dict(
    rho=rho0,
    v=v0,
    T=T0,
    p=p0,
    rho_new=rho0,
    v_new=v0,
    T_new=T0,
)

############################## density equation ##############################
def convect(rho, state):
    flux = col.conservatives._rhov(rho, state['v'])
    return -fd.centered_divergence(flux)

# Define a step function and use it to compute a trajectory.
density_step_fu = continuity.equations.explicit_continuity_eq(state_var='rho',dt=dt, grid=grid,
                                                              convect=convect, properties=properties)

############################## Momentum equation (NS) ##############################

# TODO CHECK if this is correct
def convect(rhov, v, state):
    flux = col.conservatives._rhovv(rhov, state['v'])
    return (-fd.centered_divergence(flux[0]) + dpdx, 
            -fd.centered_divergence(flux[1]))
    
    # flux = col.conservatives._rhovv_plus_p(rhov, state['v'], state['p'])
    # return tuple(-fd.centered_divergence(f) for f in flux)
    # flux = col.conservatives._rhovv(rhov, state['v'])
    # return tuple(-fd.centered_divergence(f) 
    #              -fd.central_difference(state['p'], axis) 
    #              for axis, f in enumerate(flux))


# TODO(deepak): add div(\tau.v) term
def diff_stress(v):
    tau = col.momentum_stress.stress(v, properties['dyna_viscosity'])
    return tuple(fd.centered_divergence(taui) for taui in tau)

# TODO CHECK if this is correct
def diffuse(rhov, v, state):
    return diff_stress(v)
    # # test 1
    # return tuple(properties['dyna_viscosity'] * fd.laplacian(u) for u in v)

# # TODO CHECK if this is correct
# def forcing(rhov, v, state):
#     return collocated.finite_differences.forward_difference(state['p'])

momentum_step_fu = momentum.equations.explicit_momentum_eq(state_var='v',dt=dt, grid=grid,
                                                           convect=convect, diffuse=diffuse, properties=properties)


############################## energy equation ##############################
def convect(rhoE, T, state):  # pylint: disable=function-redefined
    flux1 = col.conservatives._rhoEv(rhoE, state['v'])
    flux2 = col.conservatives._pv(state['p'], state['v'])
    return (-fd.centered_divergence(flux1) 
            -fd.centered_divergence(flux2))

def diffuse(rhoE, T, state):
    tau  = col.momentum_stress.stress(state['v'], properties['dyna_viscosity'])
    tauv = col.conservatives._tauv(tau, state['v'])
    q_diff = properties['thermal_cond'] * fd.laplacian(T)
    return q_diff + fd.centered_divergence(tauv)

# Define a step function and use it to compute a trajectory.
energy_step_fu = energy.equations.explicit_energy_eq(state_var='T', dt=dt, grid=grid,
                                                     convect=convect, diffuse=diffuse, 
                                                     properties=properties)

def _step_fu(state):
    state['p'] = base.grids.GridVariable(state['rho'].array * properties['R'] * state['T'].array, bc=state['p'].bc)
    
    state = density_step_fu (state)
    state = momentum_step_fu(state)
    state = energy_step_fu  (state)

    for var in [
        'rho', 
        'v', 
        'T'
        ]:
        state[var] = state[var+'_new']
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
        'p': (('time', 'x', 'y'), trajectory['p'].data),
    },
    coords={
        'x': grid.axes()[0],
        'y': grid.axes()[1],
        'time': dt * inner_steps * np.arange(outer_steps)
    }
)

ds.to_netcdf(dir_path+"/examples/solvers/compressible/02_poiseuille_flow/my_dataset.nc")