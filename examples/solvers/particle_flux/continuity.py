import sys, os
sys.path.append('./')
dir_path = os.getcwd()

import jax
import jax.numpy as jnp
import jax_cfd.base as base
import jax_cfd.collocated as collocated
import jax_cfd.energy as energy
import jax_cfd.continuity as continuity
import jax_cfd.data.xarray_utils as xru
import numpy as np
import seaborn
import xarray

from jax_cfd.base import finite_differences as fd
from jax_cfd.collocated import boundaries
from jax_cfd.base import grids

import functools
funcutils = base.funcutils

def run_collocated(size, seed=0, inner_steps=100, outer_steps=1000):
    # density = 1000.
    max_velocity = 2.0
    cfl_safety_factor = 0.5

    properties = dict(
        m_ion=1.0,
        diffusivity=1e-1,
        thermal_cond=1e-1,
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
    n_fu = lambda x, y: 0 * jnp.sin(x/2) * jnp.sin(y/2) + 20
    n_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((20,20),(20,20)))
    density = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, n_fu, n_bc)
    
    n_fu = lambda x, y: 0 * jnp.sin(x/2) * jnp.sin(y/2)
    n_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((0,0),(0,0)))
    ion_density = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, n_fu, n_bc)
    
    # Define the initial temperature conditions.
    T_fu = lambda x, y: 0.5 * jnp.sin(x/2) * jnp.sin(y/2)
    T_bc = None#boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((0,0),(0,0)))
    T0 = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, T_fu, T_bc)
    v0 = collocated.initial_conditions.initial_v_field(
        jax.random.PRNGKey(seed), grid, (2*np.pi*jnp.ones(grid.shape), 2*np.pi*jnp.ones(grid.shape)), )
        # [boundaries.neumann_boundary_conditions( ndim=grid.ndim, bc_vals=((0,0),(0,0)))]*2 )
    # TODO: add a function to generate a initalize pressure field
    p_bc = boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),((1,1),(1,1)))
    p0 = energy.initial_conditions.initial_T_field(
        jax.random.PRNGKey(seed), grid, 0.1*jnp.ones(grid.shape), p_bc)
    
    state = dict(
        n=density,
        ion_n=ion_density,
        vp=v0,
        T=T0,
        p=p0,
    )

    def diffuse(n):
        particle_flux = properties['diffusivity'] * fd.laplacian(n)
        return particle_flux
    
    def forcing(n):
        return grid.eval_on_mesh(lambda x, y: 1 * jnp.sin(x/2) * jnp.sin(y/2), grid.cell_center)
    
    # Define a step function and use it to compute a trajectory.
    density_step_fu = continuity.equations.explicit_continuity_eq(state_var='n',dt=dt, grid=grid, 
                            diffuse=diffuse, forcing=lambda n: -forcing(n), properties=properties)
    
    # Define a step function and use it to compute a trajectory.
    ion_density_step_fu = continuity.equations.explicit_continuity_eq(state_var='ion_n',dt=dt, grid=grid, 
                            diffuse=diffuse, forcing=forcing, properties=properties)
    def _step_fu(state):
        state = density_step_fu(state)
        state = ion_density_step_fu(state)
        return state
    
    step_fn = funcutils.repeated( _step_fu, steps=inner_steps)
    rollout_fn = jax.jit(funcutils.trajectory(step_fn, outer_steps))
    _, trajectory = jax.device_get(rollout_fn(state))

    # load into xarray for visualization and analysis
    ds = xarray.Dataset(
        {
            'n': (('time', 'x', 'y'), trajectory['n'].data),
            'ion_n': (('time', 'x', 'y'), trajectory['ion_n'].data),
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
ds_collocated.to_netcdf(dir_path+"/examples/solvers/particle_flux/my_dataset.nc")