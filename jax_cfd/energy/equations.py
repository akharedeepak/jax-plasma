import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_cfd.collocated import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.energy import time_stepping
import tree_math

from jax_cfd.base import finite_differences as fd

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ConvectFn = Callable[[GridVariableVector], GridArrayVector]
DiffuseFn = Callable[[GridVariable, float], GridArray]
ForcingFn = Callable[[GridVariableVector], GridArrayVector]


def sum_fields(*args):
  return jax.tree.map(lambda *a: sum(a), *args)


def _wrap_term_as_vector(fun, *, name):
  return tree_math.unwrap(jax.named_call(fun, name=name))


def energy_terms(
    # density: float,
    thermal_cond: float,
    dt: float,
    grid: grids.Grid,
    # convect: Optional[ConvectFn] = None,
    # diffuse: DiffuseFn = diffusion.diffuse,
    forcing: Optional[ForcingFn] = None,
) -> Callable:
    """Returns a function that performs a time step of Navier Stokes."""
    """Returns a function that performs a time step of Navier Stokes."""
    del grid  # unused

    def convect(rhoE, v):  # pylint: disable=function-redefined
        return advection.advect_linear(rhoE, v, dt)

    def diffuse(T):
        return + thermal_cond * fd.laplacian(T)

    convection = _wrap_term_as_vector(convect, name='convection')
    diffusion_ = _wrap_term_as_vector(diffuse, name='diffusion')
    if forcing is not None:
        forcing = _wrap_term_as_vector(forcing, name='forcing')

    @tree_math.wrap
    @functools.partial(jax.named_call, name='navier_stokes_momentum')
    def _explicit_terms(rhoE, T, v):
        
        drhoE_dt = 0
        # drhoE_dt = convection(rhoE, v)
        drhoE_dt += diffusion_(T)
        if forcing is not None:
            drhoE_dt += forcing(T)
        return drhoE_dt

    def explicit_terms_with_same_bcs(rhoE, T, v):
        drhoE_dt = _explicit_terms(rhoE, T, v)
        return grids.GridVariable(drhoE_dt, T.bc) 

    return explicit_terms_with_same_bcs


# TODO(shoyer): rename this to explicit_diffusion_navier_stokes
def explicit_energy_eq(
    # density: float,
    thermal_cond: float,
    dt: float,
    grid: grids.Grid,
    # convect: Optional[ConvectFn] = None,
    # diffuse: DiffuseFn = diffusion.diffuse,
    # pressure_solve: Callable = pressure.solve_fast_diag,
    forcing: Optional[ForcingFn] = None,
    time_stepper: Callable = time_stepping.forward_euler,
    properties: Optional[dict] = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = energy_terms(
      # density=density,
      thermal_cond=thermal_cond,
      dt=dt,
      grid=grid,
    #   convect=convect,
    #   diffuse=diffuse,
      forcing=forcing)

  ode = explicit_terms
  step_fn = time_stepper(ode, dt, properties=properties)
  return step_fn
