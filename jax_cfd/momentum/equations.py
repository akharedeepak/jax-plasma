import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

# from jax_cfd.collocated import advection
# from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.momentum import time_stepping
from jax_cfd.collocated import momentum_stress
import tree_math

# from jax_cfd.base import finite_differences as fd
# from jax_cfd.collocated import boundaries

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


def momentum_terms(
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    diffuse: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    forcing: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    properties: dict = None,
) -> Callable:
    """Returns a function that performs a time step of continuity Eq."""
    del grid  # unused

    convect = _wrap_term_as_vector(convect, name='convection')
    diffuse = _wrap_term_as_vector(diffuse, name='diffusion')
    forcing = _wrap_term_as_vector(forcing, name='forcing')

    @tree_math.wrap
    @functools.partial(jax.named_call, name='navier_stokes_momentum')
    def _explicit_terms(rhov, *args):
        drhovdt  = convect(rhov, *args)
        drhovdt += diffuse(rhov, *args)
        drhovdt += forcing(rhov, *args)
        return drhovdt

    def explicit_terms_with_same_bcs(rhov, *args):
        drhovdt = _explicit_terms(rhov, *args)
        return tuple(grids.GridVariable(drhoudt, rhou.bc) for drhoudt, rhou in zip(drhovdt, rhov)) 

    return explicit_terms_with_same_bcs


# TODO(shoyer): rename this to explicit_diffusion_navier_stokes
def explicit_momentum_eq(
    state_var: str,
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    diffuse: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    forcing: Optional[Callable] = lambda rhov, *args: tuple(0*rhou.array for rhou in rhov),
    time_stepper: Callable = time_stepping.forward_euler,
    properties: dict = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = momentum_terms(
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing,
      properties=properties)

  ode = time_stepping.MomentumODE(explicit_terms, properties)
  step_fn = time_stepper(ode, state_var, dt)
  return step_fn
