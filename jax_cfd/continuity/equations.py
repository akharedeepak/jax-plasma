import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from jax_cfd.collocated import advection
from jax_cfd.base import diffusion
from jax_cfd.base import grids
from jax_cfd.continuity import time_stepping
from jax_cfd.collocated import momentum_stress
import tree_math

from jax_cfd.base import finite_differences as fd
from jax_cfd.collocated import boundaries

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


def continuity_terms(
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = lambda n, *args: 0*n.array,
    diffuse: Optional[Callable] = lambda n, *args: 0*n.array,
    forcing: Optional[Callable] = lambda n, *args: 0*n.array,
    properties: dict = None,
) -> Callable:
    """Returns a function that performs a time step of Navier Stokes."""
    del grid  # unused

    convect = _wrap_term_as_vector(convect, name='convection')
    diffuse = _wrap_term_as_vector(diffuse, name='diffusion')
    forcing = _wrap_term_as_vector(forcing, name='forcing')

    @tree_math.wrap
    @functools.partial(jax.named_call, name='navier_stokes_momentum')
    def _explicit_terms(n, v):
        dn_dt  = convect(n, v)
        dn_dt += diffuse(n)
        dn_dt += forcing(n)
        return dn_dt

    def explicit_terms_with_same_bcs(n, v):
        dn_dt = _explicit_terms(n, v)
        return grids.GridVariable(dn_dt, n.bc) 

    return explicit_terms_with_same_bcs


# TODO(shoyer): rename this to explicit_diffusion_navier_stokes
def explicit_continuity_eq(
    state_var: str,
    dt: float,
    grid: grids.Grid,
    convect: Optional[Callable] = lambda n, *args: 0*n.array,
    diffuse: Optional[Callable] = lambda n, *args: 0*n.array,
    forcing: Optional[Callable] = lambda n, *args: 0*n.array,
    time_stepper: Callable = time_stepping.forward_euler,
    properties: dict = None,
) -> Callable[[GridVariableVector], GridVariableVector]:
  """Returns a function that performs a time step of Navier Stokes."""

  explicit_terms = continuity_terms(
      dt=dt,
      grid=grid,
      convect=convect,
      diffuse=diffuse,
      forcing=forcing,
      properties=properties)

  ode = explicit_terms
  step_fn = time_stepper(ode, state_var, dt, properties=properties)
  return step_fn
