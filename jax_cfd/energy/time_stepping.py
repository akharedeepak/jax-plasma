# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Time stepping for Navier-Stokes equations."""

import dataclasses
from typing import Callable, Sequence, TypeVar
import jax
import tree_math
from jax_cfd.base import grids


PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]


def state_2_primitive(state, properties):
  """Convert a state to primitive variables.
  Args:
    state: the state to convert.
    properties: a dictionary of properties, including Cv.
  Returns:
    A dictionary of primitive variables.
  """
  Cv = properties['Cv']
  
  rho, rhov, rhoE = state['rho'], state['rhov'], state['rhoE']
  v0 = tuple(grids.GridVariable(rhou0.array / rho, bc=rhou0.bc) for rhou0 in rhov)
  T0 = grids.GridVariable(rhoE.array / (Cv * rho), bc=rhoE.bc)
  primitive = dict(
    rho=rho,
    v=v0,
    T=T0,
  )
  return primitive


class ExplicitNavierStokesODE:
  """Spatially discretized version of Navier-Stokes.

  The equation is given by:

    ∂u/∂t = explicit_terms(u)
    0 = incompressibility_constraint(u)
  """

  def __init__(self, explicit_terms, pressure_projection):
    self.explicit_terms = explicit_terms
    self.pressure_projection = pressure_projection

  def explicit_terms(self, state):
    """Explicitly evaluate the ODE."""
    raise NotImplementedError

  def pressure_projection(self, state):
    """Enforce the incompressibility constraint."""
    raise NotImplementedError


@dataclasses.dataclass
class ButcherTableau:
  a: Sequence[Sequence[float]]
  b: Sequence[float]
  # TODO(shoyer): add c, when we support time-dependent equations.

  def __post_init__(self):
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")


def energy_eq_rk(
    tableau: ButcherTableau,
    equation: ExplicitNavierStokesODE,
    time_step: float,
    properties: dict = None,
) -> TimeStepFn:
  """Create a forward Runge-Kutta time-stepper for incompressible Navier-Stokes.

  This function implements the reference method (equations 16-21), rather than
  the fast projection method, from:
    "Fast-Projection Methods for the Incompressible Navier–Stokes Equations"
    Fluids 2020, 5, 222; doi:10.3390/fluids5040222

  Args:
    tableau: Butcher tableau.
    equation: equation to use.
    time_step: overall time-step size.

  Returns:
    Function that advances one time-step forward.
  """
  # pylint: disable=invalid-name
  dt = time_step
  F = tree_math.unwrap(equation)

  a = tableau.a
  b = tableau.b
  num_steps = len(b)

  # @tree_math.wrap
  def step_fn(state):
    # state = state.tree
    rho0, v0, T0 = state['rho'], state['v'], state['T']
    rhoE0 = tree_math.Vector(grids.GridVariable(rho0.array * properties['Cv'] * T0.array, T0.bc)) 
    T0, v0 = (tree_math.Vector(vi) for vi in (T0, v0))

    rhoE = [None] * num_steps
    k = [None] * num_steps

    rhoE[0] = rhoE0
    k[0] = F(rhoE0, T0, v0)

    for i in range(1, num_steps):
      rhoE[i] = rhoE0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      Ti = rhoE[i] / (rho0.data * properties['Cv'])
      k[i] = F(rhoE[i], Ti, v0)

    rhoE_final = rhoE0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])
    T_final = rhoE_final / (rho0.data * properties['Cv'])
    state['T'] = T_final.tree
    # state = tree_math.Vector(state)
    return state

  return step_fn


def forward_euler(
    equation: ExplicitNavierStokesODE, time_step: float, properties: dict = None,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          ButcherTableau(a=[], b=[1]),
          equation,
          time_step,
          properties=properties),
      name="forward_euler",
  )


def midpoint_rk2(
    equation: ExplicitNavierStokesODE, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          ButcherTableau(a=[[1/2]], b=[0, 1]),
          equation=equation,
          time_step=time_step,
      ),
      name="midpoint_rk2",
  )


def heun_rk2(
    equation: ExplicitNavierStokesODE, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          ButcherTableau(a=[[1]], b=[1/2, 1/2]),
          equation=equation,
          time_step=time_step,
      ),
      name="heun_rk2",
  )


def classic_rk4(
    equation: ExplicitNavierStokesODE, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          ButcherTableau(a=[[1/2], [0, 1/2], [0, 0, 1]],
                         b=[1/6, 1/3, 1/3, 1/6]),
          equation=equation,
          time_step=time_step,
      ),
      name="classic_rk4",
  )
