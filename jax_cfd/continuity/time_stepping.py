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


class ContinuityODE:
  """Spatially discretized version of Navier-Stokes.

  The equation is given by:

    ∂u/∂t = explicit_terms(u)
    0 = incompressibility_constraint(u)
  """

  def __init__(self, explicit_terms, properties):
    self.explicit_terms = explicit_terms
    self.properties = properties

  def explicit_terms(self, state):
    """Explicitly evaluate the ODE."""
    raise NotImplementedError



@dataclasses.dataclass
class ButcherTableau:
  a: Sequence[Sequence[float]]
  b: Sequence[float]
  # TODO(shoyer): add c, when we support time-dependent equations.

  def __post_init__(self):
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")


def continuity_eq_rk(
    state_var: str,
    tableau: ButcherTableau,
    equation: ContinuityODE,
    time_step: float,
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
  F = tree_math.unwrap(equation.explicit_terms)
  properties = equation.properties

  a = tableau.a
  b = tableau.b
  num_steps = len(b)

  # @tree_math.wrap
  def step_fn(state):
    # state = state.tree
    n0 = tree_math.Vector(state[state_var])
    state = tree_math.Vector(state)

    n = [None] * num_steps
    k = [None] * num_steps

    n[0] = n0
    k[0] = F(n0, state)

    for i in range(1, num_steps):
      n[i] = n0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      k[i] = F(n[i], state)

    n_final = n0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])
    state = state.tree
    state[state_var+'_new'] = n_final.tree
    return state

  return step_fn


def forward_euler(
    equation: ContinuityODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      continuity_eq_rk(
          state_var,
          ButcherTableau(a=[], b=[1]),
          equation,
          time_step,),
      name="forward_euler",
  )


def midpoint_rk2(
    equation: ContinuityODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      continuity_eq_rk(
          state_var,
          ButcherTableau(a=[[1/2]], b=[0, 1]),
          equation=equation,
          time_step=time_step,
      ),
      name="midpoint_rk2",
  )


def heun_rk2(
    equation: ContinuityODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      continuity_eq_rk(
          state_var,
          ButcherTableau(a=[[1]], b=[1/2, 1/2]),
          equation=equation,
          time_step=time_step,
      ),
      name="heun_rk2",
  )


def classic_rk4(
    equation: ContinuityODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      continuity_eq_rk(
          state_var,
          ButcherTableau(a=[[1/2], [0, 1/2], [0, 0, 1]],
                         b=[1/6, 1/3, 1/3, 1/6]),
          equation=equation,
          time_step=time_step,
      ),
      name="classic_rk4",
  )
