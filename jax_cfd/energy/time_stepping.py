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

import jax_cfd.collocated as col

PyTreeState = TypeVar("PyTreeState")
TimeStepFn = Callable[[PyTreeState], PyTreeState]


class EnergyODE:
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

  # def rhoE_2_T(self, rhoE, state):
  #   """convert rhoE to T."""
  #   E = rhoE.array / state['rho'].array
  #   KE = 0.5*(state['v'][0].array**2 + state['v'][1].array**2)
  #   T = grids.GridVariable((E - KE) / self.properties['Cv'], (state['T'].bc))
  #   return T
  
  # def T_2_rhoE(self, T, state):
  #   """Convert T to rhoE."""
  #   e  = self.properties['Cv'] * T.array 
  #   KE = 0.5*(state['v'][0].array**2 + state['v'][1].array**2)
  #   # TODO BC is only valide for neumann BC
  #   rhoE = grids.GridVariable(state['rho'].array * (e + KE), T.bc)
  #   return rhoE

  def rhoE_2_T(self, rhoE, state):
    """convert rhoE to T."""
    E = rhoE.array / state['rho'].array
    KE = 0.5*sum(u.array*u.array for u in state['v'])
    T = grids.GridVariable((E - KE) / self.properties['Cv'], (state['T'].bc))
    return T
  
  def T_2_rhoE(self, T, state):
    """Convert T to rhoE."""
    e    = grids.GridVariable(self.properties['Cv'] * T.array, (T.bc))
    KE   = col.conservatives._KE(state['v'])
    E    = col.conservatives._sum_GVs([e, KE])
    rhoE = col.conservatives._s1_times_s2(state['rho'], E)
    return rhoE


@dataclasses.dataclass
class ButcherTableau:
  a: Sequence[Sequence[float]]
  b: Sequence[float]
  # TODO(shoyer): add c, when we support time-dependent equations.

  def __post_init__(self):
    if len(self.a) + 1 != len(self.b):
      raise ValueError("inconsistent Butcher tableau")


def energy_eq_rk(
    state_var: str,
    tableau: ButcherTableau,
    equation: EnergyODE,
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
    T0 = state[state_var]
    rhoE0 = tree_math.Vector(equation.T_2_rhoE(T0, state)) 
    state = tree_math.Vector(state)
    T0 = tree_math.Vector(T0)

    rhoE = [None] * num_steps
    k = [None] * num_steps

    rhoE[0] = rhoE0
    k[0] = F(rhoE0, T0, state)

    for i in range(1, num_steps):
      rhoE[i] = rhoE0 + dt * sum(a[i-1][j] * k[j] for j in range(i) if a[i-1][j])
      Ti = equation.rhoE_2_T(rhoE[i].tree, state.tree)
      k[i] = F(rhoE[i], Ti, state)

    rhoE_final = rhoE0 + dt * sum(b[j] * k[j] for j in range(num_steps) if b[j])
    state = state.tree
    T_final = equation.rhoE_2_T(rhoE_final.tree, state)
    state[state_var+'_new'] = T_final
    return state

  return step_fn


def forward_euler(
    equation: EnergyODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          state_var,
          ButcherTableau(a=[], b=[1]),
          equation,
          time_step,),
      name="forward_euler",
  )


def midpoint_rk2(
    equation: EnergyODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          state_var,
          ButcherTableau(a=[[1/2]], b=[0, 1]),
          equation=equation,
          time_step=time_step,
      ),
      name="midpoint_rk2",
  )


def heun_rk2(
    equation: EnergyODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          state_var,
          ButcherTableau(a=[[1]], b=[1/2, 1/2]),
          equation=equation,
          time_step=time_step,
      ),
      name="heun_rk2",
  )


def classic_rk4(
    equation: EnergyODE, state_var: str, time_step: float,
) -> TimeStepFn:
  return jax.named_call(
      energy_eq_rk(
          state_var,
          ButcherTableau(a=[[1/2], [0, 1/2], [0, 0, 1]],
                         b=[1/6, 1/3, 1/3, 1/6]),
          equation=equation,
          time_step=time_step,
      ),
      name="classic_rk4",
  )
