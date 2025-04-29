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


"""Prepare initial conditions for simulations."""
import functools
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax_cfd.base import boundaries
from jax_cfd.base import grids
import numpy as np

# from jax_cfd.collocated import tree_math

# Specifying the full signatures of Callable would get somewhat onerous
# pylint: disable=g-bare-generic

Array = Union[np.ndarray, jax.Array]
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
BoundaryConditions = grids.BoundaryConditions



def initial_T_field(
    rng_key: grids.Array,
    grid: grids.Grid,
    Temperature: Optional[Callable[..., Array]],
    Temperature_bc: Optional[BoundaryConditions] = None,
) -> GridVariableVector:
  """Create a random temperature field.
  Args:
    rng_key: key for seeding the random initial temperature field.
    grid: the grid on which the temperature field is defined.
    """
  if callable(Temperature):
    Temperature = grid.eval_on_mesh(Temperature, grid.cell_center)
  else:
    Temperature = grids.GridArray(Temperature, grid.cell_center, grid)

  if Temperature_bc is None:
    Temperature_bc = boundaries.neumann_boundary_conditions(grid.ndim)
  return grids.GridVariable(Temperature, bc=Temperature_bc)
