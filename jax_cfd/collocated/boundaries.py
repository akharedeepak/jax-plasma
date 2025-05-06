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

# Extensively modified and expanded by CoMSAIL Lab (2023)
# (https://sites.nd.edu/jianxun-wang/research/)
# Authors: Xiantao Fan and Jian-Xun Wang
# Modifications include significant refactoring of the original code and
# the additon of new features and functionalities

"""Classes that specify how boundary conditions are applied to arrays."""

import dataclasses
from typing import Sequence, Tuple, Optional, Union
import numpy as np

from jax import lax
import jax
import jax.numpy as jnp

# CHECK: this import for collocation
# from Jax_FSI.collocation import grids
# from Jax_FSI.src import wall_model
from jax_cfd.base import grids
from jax_cfd.collocated import finite_differences as fd
# from Jax_FSI.src import out_flow_boundary
import copy

BoundaryConditions = grids.BoundaryConditions
GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
Array = Union[np.ndarray, jax.Array]


class BCType:
    PERIODIC = 'periodic'
    DIRICHLET = 'dirichlet'
    NEUMANN = 'neumann'


# Revised by X. Fan
@dataclasses.dataclass(init=False, frozen=True)
class ConstantBoundaryConditions(BoundaryConditions):
    """Boundary conditions for a PDE variable that are constant in space and time.

    Example usage:
      grid = Grid((10, 10))
      array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
      bc = ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)),
                                          ((0.0, 10.0),(1.0, 0.0)))
      u = GridVariable(array, bc)

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """
    types: Tuple[Tuple[str, str], ...]
    _values: Tuple[Tuple[Optional[float], Optional[float]], ...]

    def __init__(self, types: Sequence[Tuple[str, str]],
                 values: Sequence[Tuple[Optional[float], Optional[float]]],
                 wall=False,
                 hm=None
                 ):
        types = tuple(types)
        values = tuple(values)
        object.__setattr__(self, 'types', types)
        object.__setattr__(self, '_values', values)
        object.__setattr__(self, '_wall', wall)
        object.__setattr__(self, '_hm', hm)

    def shift(
        self,
        u: GridArray,
        offset: int,
        axis: int,
        t: Optional[float] = None
    ) -> GridArray:
        """Shift an GridArray by `offset`.

        Args:
          u: an `GridArray` object.
          offset: positive or negative integer offset to shift.
          axis: axis to shift along.

        Returns:
          A copy of `u`, shifted by `offset`. The returned `GridArray` has offset
          `u.offset + offset`.
        """
        padded = self._pad(u, offset, axis, t)
        trimmed = self._trim(padded, -offset, axis, t)
        return trimmed

    def _pad(
        self,
        u: GridArray,
        width: int,
        axis: int,
        t: Optional[float] = None
    ) -> GridArray:
        """Pad a GridArray by `padding`.

        Important: _pad makes no sense past 1 ghost cell for nonperiodic
        boundaries. This is sufficient for jax_fsi finite difference code.

        Args:
          u: a `GridArray` object.
          width: number of elements to pad along axis. Use negative value for lower
            boundary or positive value for upper boundary.
          axis: axis to pad along.

        Returns:
          Padded array, elongated along the indicated axis.
        """
        if width == 0:

            return u

        if width < 0:  # pad lower boundary
            bc_type = self.types[axis][0]
            padding = (-width, 0)
        else:  # pad upper boundary
            bc_type = self.types[axis][1]
            padding = (0, width)

        full_padding = [(0, 0)] * u.grid.ndim
        full_padding[axis] = padding

        offset = list(u.offset)
        offset[axis] -= padding[0]

        if bc_type != BCType.PERIODIC and abs(width) > 1:
            raise ValueError(
                'Padding past 1 ghost cell is not defined in nonperiodic case.')

        if bc_type == BCType.PERIODIC:
            # self.values are ignored here
            pad_kwargs = dict(mode='wrap')
        elif bc_type == BCType.DIRICHLET:
            if np.isclose(u.offset[axis], 0.5):  # cell center
                # make the linearly interpolated value equal to the boundary by setting
                # the padded values to the negative symmetric values
                if callable(self._values[axis][np.where(width < 0, 0, -1)]):
                    bc_value = self._values[axis][np.where(
                        width < 0, 0, -1)](u, t)
                else:
                    bc_value = self._values[axis][np.where(width < 0, 0, -1)]
                    shape = jax.lax.index_in_dim(u.data, np.where(
                        width < 0, 0, -1), axis=axis, keepdims=True)
                    bc_value = jnp.ones_like(shape)*bc_value

                pad_data = 2 * bc_value - \
                    jax.lax.index_in_dim(u.data, np.where(
                        width < 0, 0, -1), axis=axis, keepdims=True)

                if width < 0:
                    data = jnp.append(pad_data, u.data, axis=axis)
                else:
                    data = jnp.append(u.data, pad_data, axis=axis)
                return GridArray(data, tuple(offset), u.grid)

            # offset one grid of cell centre, especially for non-periodic boundary
            # after linear interpolate, same as cell edge but for clarify
            elif np.isclose(u.offset[axis], 1.5):

                if callable(self._values[axis][np.where(width < 0, 0, -1)]):
                    bc_value = self._values[axis][np.where(
                        width < 0, 0, -1)](u.grid, t)
                else:
                    bc_value = self._values[axis][np.where(width < 0, 0, -1)]
                    shape = jax.lax.index_in_dim(u.data, np.where(
                        width < 0, 0, -1), axis=axis, keepdims=True)
                    bc_value = jnp.ones_like(shape)*bc_value

                pad_data = bc_value
                if width < 0:
                    data = jnp.append(pad_data, u.data, axis=axis)
                else:
                    data = jnp.append(u.data, pad_data, axis=axis)
                return GridArray(data, tuple(offset), u.grid)
            elif np.isclose(u.offset[axis] % 1, 0):  # cell edge
                # initial the boundary vel and noise
                if callable(self._values[axis][np.where(width < 0, 0, -1)]):
                    bc_value = self._values[axis][np.where(
                        width < 0, 0, -1)](u.grid, t)
                else:
                    bc_value = self._values[axis][np.where(width < 0, 0, -1)]
                    shape = jax.lax.index_in_dim(u.data, np.where(
                        width < 0, 0, -1), axis=axis, keepdims=True)
                    bc_value = jnp.ones_like(shape)*bc_value

                pad_data = bc_value
                if width < 0:
                    data = jnp.append(pad_data, u.data, axis=axis)
                else:
                    data = jnp.append(u.data, pad_data, axis=axis)
                return GridArray(data, tuple(offset), u.grid)
            else:
                raise ValueError('expected offset to be an edge or cell center, got '
                                 f'offset[axis]={u.offset[axis]}')
        elif bc_type == BCType.NEUMANN:
            if not (np.isclose(u.offset[axis] % 1, 0) or
                    np.isclose(u.offset[axis] % 1, 0.5)):
                raise ValueError('expected offset to be an edge or cell center, got '
                                 f'offset[axis]={u.offset[axis]}')
            else:
                # When the data is cell-centered, computes the backward difference.
                # When the data is on cell edges, boundary is set such that
                # (u_last_interior - u_boundary)/grid_step = neumann_bc_value.
                if callable(self._values[axis][np.where(width < 0, 0, -1)]):
                    bc_value = self._values[axis][np.where(
                        width < 0, 0, -1)](u.grid, t)
                else:
                    bc_value = self._values[axis][np.where(width < 0, 0, -1)]
                    shape = jax.lax.index_in_dim(u.data, np.where(
                        width < 0, 0, -1), axis=axis, keepdims=True)
                    bc_value = jnp.ones_like(shape)*bc_value

                if u.grid.stretch is None:
                    step = u.grid.step[axis]
                else:
                    step = u.grid.step[axis][-1] if width > 0 else u.grid.step[axis][0]

                if np.isclose(u.offset[axis] % 1, 0):  # cell edge
                    u_last_interior = jax.lax.index_in_dim(u.data, np.where(width < 0, 1, -2), axis=axis, keepdims=True) \
                        # + 0.5 * \
                    # jax.lax.index_in_dim(u.data, np.where(
                    #     width < 0, -1, -2), axis=axis, keepdims=True
                    step = step * 2
                elif np.isclose(u.offset[axis] % 1, 0.5):  # cell centre
                    u_last_interior = jax.lax.index_in_dim(
                        u.data, np.where(width < 0, 0, -1), axis=axis, keepdims=True)

                pad_data = u_last_interior-bc_value * step

                if width < 0:
                    data = jnp.append(pad_data, u.data, axis=axis)
                else:
                    data = jnp.append(u.data, pad_data, axis=axis)
                return GridArray(data, tuple(offset), u.grid)

        else:
            raise ValueError('invalid boundary type')

        data = jnp.pad(u.data, full_padding, **pad_kwargs)
        return GridArray(data, tuple(offset), u.grid)

    def _trim(
        self,
        u: GridArray,
        width: int,
        axis: int,
        t: Optional[float] = None
    ) -> GridArray:
        """Trim padding from a GridArray.

        Args:
          u: a `GridArray` object.
          width: number of elements to trim along axis. Use negative value for lower
            boundary or positive value for upper boundary.
          axis: axis to trim along.

        Returns:
          Trimmed array, shrunk along the indicated axis.
        """
        if width < 0:  # trim lower boundary
            padding = (-width, 0)
        else:  # trim upper boundary
            padding = (0, width)

        limit_index = u.data.shape[axis] - padding[1]
        data = lax.slice_in_dim(u.data, padding[0], limit_index, axis=axis)
        offset = list(u.offset)
        offset[axis] += padding[0]
        return GridArray(data, tuple(offset), u.grid)

    def values(
        self,
        u: GridArray,
        axis: int,
        grid: grids.Grid,
            t: Optional[float] = None) -> Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]]:
        """Returns boundary values on the grid along axis.

        Args:
          axis: axis along which to return boundary values.
          grid: a `Grid` object on which to evaluate boundary conditions.

        Returns:
          A tuple of arrays of grid.ndim - 1 dimensions that specify values on the
          boundary. In case of periodic boundaries, returns a tuple(None,None).
        """
        bc_end = list((0, 0))
        pad_data = [(0, 0)]*u.grid.ndim
        # _pad_data = [(0, 0)]*u.grid.ndim
        x = axis
        # pad_data_ = []
        boundary_value = []
        for y in range(2):
            if callable(self._values[x][y]):

                if self._wall is False:
                    boundary_data = self._values[x][y](
                        u.grid, t)
                elif self._wall is True:
                    loc = np.where(y == 0, self._hm, -self._hm-1)
                    boundary_data = self._values[x][y](u.grid,  t)

                boundary_value.append(boundary_data)
            else:
                boundary_value.append(self._values[x][y])

        pad_data[x] = boundary_value

        for ver in range(2):
            if self.types[axis][ver] == 'dirichlet':
                if None in pad_data[axis]:
                    return (None, None)
                else:
                    data = pad_data[axis][-ver] if (type(pad_data[axis][-ver]) is int) or (
                        type(pad_data[axis][-ver]) is float) else pad_data[axis][-ver].squeeze()

                    bc = jnp.full(
                        grid.shape[:axis] + grid.shape[axis + 1:], data)

            if self.types[axis][ver] == 'neumann':
                if None in pad_data[axis]:
                    return (None, None)
                if axis == 0:
                    if self._wall is False:
                        bc = u.data[jnp.where(ver == 0, 1, -2), :]
                    elif self._wall is True:
                        bc = pad_data[x][ver]
                elif axis == 1:
                    if self._wall is False:
                        bc = u.data[:, jnp.where(ver == 0, 1, -2)]
                    elif self._wall is True:
                        bc = pad_data[x][ver]

            bc_end[ver] = bc
        return tuple(bc_end)

    trim = _trim
    pad = _pad


class HomogeneousBoundaryConditions(ConstantBoundaryConditions):
    """Homogeneous Boundary conditions for a PDE variable.

    Example usage:
      grid = Grid((10, 10))
      array = GridArray(np.zeros((10, 10)), offset=(0.5, 0.5), grid)
      bc = HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                          (BCType.DIRICHLET, BCType.DIRICHLET)))
      u = GridVariable(array, bc)

    Attributes:
      types: `types[i]` is a tuple specifying the lower and upper BC types for
        dimension `i`.
    """

    def __init__(self, types: Sequence[Tuple[str, str]]):

        ndim = len(types)
        values = ((0.0, 0.0),) * ndim
        super(HomogeneousBoundaryConditions, self).__init__(types, values)


def periodic_boundary_conditions(ndim: int) -> BoundaryConditions:
    """Returns periodic BCs for a variable with `ndim` spatial dimension."""
    return HomogeneousBoundaryConditions(
        ((BCType.PERIODIC, BCType.PERIODIC),) * ndim)


def dirichlet_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Dirichelt BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spacial dimension.
      bc_vals: if None, assumes Homogeneous BC. Else needs to be a tuple of lower
        and upper boundary values for each dimension. If None, returns Homogeneous
        BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim)
    else:
        return ConstantBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),) * ndim, bc_vals)


def neumann_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
) -> BoundaryConditions:
    """Returns Neumann BCs for a variable with `ndim` spatial dimension.

    Args:
      ndim: spacial dimension.
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.NEUMANN, BCType.NEUMANN),) * ndim)
    else:
        return ConstantBoundaryConditions(
            ((BCType.NEUMANN, BCType.NEUMANN),) * ndim, bc_vals)


def periodic_and_dirichlet_boundary_conditions(
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                              (BCType.DIRICHLET, BCType.DIRICHLET)))
    else:
        return ConstantBoundaryConditions(((BCType.PERIODIC, BCType.PERIODIC),
                                           (BCType.DIRICHLET, BCType.DIRICHLET)),
                                          ((None, None), bc_vals))


def periodic_and_neumann_boundary_conditions(
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs periodic for dimension 0 and Neumann for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals:
        return HomogeneousBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)))
    else:
        return ConstantBoundaryConditions(
            ((BCType.PERIODIC, BCType.PERIODIC), (BCType.NEUMANN, BCType.NEUMANN)),
            ((None, None), bc_vals))


def has_all_periodic_boundary_conditions(*arrays: GridVariable) -> bool:
    """Returns True if arrays have periodic BC in every dimension, else False."""
    for array in arrays:
        for lower_bc_type, upper_bc_type in array.bc.types:
            if lower_bc_type != BCType.PERIODIC or upper_bc_type != BCType.PERIODIC:
                return False
    return True


# Added by Xiantao Fan
def convective_outflow_boundary() -> BoundaryConditions:
    """du/dt+Uconvect*▽u=0, where Uconvect is the mean velocity,
    Use backward difference for the convective term

    see Jax_FSI/src/out_flow_boundary.py
    """

    return 'Done by Xiantao Fan'


def inflow_outflow_periodic_boundary_conditions(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and periodic for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """

    bc_type = ((BCType.DIRICHLET, BCType.NEUMANN),
               (BCType.PERIODIC, BCType.PERIODIC))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals)


def inflow_outflow_noslip_periodic_boundary_conditions(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and periodic for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """

    bc_type = ((BCType.DIRICHLET, BCType.NEUMANN),
               (BCType.DIRICHLET, BCType.DIRICHLET))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals)


def inflow_outflow_Noslip_boundary_conditions(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and Noslip for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
               (BCType.DIRICHLET, BCType.DIRICHLET))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(
            (bc_type),
            (bc_vals))


def inflow_outflow_stress_free_boundary_conditions(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and Noslip for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    bc_type = ((BCType.DIRICHLET, BCType.NEUMANN),
               (BCType.DIRICHLET, BCType.DIRICHLET))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(
            (bc_type),
            (bc_vals))


def inflow_outflow_stress_free_boundary_conditions_2(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and Noslip for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
               (BCType.DIRICHLET, BCType.DIRICHLET))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(
            (bc_type),
            (bc_vals))


def periodic_Noslip_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
    wall: Optional[float] = False
) -> ConstantBoundaryConditions:
    """Returns BCs periodic for dimension 0 and Dirichlet for dimension 1.
    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC. For periodic dimensions the lower, upper
        boundary values should be (None, None).
    Returns:
      BoundaryCondition instance.
    """
    bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
               (BCType.DIRICHLET, BCType.DIRICHLET))
    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals, wall)


def periodic_flat_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
    wall: Optional[float] = False,
) -> ConstantBoundaryConditions:
    """Returns BCs periodic for dimension 0 and equilibrium wall boundary for dimension 1.
    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC. For periodic dimensions the lower, upper
        boundary values should be (None, None).
    Returns:
      BoundaryCondition instance.
    """
    bc_type = ((BCType.PERIODIC, BCType.PERIODIC),
               (BCType.DIRICHLET, BCType.NEUMANN))
    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals, wall)


def inflow_outflow_flat_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
    wall: Optional[float] = False,
) -> ConstantBoundaryConditions:
    """Returns BCs periodic for dimension 0 and equilibrium wall boundary for dimension 1.
    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC. For periodic dimensions the lower, upper
        boundary values should be (None, None).
    Returns:
      BoundaryCondition instance.
    """
    bc_type = ((BCType.DIRICHLET, BCType.DIRICHLET),
               (BCType.DIRICHLET, BCType.NEUMANN))
    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals, wall)


def inflow_outflow_zero_flat_boundary_conditions(
    ndim: int,
    bc_vals: Optional[Sequence[Tuple[float, float]]] = None,
    wall: Optional[float] = False,
) -> ConstantBoundaryConditions:
    """Returns BCs periodic for dimension 0 and equilibrium wall boundary for dimension 1.
    Args:
      ndim: spatial dimension.
      bc_vals: A tuple of lower and upper boundary values for each dimension.
        If None, returns Homogeneous BC. For periodic dimensions the lower, upper
        boundary values should be (None, None).
    Returns:
      BoundaryCondition instance.
    """
    bc_type = ((BCType.DIRICHLET, BCType.NEUMANN),
               (BCType.DIRICHLET, BCType.NEUMANN))
    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)
    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals, wall)


def inflow_outflow_Freeslip_boundary_conditions(
        ndim: int,
        bc_vals: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs inflow and outflow for dimension 0(x) and freeslip for dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    bc_type = ((BCType.DIRICHLET, BCType.NEUMANN),
               (BCType.NEUMANN, BCType.NEUMANN))

    for _ in range(ndim - 2):
        bc_type += ((BCType.PERIODIC, BCType.PERIODIC),)

    if not bc_vals:
        return HomogeneousBoundaryConditions(bc_type)
    else:
        return ConstantBoundaryConditions(bc_type, bc_vals)


def cavity_flow_boundary_conditions(
        bc_vals_1: Optional[Tuple[float, float]] = None,
        bc_vals_2: Optional[Tuple[float, float]] = None) -> BoundaryConditions:
    """Returns BCs with different velocity for dimension 0(x) and dimension 1.

    Args:
      bc_vals: the lower and upper boundary condition value for each dimension. If
        None, returns Homogeneous BC.

    Returns:
      BoundaryCondition subclass.
    """
    if not bc_vals_1:
        return HomogeneousBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET), (BCType.DIRICHLET, BCType.DIRICHLET)))
    else:
        return ConstantBoundaryConditions(
            ((BCType.DIRICHLET, BCType.DIRICHLET),
             (BCType.DIRICHLET, BCType.DIRICHLET)),
            (bc_vals_1, bc_vals_2))


def get_pressure_bc_from_velocity(v: GridVariableVector) -> BoundaryConditions:
    """Returns pressure boundary conditions for the specified velocity."""
    # Expect each component of v to have the same BC, either both PERIODIC or
    # both DIRICHLET.
    velocity_bc_types = grids.consistent_boundary_conditions(*v).types
    velocity_bc_values = grids.consistent_boundary_conditions(*v)._values
    pressure_bc_types = []
    pressure_bc_values = []

    for i in range(v[0].grid.ndim):
        bc_value = []
        bc_type = []
        for j in range(2):
            if velocity_bc_types[i][j] == BCType.PERIODIC:
                bc_type.append(BCType.PERIODIC)
                bc_value.append(0.0)

            elif velocity_bc_types[i][j] == BCType.DIRICHLET:

                if callable(velocity_bc_values[i][j]) is True and j == 0:
                    bc_type.append(BCType.NEUMANN)
                    bc_value.append(
                        lambda x, y: out_flow_boundary.inflow_boundary_pressure(y['velocity'], y))
                elif callable(velocity_bc_values[i][j]) is True and j == 1:
                    bc_type.append(BCType.NEUMANN)
                    bc_value.append(
                        lambda x, y: out_flow_boundary.convective_outflow_boundary_pressure(y['velocity'], y))
                else:
                    bc_type.append(BCType.NEUMANN)
                    bc_value.append(0.0)

            elif velocity_bc_types[i][j] == BCType.NEUMANN:
                bc_type.append(BCType.DIRICHLET)
                bc_value.append(0.0)
            else:
                raise ValueError('Expected periodic or dirichlete velocity BC, '
                                 f'got {velocity_bc_types[i][j]}')

        pressure_bc_types.append(tuple(bc_type))
        pressure_bc_values.append(tuple(bc_value))

    if v[0].grid.ndim == 2:
        bound = ConstantBoundaryConditions(
            pressure_bc_types, pressure_bc_values)
    elif v[0].grid.ndim == 3:
        bound = ConstantBoundaryConditions(
            pressure_bc_types, pressure_bc_values)

    return bound


def get_advection_flux_bc_from_velocity_and_scalar(
    u: GridVariable, c: GridVariable,
    flux_direction: int,
    original_v: GridVariable,
    t: dict = {}) -> ConstantBoundaryConditions:
    """Returns advection flux boundary conditions for the specified velocity.

    Infers advection flux boundary condition in flux direction
    from scalar c and velocity u in direction flux_direction.

    Args:
      u: velocity component in flux_direction.
      c: scalar to advect.
      flux_direction: direction of velocity.

    Returns:
      BoundaryCondition instance for advection flux of c in flux_direction.
    """
    # only no penetration and periodic boundaries are supported.
    flux_bc_types = []
    flux_bc_values = []

    for axis in range(c.grid.ndim):
        if u.bc.types[axis][0] == 'periodic':
            flux_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
            flux_bc_values.append((0, 0))
        elif flux_direction != axis:
            """If the velocity is not in the flux direction, the flux is zero. This is a fake boundary condition."""
            flux_bc_types.append((BCType.DIRICHLET, BCType.DIRICHLET))
            flux_bc_values.append((0.0, 0.0))
        else:
            flux_bc_types_ax = []
            flux_bc_values_ax = []
            for i in range(2):  # lower and upper boundary.

                # case 1: constant boundary
                if (u.bc.types[axis][i] == BCType.DIRICHLET and
                        c.bc.types[axis][i] == BCType.DIRICHLET):
                    flux_bc_types_ax.append(BCType.DIRICHLET)
                    if np.isclose(u.offset[axis], 1.5) and np.isclose(c.offset[axis], 1.5):

                        if callable(u.bc._values[axis][i]):
                            bc_value = original_v.bc._values[axis][i](
                                original_v.grid, t)

                        else:
                            bc_value = original_v.bc._values[axis][i]

                        if i == 0:
                            data_u = 0.5*bc_value +\
                                0.5 * \
                                jax.lax.index_in_dim(
                                    original_v.data, 0, axis=axis, keepdims=True)
                        elif i == 1:
                            data_u = 0.5*bc_value +\
                                0.5 * \
                                jax.lax.index_in_dim(
                                    original_v.data, -1, axis=axis, keepdims=True)

                        data_c = data_u
                        flux_bc_values_ax.append(lambda x, t: data_u*data_c)
                    else:

                        # if callable(u.bc._values[axis][i]):
                        #     data_u = u.bc._values[axis][i]
                        #     data_c = c.bc._values[axis][i]
                        #     flux_bc_values_ax.append(
                        #         data_u(u.grid, t)*data_c(c.grid, t))
                        # Deepak
                        if callable(u.bc._values[axis][i]) or callable(c.bc._values[axis][i]):
                            data_u = u.bc._values[axis][i](u, t) if callable(u.bc._values[axis][i]) else u.bc._values[axis][i]
                            data_c = c.bc._values[axis][i](c, t) if callable(c.bc._values[axis][i]) else c.bc._values[axis][i]
                            flux_bc_values_ax.append(data_u * data_c)
                        else:
                            data_u = u.bc._values[axis][i]
                            data_c = c.bc._values[axis][i]
                            flux_bc_values_ax.append(data_u*data_c)

                elif (u.bc.types[axis][i] == BCType.NEUMANN and
                      c.bc.types[axis][i] == BCType.NEUMANN):
                    if not isinstance(c.bc, ConstantBoundaryConditions):
                        raise NotImplementedError(
                            'Flux boundary condition is not implemented for scalar' +
                            f' with {c.bc}')
                    if not np.isclose(c.bc._values[axis][i], 0.0):
                        raise NotImplementedError(
                            'Flux boundary condition is not implemented for scalar' +
                            f' with {c.bc}')
                    flux_bc_types_ax.append(BCType.NEUMANN)
                    flux_bc_values_ax.append(0.0)

                else:
                    flux_bc_types_ax.append(c.bc.types[axis][i])
                    flux_bc_values_ax.append(c.bc._values[axis][i])

            flux_bc_types.append(tuple(flux_bc_types_ax))
            flux_bc_values.append(tuple(flux_bc_values_ax))
    return ConstantBoundaryConditions(flux_bc_types, flux_bc_values)


def get_u_star_bc_from_velocity_and_pressure(u: GridVariable, p: GridVariable,
                                             t: dict) -> ConstantBoundaryConditions:
    """
    Returns boundary conditions for the intermediate velocity field u_star.

    Bcs:
    Since the divergence of u_star will be calcualted by backward finite difference, only the low boudary
    is valid.
    Usually the low bc of u_star is dirichlet or periodic.
    If it is periodic, keep the same bc as velocity
    If it is dirichlet:
    u_star = u_{t+1} + dt*grad(p_t)
    """
    u_star_bc_types = []
    u_star_bc_values = []

    grid = grids.consistent_grid(u)

    for axis in range(grid.ndim):
        if u.bc.types[axis][0] == 'periodic':
            u_star_bc_types.append((BCType.PERIODIC, BCType.PERIODIC))
            u_star_bc_values.append((0, 0))
        else:
            u_star_bc_types_ax = []
            u_star_bc_values_ax = []
            for i in range(2):
                if u.bc.types[axis][i] == BCType.DIRICHLET:
                    u_star_bc_types_ax.append(BCType.DIRICHLET)
                    gradp_bc = p.bc._values[axis][i](p.grid, t) if callable(
                        p.bc._values[axis][i]) else p.bc._values[axis][i]

                    if callable(u.bc._values[axis][i]):
                        bc_value = u.bc._values[axis][i](
                            u.grid, t)+t['dt']*gradp_bc / t['density']

                    else:
                        bc_value = u.bc._values[axis][i]+t['dt']*gradp_bc / t['density']
                    u_star_bc_values_ax.append(bc_value)

                elif u.bc.types[axis][i] == BCType.NEUMANN:
                        u_star_bc_types_ax.append(BCType.NEUMANN)
                        if callable(u.bc._values[axis][i]):
                            bc_value = u.bc._values[axis][i](u.grid, t)

                        else:
                            bc_value = u.bc._values[axis][i]
                        u_star_bc_values_ax.append(bc_value)

                else:
                    raise ValueError('Expected periodic, neumann or dirichlete velocity BC, '
                                     f'got {u.bc.types[axis][i]}')
            u_star_bc_types.append(tuple(u_star_bc_types_ax))
            u_star_bc_values.append(tuple(u_star_bc_values_ax))

    return ConstantBoundaryConditions(u_star_bc_types, u_star_bc_values)


# Added by Xiantao Fan, for implicit update
def get_implicit_delat_u_bc(u_new: GridArray, u0: GridVariable, t: Optional[float] = None) -> GridArray:
    """Get the boundary condition of the implicit update."""
    # u_new = grids.GridVariable(u_new, u0.bc)
    # u0 = u0.enforce_edge_bc(t)
    delta_u_bc_types_all = []
    delta_u_bc_values_all = []

    grid = grids.consistent_grid(u0)
#   t_old = t.copy()
#   t_old['t'] = t_old['t'] - t_old['dt']

    for axis in range(grid.ndim):
        delta_u_bc_types = []
        delta_u_bc_values = []
        for i in range(2):
            if u0.bc.types[axis][i] == 'periodic':
                delta_u_bc_types.append('periodic')
                delta_u_bc_values.append(0)
            elif u0.bc.types[axis][i] == 'neumann':
                delta_u_bc_types.append('neumann')
                if callable(u0.bc._values[axis][i]):
                    bc_value = u0.bc._values[axis][i](grid, t)
                else:
                    bc_value = u0.bc._values[axis][i]
                delta_u_bc_values.append(bc_value)
            elif u0.bc.types[axis][i] == 'dirichlet':
                delta_u_bc_types.append('dirichlet')
                if callable(u0.bc._values[axis][i]):
                    bc_value_new = u0.bc._values[axis][i](grid, t)
                    bc_value_old = u0.bc._values[axis][i](grid, t)
                    bc_value = bc_value_new - bc_value_old
                else:
                    bc_value = u0.bc._values[axis][i] - u0.bc._values[axis][i]
                delta_u_bc_values.append(bc_value)
        delta_u_bc_types_all.append(delta_u_bc_types)
        delta_u_bc_values_all.append(delta_u_bc_values)
    return ConstantBoundaryConditions(delta_u_bc_types_all, delta_u_bc_values_all)


def set_bc_tau_ax(grid, tau, vis_centre, i, j, vis_wall, wall, wall_axis=1):
    if i == 0 and j == 1:
        coe = [0, 1]
        index = 0
    elif i == 1 and j == 0:
        coe = [0, 1]
        index = 0
    elif i == 1 and j == 2:
        coe = [0, 1]
        index = 1
    elif i == 2 and j == 1:
        coe = [0, 1]
        index = 1
    else:
        coe = [1, 0]
        index = 0  # it dosen't matter

    bc_type_sum = []
    bc_value_sum = []
    for axis in range(grid.ndim):
        bc_type = []
        bc_value = []
        # for k in range(0,2):
        if tau.bc.types[axis][0] != "periodic":

            s_ij_bc_value0 = tau.bc._values[axis][0]
            s_ij_bc_value1 = tau.bc._values[axis][1]

            if wall is True:
                vt_bc_value0 = (coe[0]*jax.lax.index_in_dim(vis_centre.data, 0, axis=axis, keepdims=True)
                                + coe[1]*jax.lax.index_in_dim(vis_wall[index].data, 0, axis=axis, keepdims=True))
                vt_bc_value1 = (coe[0]*jax.lax.index_in_dim(vis_centre.data, -1, axis=axis, keepdims=True)
                                + coe[1]*jax.lax.index_in_dim(vis_wall[index].data, -1, axis=axis, keepdims=True))

            else:
                # tau at the wall should be zero
                if np.isclose(tau.offset[wall_axis], 1):
                    coe_ = 0
                else:
                    coe_ = 1
                vt_bc_value0 = coe_*jax.lax.index_in_dim(
                    vis_centre.data, 0, axis=axis, keepdims=True)
                vt_bc_value1 = coe_*jax.lax.index_in_dim(
                    vis_centre.data, -1, axis=axis, keepdims=True)

            bc_value.append(
                (2*s_ij_bc_value0*vt_bc_value0, 2*s_ij_bc_value1*vt_bc_value1))

            bc_type.append(tau.bc.types[axis])

        else:
            bc_type.append(tau.bc.types[axis])
            bc_value.append(tau.bc._values[axis])
        bc_type_sum.append(*bc_type)
        bc_value_sum.append(*bc_value)

    new_bc = ConstantBoundaryConditions(
        tuple(bc_type_sum), tuple(bc_value_sum))
    return new_bc


def reset_bc(bc):
    """
    Reset the boundary condition to the with fixed value

    This is only valid for boundary conditions to prevent recursive function calls

    """
    type_bc = bc.types

    values = ((0, 0), (0, 0), (0, 0))
    return ConstantBoundaryConditions(type_bc, values)

# get bc for s_ij from nonperiodic velocity


def get_low_sij_bc_value(u, axis, grad_axis, ver, t=None):
    """
    get the low boundary value for s_ij
    """
    grid = u.grid
    u_bc = u.bc._values[axis][ver](grid, t) if callable(
        u.bc._values[axis][ver]) else u.bc._values[axis][ver]

    if type(u_bc) == int or type(u_bc) == float:
        du = 0
    else:
        u_bc_variable = grids.GridVariable(
            grids.GridArray(u_bc, u.offset, grid), reset_bc(u.bc))
        du = jax.lax.index_in_dim(fd.forward_difference(
            u_bc_variable, grad_axis, t).data, 0, axis=axis, keepdims=True)

    return 0.5*du


def set_bc_sij_ax(grid, s_ij_back, s_ij, v, i, j, t, wall_axis=1):
    """Returns boundary conditions for strain rate s_ij based on the velocity.

    Here, we only need the low boundary value for s_ij, since the divergence of s_ij is calculated by backward_difference.

    if bc type is periodic, keep the same bc as velocity
    if bc type is dirichlet, parallel direction is zero, normal direction is nonzero
    if bc type is neumann, normal direction is the specified vaule, others are calculated
    This is done by backward_difference, should extract the low boundary value as the value for real s_ij

    Args:
      s_ij: calculated s_ij based on forward_difference method, provide upper wall boundary.
      s_ij_back: calculate s_ij based on backward_difference method, provide lower wall boundary.
      u: velocity component.
      wall_axis: direction of wall normal direction.
      i,j: the index of S_ij.

    Returns:
      BoundaryCondition instance for s_ij.
    """
    bc_type_sum = v[0].bc.types
    du_bc_x = [0, 0]
    du_bc_y = [0, 0]
    du_bc_z = [0, 0]
    if i == j == 0:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis,  grad_axis=0, ver=0, t=t) +\
                    get_low_sij_bc_value(v[1], axis, grad_axis=0, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=0, ver=0, t=t) +\
                    get_low_sij_bc_value(v[1], axis, grad_axis=0, ver=0, t=t)

    elif i == 0 and j == 1:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 1:
                du_bc_y[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) +\
                    get_low_sij_bc_value(
                        v[1], axis=axis, grad_axis=0, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=0, ver=0, t=t)

    elif i == 0 and j == 2:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=0, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = jax.lax.index_in_dim(s_ij_back[0].data, 1, axis=axis, keepdims=True) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=0, ver=0, t=t)

    elif i == 1 and j == 0:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=1, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=0, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=1, ver=0, t=t)

    elif i == 1 and j == 1:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=1, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=1, ver=0, t=t)

    elif i == 1 and j == 2:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

            elif axis == 2:
                du_bc_z[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=1, ver=0, t=t)

    elif i == 2 and j == 0:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=0, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)
    elif i == 2 and j == 1:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = get_low_sij_bc_value(v[0], axis, grad_axis=1, ver=0, t=t) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)
    elif i == 2 and j == 2:
        for axis in range(grid.ndim):
            if axis == 0:
                du_bc_x[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 1:
                du_bc_y[0] = get_low_sij_bc_value(v[0], axis, grad_axis=2, ver=0, t=t) + \
                    get_low_sij_bc_value(v[1], axis, grad_axis=2, ver=0, t=t)

            elif axis == 2:
                du_bc_z[0] = jax.lax.index_in_dim(s_ij_back[0].data, 0, axis=axis, keepdims=True) + \
                    jax.lax.index_in_dim(
                        s_ij_back[1].data, 0, axis=axis, keepdims=True)

    bc_value_sum = [du_bc_x, du_bc_y,
                    du_bc_z] if grid.ndim == 3 else [du_bc_x, du_bc_y]

    new_bc = ConstantBoundaryConditions(
        tuple(bc_type_sum), tuple(bc_value_sum))
    return new_bc


def get_vis_bc_value_from_velocity(
        grid,
        v: GridVariableVector,
        vis: grids.GridArrayTensor):
    """
    directly from the calculation
    """
    bc = grids.consistent_boundary_conditions(*v)
    bc_type_sum = []
    bc_value_sum = []
    for i in range(grid.ndim):
        bc_type = []
        bc_value = []

        if bc.types[i][0] != 'periodic':
            if i == 0:
                value10 = 1 * \
                    jax.lax.index_in_dim(vis.data, 0, axis=0, keepdims=True)
                value20 = 1 * \
                    jax.lax.index_in_dim(vis.data, -1, axis=0, keepdims=True)
                bc_value.append((value10,  value20))
            elif i == 1:
                value11 = 0 * \
                    jax.lax.index_in_dim(vis.data, 0, axis=1, keepdims=True)
                value21 = 0 * \
                    jax.lax.index_in_dim(vis.data, -1, axis=1, keepdims=True)
                bc_value.append((value11,  value21))
            else:
                value12 = 1 * \
                    jax.lax.index_in_dim(vis.data, 0, axis=2, keepdims=True)
                value22 = 1 * \
                    jax.lax.index_in_dim(vis.data, -1, axis=2, keepdims=True)
                bc_value.append((value12, value22))
            bc_type.append(bc.types[i])
            # bc_value.append((lambda x,  t: value1, lambda x, t: value2))
        else:
            bc_type.append(bc.types[i])
            bc_value.append(bc._values[i])

        bc_type_sum.append(*bc_type)
        bc_value_sum.append(*bc_value)

    return ConstantBoundaryConditions(tuple(bc_type_sum), tuple(bc_value_sum))
