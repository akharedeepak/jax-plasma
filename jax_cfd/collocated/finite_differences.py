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

"""Functions for approximating derivatives.

Finite difference methods operate on GridVariable and return GridArray.
Evaluating finite differences requires boundary conditions, which are defined
for a GridVariable. But the operation of taking a derivative makes the boundary
condition undefined, which is why the return type is GridArray.

For example, if the variable c has the boundary condition c_b = 0, and we take
the derivate dc/dx, then it is unclear what the boundary condition on dc/dx
should be. So programmatically, after taking finite differences and doing
operations, the user has to explicitly assign boundary conditions to the result.

Example:
  c = GridVariable(c_array, c_boundary_condition)
  dcdx = finite_differences.forward_difference(c)  # returns GridArray
  c_new = c + dt * (-velocity * dcdx)  # operations on GridArrays
  c_new = GridVariable(c_new, c_boundary_condition)  # assocaite BCs
"""

import typing
from typing import Optional, Sequence, Tuple
# CHECK: this import for collocation
# from Jax_FSI.collocation import grids
# from Jax_FSI.collocation import interpolation
from jax_cfd.base import grids
from jax_cfd.base import interpolation
import numpy as np
import jax.numpy as jnp
import jax

GridArray = grids.GridArray
GridVariable = grids.GridVariable
GridArrayTensor = grids.GridArrayTensor


def stencil_sum(*arrays: GridArray) -> GridArray:
    """Sum arrays across a stencil, with an averaged offset."""
    # pylint: disable=line-too-long
    offset = grids.averaged_offset(*arrays)
    # pytype appears to have a bad type signature for sum():
    # Built-in function sum was called with the wrong arguments [wrong-arg-types]
    #          Expected: (iterable: Iterable[complex])
    #   Actually passed: (iterable: Generator[Union[jax.interpreters.xla.DeviceArray, numpy.ndarray], Any, None])
    result = sum(array.data for array in arrays)  # type: ignore
    grid = grids.consistent_grid(*arrays)
    # name = grids.consistent_name(*arrays)
    # if hasattr(*arrays, 'name'):
    return grids.GridArray(result, offset, grid)


# incompatible with typing.overload
# pylint: disable=pointless-statement
# pylint: disable=function-redefined
# pylint: disable=unused-argument


def expand_step(axis, grid, roll=None, move=False):
    """expand the 1D step to aimed shape,
      if it's not 1D, directly get the new step"""
    if len(grid.node[axis].shape) != 1:
        return jnp.diff(grid.node[axis], axis=0)

    else:
        if roll is not None:
            if roll < 0:  # pad low boundary value
                pad_width = (1, 0)
                trim = -1
            else:
                pad_width = (0, 1)
                trim = 1
            step = jnp.pad(grid.step[axis], pad_width, mode='edge')
            step_left = step/2
            step_right = jnp.roll(step_left, roll)
            step = step_left+step_right

            step = step[:trim] if trim < 0 else step[trim:]

        else:
            step = grid.step[axis]
            if move is True:
                step = jnp.append(step[1:], step[-1])

        if axis == 0:
            if grid.ndim == 2:
                axes = [1]
            elif grid.ndim == 3:
                axes = [1, 2]

        if axis == 1:
            if grid.ndim == 2:
                axes = [0]
            elif grid.ndim == 3:
                axes = [0, 2]

        if axis == 2:
            axes = [0, 1]

        length = tuple(grid.shape[a] for a in axes)
        step = jnp.expand_dims(step, axis=axes)
        for length_i, axes_i in zip(length, axes):
            step = jnp.repeat(step, length_i, axis=axes_i)
        return step


@typing.overload
def central_difference(u: GridVariable, axis: int) -> GridArray:
    ...


@typing.overload
def central_difference(
        u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
    ...


def central_difference(u, axis=None):
    """Approximates grads with central differences.
    We do interpolate at first and then do the central difference.
    """
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(central_difference(u, a) for a in axis)
    # right side
    new_offset_right = tuple(o + .5 if i == axis else o
                        for i, o in enumerate(u.offset))
    # left side
    new_offset_left = tuple(o - .5 if i == axis else o
                        for i, o in enumerate(u.offset))

    u_right = interpolation.linear(u, new_offset_right).array
    u_left = interpolation.linear(u, new_offset_left).array
    if u.grid.stretch is None:
        new_step = u.grid.step[axis]
    else:
        new_step = expand_step(axis, u.grid, None)
    return stencil_sum(u_right, -u_left) / new_step

    # if axis is None:
    #     axis = range(u.grid.ndim)
    # if not isinstance(axis, int):
    #     return tuple(central_difference(u, a, t) for a in axis)
    # if u.grid.stretch is None:
    #     diff = stencil_sum(u.shift(+1, axis, t), -u.shift(-1, axis, t))
    #     return diff / (2 * u.grid.step[axis])



@typing.overload
def backward_difference(u: GridVariable, axis: int) -> GridArray:
    ...


@typing.overload
def backward_difference(
        u: GridVariable, axis: Optional[Tuple[int, ...]]) -> Tuple[GridArray, ...]:
    ...


def backward_difference(u, axis=None):
    """Approximates grads with finite differences in the backward direction."""
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(backward_difference(u, a) for a in axis)
    diff = stencil_sum(u.array, -u.shift(-1, axis))
    if u.grid.stretch is None:
        return diff / u.grid.step[axis]
    else:
        # make the shape of "diff" and "step" consistence
        if np.isclose(u.offset[axis], 0.5):
            roll = -1
        elif np.isclose(u.offset[axis], 1.5):
            roll = 1
        elif np.isclose(u.offset[axis], 1):
            roll = None
        else:
            raise ValueError(
                f'expected offset values in {{0.5, 1, 1.5}}, got {u.offset[axis]}')
        new_step = expand_step(axis, u.grid, roll)
        return diff * (1 / new_step)


@typing.overload
def forward_difference(u: GridVariable, axis: int) -> GridArray:
    ...


@typing.overload
def forward_difference(
    u: GridVariable,
        axis: Optional[Tuple[int, ...]] = ...) -> Tuple[GridArray, ...]:
    ...


def forward_difference(u, axis=None):
    """Approximates grads with finite differences in the forward direction."""
    if axis is None:
        axis = range(u.grid.ndim)
    if not isinstance(axis, int):
        return tuple(forward_difference(u, a) for a in axis)
    diff = stencil_sum(u.shift(+1, axis), -u.array)

    if u.grid.stretch is None:
        return diff / u.grid.step[axis]
    else:
        # make the shape of "diff" and "step" consistence
        if np.isclose(u.offset[axis], 0.5):
            roll = 1
            move = False
        elif np.isclose(u.offset[axis], 0):
            roll = None
            move = False
        elif np.isclose(u.offset[axis], 1):
            roll = None
            move = True
        else:  # usually do not need this one
            raise ValueError(
                f'expected offset values in {{0, 0.5, 1}}, got {u.offset[axis]}')

        new_step = expand_step(axis, u.grid, roll, move)
        return diff * (1 / new_step)


def laplacian(u: GridVariable) -> GridArray:
    """Approximates the Laplacian of `u`."""
    if u.grid.stretch is None:
        scales = np.square(1 / np.array(u.grid.step, dtype=u.dtype))
        result = -2 * u.array * np.sum(scales)
        for axis in range(u.grid.ndim):
            result += stencil_sum(u.shift(-1, axis),
                                  u.shift(+1, axis)) * scales[axis]
        return result
    else:
        gradient_total = []
        for axis in range(u.grid.ndim):
            gradient_face_left = backward_difference(u, axis)
            gradient_face_right = forward_difference(u, axis)
            if np.isclose(u.offset[axis] % 1, 0):
                roll = 1
            else:
                roll = None
            new_setp = expand_step(axis, u.grid, roll)
            gradient_final = stencil_sum(
                gradient_face_right, -gradient_face_left)*(1/new_setp)
            gradient_total.append(gradient_final)
        return sum(gradient_total)


def divergence(v: Sequence[GridVariable], sum_all=True) -> GridArray:
    """Approximates the divergence of `v` using backward differences."""
    grid = grids.consistent_grid(*v)
    if len(v) != grid.ndim:
        raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                         f'Expected length {grid.ndim}; got {len(v)}.')
    differences = [backward_difference(u, axis) for axis, u in enumerate(v)]

    if sum_all == True:
        return sum(differences)
    else:
        return differences


def forward_individual_divergence(v: Sequence[GridVariable], sum_all=True) -> GridArray:
    """Approximates the divergence of `v` using backward differences."""
    grid = grids.consistent_grid(*v)
    if len(v) != grid.ndim:
        raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                         f'Expected length {grid.ndim}; got {len(v)}.')
    differences = [forward_difference(u, axis) for axis, u in enumerate(v)]

    return differences


def centered_divergence(v: Sequence[GridVariable]) -> GridArray:
    """Approximates the divergence of `v` using centered differences."""
    grid = grids.consistent_grid(*v)
    if len(v) != grid.ndim:
        raise ValueError('The length of `v` must be equal to `grid.ndim`.'
                         f'Expected length {grid.ndim}; got {len(v)}.')
    differences = [central_difference(u, axis) for axis, u in enumerate(v)]
    return sum(differences)


@typing.overload
def gradient_tensor(v: GridVariable) -> GridArrayTensor:
    ...


@typing.overload
def gradient_tensor(v: Sequence[GridVariable]) -> GridArrayTensor:
    ...


def gradient_tensor(v):
    """Approximates the cell-centered gradient of `v`."""
    if not isinstance(v, GridVariable):
        return GridArrayTensor(np.stack([gradient_tensor(u) for u in v], axis=-1))
    grad = []
    for axis in range(v.grid.ndim):
        offset = v.offset[axis]
        if offset == 0:
            derivative = forward_difference(v, axis)
        elif offset == 1:
            derivative = backward_difference(v, axis)
        elif offset == 0.5:
            v_centered = interpolation.linear(v, v.grid.cell_center)
            derivative = central_difference(v_centered, axis)
        else:
            raise ValueError(
                f'expected offset values in {{0, 0.5, 1}}, got {offset}')
        grad.append(derivative)
    return GridArrayTensor(grad)


def curl_2d(v: Sequence[GridVariable]) -> GridArray:
    """Approximates the curl of `v` in 2D using forward differences."""
    v = v['velocity']
    if len(v) != 2:
        raise ValueError(f'Length of `v` is not 2: {len(v)}')
    grid = grids.consistent_grid(*v)
    if grid.ndim != 2:
        raise ValueError(f'Grid dimensionality is not 2: {grid.ndim}')
    return forward_difference(v[1], axis=0) - forward_difference(v[0], axis=1)


def curl_3d(
        v: Sequence[GridVariable]) -> Tuple[GridArray, GridArray, GridArray]:
    """Approximates the curl of `v` in 2D using forward differences."""
    if len(v) != 3:
        raise ValueError(f'Length of `v` is not 3: {len(v)}')
    grid = grids.consistent_grid(*v)
    if grid.ndim != 3:
        raise ValueError(f'Grid dimensionality is not 3: {grid.ndim}')
    curl_x = (forward_difference(v[2], axis=1) -
              forward_difference(v[1], axis=2))
    curl_y = (forward_difference(v[0], axis=2) -
              forward_difference(v[2], axis=0))
    curl_z = (forward_difference(v[1], axis=0) -
              forward_difference(v[0], axis=1))
    return (curl_x, curl_y, curl_z)


def seperate_laplacian(u: GridVariable, axis) -> GridArray:
    """Approximates the Laplacian of `u`."""
    if u.grid.stretch is None:
        step = u.grid.step[axis]
    else:
        if np.isclose(u.offset[axis] % 1, 0):
            roll = 1
        else:
            roll = None
        step = expand_step(axis, u.grid, roll)

    gradient_face_left = backward_difference(u, axis)
    gradient_face_right = forward_difference(u, axis)
    gradient_final = stencil_sum(
        gradient_face_right, -gradient_face_left)*(1/step)
    return gradient_final


def central_laplacian(u: GridVariable) -> GridArray:
    """Approximates the Laplacian of `u`."""
    gradient_final = []
    for axis in range(u.grid.ndim):
        if u.grid.stretch is None:
            step = u.grid.step[axis]
        else:
            if np.isclose(u.offset[axis] % 1, 0):
                roll = 1
            else:
                roll = None
            step = expand_step(axis, u.grid, roll)

        gradient_face = central_difference(u, axis)
        gradient_face_variable = grids.GridVariable(gradient_face, bc=u.bc)
        gradient_final.append(central_difference(
            gradient_face_variable, axis))
    return sum(gradient_final)
