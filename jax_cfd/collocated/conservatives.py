from typing import Callable, Sequence, TypeVar

import jax.numpy as jnp

import jax_cfd.base.grids as grids


from jax_cfd.collocated.boundaries import BCType, ConstantBoundaryConditions

def _sum_GVs(s:Sequence[grids.GridVariable]) -> grids.GridVariable:
    '''
    s = [s1, s2, ...]: scalar GridVariables
    return sum(s) GridVariable
    '''
    assert len(s) >= 2 
    sum_s_array = sum(si.array for si in s)
    s_bc_types = []
    s_bc_values = []
    for axis in range(s[0].grid.ndim):
        s_bc_types_ax = []
        s_bc_values_ax = []
        
        for lr, _shift, _trim in zip(range(2), [-1, 1], [0, -1]):  # lower and upper boundary.
            if all(si[axis][lr] == BCType.NEUMANN for si in s):
                s_bc_types_ax.append(BCType.NEUMANN)
                # d(s1+s2)/dx = d(s1)/dx + d(s2)/dx
                data_s = sum(si.bc._values[axis][lr] for si in s)
            else:
                s_bc_types_ax.append(BCType.DIRICHLET)
                if (si[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC) for si in s):
                    data_s = sum(jnp.take(si.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis) for si in s)
                else:
                    raise NotImplementedError
            s_bc_values_ax.append(data_s)

        s_bc_types.append(tuple(s_bc_types_ax))
        s_bc_values.append(tuple(s_bc_values_ax))
    
    s_bc = ConstantBoundaryConditions(s_bc_types, s_bc_values)
    return grids.GridVariable(sum_s_array, s_bc)


def _s1_times_s2(s1:grids.GridVariable, s2:grids.GridVariable) -> grids.GridVariable:
    '''
    s1, s2: scalar GridVariable
    return s1.s2 GridVariable
    '''
    rhouu_array = s1.array * s2.array
    s1s2_bc_types = []
    s1s2_bc_values = []
    for axis in range(s1.grid.ndim):
        s1s2_bc_types_ax = []
        s1s2_bc_values_ax = []

        for lr, _shift, _trim in zip(range(2), [-1, 1], [0, -1]):  # lower and upper boundary.
            if (s1.bc.types[axis][lr] == BCType.NEUMANN and
                s2.bc.types[axis][lr] == BCType.NEUMANN and
                s1.bc._values[axis][lr] == 0 and
                s2.bc._values[axis][lr] == 0):
                s1s2_bc_types_ax.append(BCType.NEUMANN)
                data_s1 = s1.bc._values[axis][lr]
                data_s2 = s2.bc._values[axis][lr]
            else:
                s1s2_bc_types_ax.append(BCType.DIRICHLET)
                if (s1.bc.types[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC)):
                    data_s1 = jnp.take(s1.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
                else:
                    raise NotImplementedError
                
                if (s2.bc.types[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC)):
                    data_s2 = jnp.take(s2.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
                else:
                    raise NotImplementedError

            s1s2_bc_values_ax.append(data_s1 * data_s2)

        s1s2_bc_types.append(tuple(s1s2_bc_types_ax))
        s1s2_bc_values.append(tuple(s1s2_bc_values_ax))
    
    s1s2_bc = ConstantBoundaryConditions(s1s2_bc_types, s1s2_bc_values)
    s1s2 = grids.GridVariable(rhouu_array, s1s2_bc)

    return s1s2

def _s_times_v(s:grids.GridVariable, v:Sequence[grids.GridVariable]) -> Sequence[grids.GridVariable]:
    '''
    s: scalar
    v: vector
    return (s.v1, s.v2, ...)
    '''
    return tuple(_s1_times_s2(s, u) for u in v)

def _rhov(rho, v):
    return _s_times_v(rho, v)

def _rhouv(rhou, v):
    return _s_times_v(rhou, v)

def _rhovv(rhov, v):
    return tuple(_rhouv(rhou, v) for rhou in rhov)

def _rhoE(rho, E):
    return _s1_times_s2(rho, E)

def _rhoEv(rhoE, v):
    return _s_times_v(rhoE, v)

# def _KE(v):
#     v_sq = tuple(_s1_times_s2(u, u) for u in v)
#     KE = 0
#     for u_sq in v_sq:
#         KE += 


# def _s1_plus_s2(s1:grids.GridVariable, s2:grids.GridVariable) -> grids.GridVariable:
#     '''
#     s1, s2: scalar GridVariable
#     return s1 + s2 GridVariable
#     '''
#     s1s2_array = s1.array + s2.array
#     s1s2_bc_types = []
#     s1s2_bc_values = []
#     for axis in range(s1.grid.ndim):
#         s1s2_bc_types_ax = []
#         s1s2_bc_values_ax = []
        
#         for lr, _shift, _trim in zip(range(2), [-1, 1], [0, -1]):  # lower and upper boundary.
#             if (s1.bc.types[axis][lr] == BCType.NEUMANN and
#                 s2.bc.types[axis][lr] == BCType.NEUMANN):
#                 s1s2_bc_types_ax.append(BCType.NEUMANN)
#                 data_s1 = s1.bc._values[axis][lr]
#                 data_s2 = s2.bc._values[axis][lr]
#             else:
#                 s1s2_bc_types_ax.append(BCType.DIRICHLET)
#                 if (s1.bc.types[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC)):
#                     data_s1 = jnp.take(s1.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                 else:
#                     raise NotImplementedError
                
#                 if (s2.bc.types[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC)):
#                     data_s2 = jnp.take(s2.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                 else:
#                     raise NotImplementedError

#             s1s2_bc_values_ax.append(data_s1 + data_s2)

#         s1s2_bc_types.append(tuple(s1s2_bc_types_ax))
#         s1s2_bc_values.append(tuple(s1s2_bc_values_ax))
    
#     s1s2_bc = ConstantBoundaryConditions(s1s2_bc_types, s1s2_bc_values)
#     s1s2 = grids.GridVariable(s1s2_array, s1s2_bc)

#     return s1s2


# def _rhouu_plus_p(rhouu, p):

#     rhouu_array = rhouu.array + p.array
#     ruup_bc_types = []
#     ruup_bc_values = []
#     for axis in range(p.grid.ndim):
#         ruup_bc_types_ax = []
#         ruup_bc_values_ax = []
#         # for i in range(2):  # lower and upper boundary.
#         for i, _shift, _trim in zip(range(2), [-1, 1], [0, -1]):  # lower and upper boundary.
#             if (rhouu.bc.types[axis][i] == BCType.NEUMANN and
#                       p.bc.types[axis][i] == BCType.NEUMANN):
#                 ruup_bc_types_ax.append(BCType.NEUMANN)
#                 data_ruu = rhouu.bc._values[axis][i]
#                 data_p   = p.bc._values[axis][i]
#             else:
#                 ruup_bc_types_ax.append(BCType.DIRICHLET)
#                 if rhouu.bc.types[axis][i] == BCType.DIRICHLET or rhouu.bc.types[axis][i] == BCType.NEUMANN or rhouu.bc.types[axis][i] == BCType.PERIODIC:
#                     data_ruu = jnp.take(rhouu.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                 else:
#                     raise NotImplementedError
                
#                 if p.bc.types[axis][i] == BCType.DIRICHLET or p.bc.types[axis][i] == BCType.NEUMANN or p.bc.types[axis][i] == BCType.PERIODIC:
#                     data_p   = jnp.take(p.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                 else:
#                     raise NotImplementedError

#             ruup_bc_values_ax.append(data_ruu + data_p)

#         ruup_bc_types.append(tuple(ruup_bc_types_ax))
#         ruup_bc_values.append(tuple(ruup_bc_values_ax))
    
#     ruup_bc = ConstantBoundaryConditions(ruup_bc_types, ruup_bc_values)
#     rhouu_plus_p = grids.GridVariable(rhouu_array, ruup_bc)

#     return rhouu_plus_p



# def _rhov(rho, v):

#     rhov = []
#     for u in v:
#         rhou_array = rho.array * u.array
#         rhou_bc_types = []
#         rhou_bc_values = []
#         for axis in range(rho.grid.ndim):
#             rhou_bc_types_ax = []
#             rhou_bc_values_ax = []

#             # for i in range(2):  # lower and upper boundary.
#             for i, _shift, _trim in zip(range(2), [-1, 1], [0, -1]):  # lower and upper boundary.
#                 if (rho.bc.types[axis][i] == BCType.NEUMANN and
#                       u.bc.types[axis][i] == BCType.NEUMANN):
#                     rhou_bc_types_ax.append(BCType.NEUMANN)
#                     data_rho = rho.bc._values[axis][i]
#                     data_u   = u.bc._values[axis][i]
#                 else:
#                     rhou_bc_types_ax.append(BCType.DIRICHLET)
#                     if rho.bc.types[axis][i] == BCType.DIRICHLET or rho.bc.types[axis][i] == BCType.NEUMANN or rho.bc.types[axis][i] == BCType.PERIODIC:
#                         data_rho = jnp.take(rho.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                     else:
#                         raise NotImplementedError
                    
#                     if u.bc.types[axis][i] == BCType.DIRICHLET or u.bc.types[axis][i] == BCType.NEUMANN or u.bc.types[axis][i] == BCType.PERIODIC:
#                         data_u = jnp.take(u.shift(_shift, axis).data, indices=jnp.array([_trim]), axis=axis)
#                     else:
#                         raise NotImplementedError
                    
#                 rhou_bc_values_ax.append(data_rho * data_u)

#             rhou_bc_types.append( tuple(rhou_bc_types_ax))
#             rhou_bc_values.append(tuple(rhou_bc_values_ax))
        
#         rhou_bc = ConstantBoundaryConditions(rhou_bc_types, rhou_bc_values)
#         rhov.append(grids.GridVariable(rhou_array, rhou_bc))

#     return tuple(rhov)

# def _rhovv_plus_p(rhov, v, p):
#     """
#     for 3D:
#     (rho*u*u + p, rho*u*v    , rho*u*w    )
#     (rho*u*u    , rho*u*v + p, rho*u*w    )
#     (rho*u*u    , rho*u*v    , rho*u*w + p)
#     """
#     rhovv_plus_p = []
#     for eqi, rhou in enumerate(rhov):
#         rhouv = _rhouv(rhou, v)
        
#         # for 3D, eqi==0: (rhouu + p, rhouv, rhouw)
#         rhouv_plus_p = []
#         for axis, rhouu in enumerate(rhouv):
#             if eqi==axis:
#                 rhouv_plus_p.append(_rhouu_plus_p(rhouu, p))
#             else:
#                 rhouv_plus_p.append(rhouu)
#         rhovv_plus_p.append(tuple(rhouv_plus_p))
    
#     return tuple(rhovv_plus_p)

