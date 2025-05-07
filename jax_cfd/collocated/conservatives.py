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
            if all(si.bc.types[axis][lr] == BCType.NEUMANN for si in s):
                s_bc_types_ax.append(BCType.NEUMANN)
                # d(s1+s2)/dx = d(s1)/dx + d(s2)/dx
                data_s = sum(si.bc._values[axis][lr] for si in s)
            else:
                s_bc_types_ax.append(BCType.DIRICHLET)
                if (si.bc.types[axis][lr] in (BCType.DIRICHLET, BCType.NEUMANN, BCType.PERIODIC) for si in s):
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

def _pv(p, v):
    return _s_times_v(p, v)

def _tauv(tau, v):
    return tuple(
            _sum_GVs([_s1_times_s2(tauij, u) for tauij, u in zip(taui, v)]) 
            for taui in tau)

def _KE(v):
    v_sq = tuple(_s1_times_s2(u, u) for u in v)
    KE = _sum_GVs(v_sq)
    KE.array = KE.array / 2
    return KE

def _rhouu_plus_p(rhouu, p):
    return _sum_GVs([rhouu, p])

def _rhovv_plus_p(rhov, v, p):
    """
    for 3D:
    (rho*u*u + p, rho*u*v    , rho*u*w    )
    (rho*u*u    , rho*u*v + p, rho*u*w    )
    (rho*u*u    , rho*u*v    , rho*u*w + p)
    """
    rhovv_plus_p = []
    for eqi, rhou in enumerate(rhov):
        rhouv = _rhouv(rhou, v)
        
        # for 3D, eqi==0: (rhouu + p, rhouv, rhouw)
        rhouv_plus_p = []
        for axis, rhouu in enumerate(rhouv):
            if eqi==axis:
                rhouv_plus_p.append(_rhouu_plus_p(rhouu, p))
            else:
                rhouv_plus_p.append(rhouu)
        rhovv_plus_p.append(tuple(rhouv_plus_p))
    
    return tuple(rhovv_plus_p)
