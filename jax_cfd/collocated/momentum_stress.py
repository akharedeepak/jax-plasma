import jax.numpy as jnp

from jax_cfd.collocated import finite_differences as fd
from jax_cfd.collocated import boundaries
from jax_cfd.base import grids

Array = grids.Array
GridArray = grids.GridArray
GridArrayVector = grids.GridArrayVector
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector


# def calculate_strain_tensor(grid, v, t):
# # calculate forward_sij
#     s_dev_central = []
#     for i in range(grid.ndim):
#         s_dev_sub = []
#         for j in range(grid.ndim):
#             s_dev_sub.append([0.5*fd.central_difference(v[i], j, t),
#                               0.5*fd.central_difference(v[j], i, t)])
#         s_dev_central.append(s_dev_sub)

#     s_ij_center = grids.GridArrayTensor(
#         [[utility.sum_fields(*s) for s in s_dev_central[i]] for i in range(grid.ndim)])
    
#     tau = jax.tree_util.tree_map(
#             lambda x, y: 2. * x * y, viscosity_sgs_tensor, s_ij_with_bc_tensor)

# TODO CHECK if this is correct
def stress(v: GridVariable) -> GridArray:
    '''
    Calculate the stress tensor from the velocity field.
    The stress tensor is calculated using the formula:  [\nabla v + \nabla v^T] + \frac{2}{3}(\nabla \cdot v)I
    '''
    
    # dudx_ldb = fd.backward_difference(v[0], 0, t).data[:1]
    # dvdx_ldb = fd.backward_difference(v[1], 0, t).data[:1]
    # dudx_rdb = fd.forward_difference(v[0], 0, t).data[-1:]
    # dvdx_rdb = fd.forward_difference(v[1], 0, t).data[-1:]
    # dudy_bdb = fd.backward_difference(v[0], 1, t).data[:,:1]
    # dvdy_bdb = fd.backward_difference(v[1], 1, t).data[:,:1]
    # dudy_tdb = fd.forward_difference(v[0], 1, t).data[:,-1:]
    # dvdy_tdb = fd.forward_difference(v[1], 1, t).data[:,-1:]

    ndim = len(v)
    ## ((du/dx, du/dy), 
    ##  (dv/dx, dv/dy))
    gradV  = tuple(
            tuple(
                fd.central_difference(u, a) 
            for a in range(ndim))
            for u in v)
    
    ##  (([(dudx_l, dudx_r), (dudx_t, dudx_b)], [(dudy_l, dudy_r), (dudy_t, dudy_b)]),
    ##   ([(dvdx_l, dvdx_r), (dvdx_t, dvdx_b)], [(dvdy_l, dvdy_r), (dvdy_t, dvdy_b)]))
    gradV_bd= tuple(
            tuple(
                [(jnp.take(fd.backward_difference(u, a).data, indices=jnp.array([ 0]), axis=_a),
                jnp.take(fd.forward_difference( u, a).data, indices=jnp.array([-1]), axis=_a)) for _a in range(ndim)]
            for a in range(ndim))
            for u in v)
    
    ## ((divv, 0), 
    ##  (0, divv))
    divVi    = - 2/3 * sum(gradV[i][i] for i in range(ndim))
    divVi_bd  = [tuple(- 2/3 * sum(gradV_bd[i][i][_ax][_alr] for i in range(ndim))  for _alr in range(ndim)) for _ax in range(ndim)]
    divV  = tuple(
            tuple( 
                divVi if i == j else 0
            for j in range(ndim))
            for i in range(ndim))
    divV_bd  = tuple(
            tuple( 
                divVi_bd if i == j else [(0, 0), (0, 0)]
            for j in range(ndim))
            for i in range(ndim))
    
    tau   = tuple(
            tuple( 
                GridVariable(gradV[i][j] + gradV[j][i] + divV[i][j] 
                , bc=boundaries.ConstantBoundaryConditions((('dirichlet', 'dirichlet'),('dirichlet','dirichlet')),
                                                           ((gradV_bd[i][j][0][0]+gradV_bd[j][i][0][0]+divV_bd[i][j][0][0], gradV_bd[i][j][0][1]+gradV_bd[j][i][0][1]+divV_bd[i][j][0][1]),
                                                            (gradV_bd[i][j][1][0]+gradV_bd[j][i][1][0]+divV_bd[i][j][1][0], gradV_bd[i][j][1][1]+gradV_bd[j][i][1][1]+divV_bd[i][j][1][1]))))
            for j in range(ndim))
            for i in range(ndim))

    # div_tau =   tuple(
    #             nu*sum( 
    #                 fd.central_difference(tau[i][j], j) 
    #             for j in range(ndim))
    #             for i in range(ndim))
    return tau
