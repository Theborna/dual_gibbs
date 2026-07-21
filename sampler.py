

import numba
import numpy as np

# ==========================================
# 1. GENERAL GRAPH SAMPLING KERNELS
# ==========================================
@numba.jit(nopython=True, fastmath=True, cache=False)
def primal_sampling_kernel_general(x, neighbors, neighbor_weights, inv_diag, std_dev, n_vars, 
                                   n_samples, thinning, indices, mean_only=False):
    """
    General Gibbs Sampler for primal formulation.
    Conditional mean: mu_i = -Q_ii^{-1} * sum_j Q_ij * x_j
    
    Args:
        neighbors: List of arrays, neighbors[i] = array of neighbor indices
        neighbor_weights: List of arrays, neighbor_weights[i][j] = Q[i, neighbors[i][j]]
        inv_diag: 1.0 / Q_ii for each node
        std_dev: 1.0 / sqrt(Q_ii)
    """
    samples = np.zeros((n_samples, n_vars))
    idx_ptr = 0
    buffer_len = len(indices)

    for s_idx in range(n_samples):
        samples[s_idx, :] = x.copy()
        for _ in range(thinning):
            gaussians = np.random.standard_normal(n_vars)
            for i in range(n_vars):
                u = indices[idx_ptr]
                idx_ptr += 1
                if idx_ptr >= buffer_len: idx_ptr = 0
                
                sum_weighted = 0.0
                deg_u = len(neighbors[u])
                for j in range(deg_u):
                    v = neighbors[u][j]
                    sum_weighted += neighbor_weights[u][j] * x[v]
                
                # Conditional mean: -Q_ii^{-1} * sum_j Q_ij * x_j
                mean_val = -inv_diag[u] * sum_weighted
                x[u] = mean_val + std_dev[u] * gaussians[i] if not mean_only else mean_val
        
    return samples


@numba.jit(nopython=True, fastmath=True, cache=False)
def dual_sampling_kernel_general(x_edges, S_nodes, u_list, v_list, edge_indices,
                                 s_squared_nodes, inv_diag, std_dev,
                                 n_samples, thinning, num_edges, mean_only=False):
    """
    O(1) Dual Sampler for GENERAL graphs using implicit factorization.
    
    Works for heterogeneous graphs: Q_D = D_sigma + B^T D_s B
    
    Key insight: sum_{f != e} Q_ef * x_f = s_u^2 * (S_u - x_e) - s_v^2 * (S_v + x_e)
    where S_u = sum of signed edge variables incident to node u
    
    Args:
        S_nodes: Node sums, S_u = sum_{f incident to u} sgn(u,f) * x_f
        s_squared_nodes: Array of s_i^2 for each node i
        inv_diag: 1.0 / Q_ee for each edge
        std_dev: 1.0 / sqrt(Q_ee)
    """
    samples = np.zeros((n_samples, num_edges))
    idx_ptr = 0
    buffer_len = len(edge_indices)
    
    for s_idx in range(n_samples):
        samples[s_idx, :] = x_edges.copy()
        for _ in range(thinning):
            gaussians = np.random.standard_normal(num_edges)
            for i in range(num_edges):
                e = edge_indices[idx_ptr]
                idx_ptr += 1
                if idx_ptr >= buffer_len: idx_ptr = 0
                
                u = u_list[e]
                v = v_list[e]
                
                old_val = x_edges[e]
                # Remove edge e from node sums
                S_u_res = S_nodes[u] - old_val
                S_v_res = S_nodes[v] + old_val
                
                # General implicit formula:
                # sum_{f != e} Q_ef * x_f = s_u^2 * S_u_res - s_v^2 * S_v_res
                interaction = s_squared_nodes[u] * S_u_res - s_squared_nodes[v] * S_v_res
                
                # Conditional mean: -Q_ee^{-1} * interaction
                mean_val = -inv_diag[e] * interaction
                
                # Sample new value
                new_val = mean_val + std_dev[e] * gaussians[i] if not mean_only else mean_val
                
                x_edges[e] = new_val
                # Update node sums with new value
                S_nodes[u] = S_u_res + new_val
                S_nodes[v] = S_v_res - new_val

    return samples