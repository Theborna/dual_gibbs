from sampler import (
    primal_sampling_kernel_general, dual_sampling_kernel_general
)
import numpy as np
import scipy.sparse as sp
import networkx as nx


def _build_scan_indices(n, buffer_size, scan="random"):
    """
    Builds the flat coordinate-index buffer consumed by the numba sampling
    kernels (sampler.py). The kernels are agnostic to how this buffer was
    built; they just cycle through it, so adding a scan order only requires
    changing how this array is constructed.

    scan:
      'random'       i.i.d. uniform coordinate choice each substep. This is
                      the random-sweep Gibbs sampler analyzed in the paper
                      (Propositions 1-5 are proved specifically for this
                      scan; see Roberts & Sahu 1997, Theorem 2, and Amit 1996).
      'permutation'   a fresh uniform random permutation of {0,...,n-1} every
                      n substeps, so each coordinate is updated exactly once
                      per sweep, in random order ("random-scan without
                      replacement" / shuffled sweep).
      'fixed'         the deterministic cycle 0,1,...,n-1,0,1,...,n-1,...
                      (systematic / fixed-order sweep).

    NOTE: 'permutation' and 'fixed' are NOT covered by the paper's theory
    (Section IX, Future Work); they are provided for empirical comparison
    only. For these two, buffer_size is rounded down to the nearest multiple
    of n so that sweep boundaries stay aligned when the buffer wraps.
    """
    if scan == "random":
        return np.random.randint(0, n, buffer_size)
    elif scan == "permutation":
        buffer_size = max(n, (buffer_size // n) * n)
        n_sweeps = buffer_size // n
        idx = np.empty(buffer_size, dtype=np.int64)
        for k in range(n_sweeps):
            idx[k * n:(k + 1) * n] = np.random.permutation(n)
        return idx
    elif scan == "fixed":
        buffer_size = max(n, (buffer_size // n) * n)
        n_sweeps = buffer_size // n
        return np.tile(np.arange(n), n_sweeps)
    else:
        raise ValueError(f"unknown scan '{scan}'; expected 'random', 'permutation', or 'fixed'")


# ==========================================
# BASE CLASS FOR GENERAL GRAPHS
# ==========================================
class GMRF_General:
    """
    General GMRF with arbitrary node variances s_i^2 and edge variances sigma_e^2.
    Works for any graph structure.
    """
    def __init__(self, G, s_nodes=None, sigma_edges=None):
        """
        Args:
            G: NetworkX graph
            s_nodes: array of length |V| or scalar (node variances)
            sigma_edges: array of length |E| or scalar (edge variances)
        """
        self.G = G
        self.nodes = list(self.G.nodes())
        self.edges = list(self.G.edges())
        self.num_nodes = len(self.nodes)
        self.num_edges = len(self.edges)
        
        # Handle scalar or array inputs
        if s_nodes is None or np.isscalar(s_nodes):
            self.s_nodes = np.full(self.num_nodes, s_nodes if s_nodes else 1.0)
        else:
            self.s_nodes = np.array(s_nodes)
            
        if sigma_edges is None or np.isscalar(sigma_edges):
            self.sigma_edges = np.full(self.num_edges, sigma_edges if sigma_edges else 1.0)
        else:
            self.sigma_edges = np.array(sigma_edges)
        
        print(f"Graph: {G}")
        print(f"Nodes: {self.num_nodes}, Edges: {self.num_edges}")
        
        self.B = self._build_incidence_matrix()
        self.Q_primal = self._build_primal_Q()
        self.Q_dual = self._build_dual_Q()

        self._prep_primal_general()
        self._prep_dual_general()

    def _build_incidence_matrix(self):
        """Build signed incidence matrix B where B[v,e] = +1/-1/0"""
        rows, cols, data = [], [], []
        for e_idx, (u, v) in enumerate(self.edges):
            rows.append(u); cols.append(e_idx); data.append(1.0)
            rows.append(v); cols.append(e_idx); data.append(-1.0)
        return sp.csc_matrix((data, (rows, cols)), shape=(self.num_nodes, self.num_edges))

    def _build_primal_Q(self):
        """Q_primal = D_s^{-1} + B D_sigma^{-1} B^T"""
        D_s_inv = sp.diags(1.0 / (self.s_nodes**2))
        D_sigma_inv = sp.diags(1.0 / (self.sigma_edges**2))
        return D_s_inv + self.B @ D_sigma_inv @ self.B.T

    def _build_dual_Q(self):
        """Q_dual = D_sigma + B^T D_s B"""
        D_sigma = sp.diags(self.sigma_edges**2)
        D_s = sp.diags(self.s_nodes**2)
        return D_sigma + self.B.T @ D_s @ self.B

    def _prep_primal_general(self):
        """Prepare data structures for general primal Gibbs sampling"""
        Q = self.Q_primal.tocsr()
        
        # Extract neighbors and weights for each node
        self.primal_neighbors = []
        self.primal_neighbor_weights = []
        
        for i in range(self.num_nodes):
            row_start = Q.indptr[i]
            row_end = Q.indptr[i + 1]
            row_indices = Q.indices[row_start:row_end]
            row_data = Q.data[row_start:row_end]
            
            # Extract off-diagonal neighbors
            mask = row_indices != i
            neighbors = row_indices[mask]
            weights = row_data[mask]
            
            self.primal_neighbors.append(neighbors)
            self.primal_neighbor_weights.append(weights)
        
        # Diagonal and standard deviations
        diag = Q.diagonal()
        self.primal_inv_diag = 1.0 / diag
        self.primal_std = np.sqrt(self.primal_inv_diag)

    def _prep_dual_general(self):
        """Prepare data structures for O(1) implicit dual Gibbs sampling"""
        Q = self.Q_dual.tocsr()
        
        # Diagonal and standard deviations
        diag = Q.diagonal()
        self.dual_inv_diag = 1.0 / diag
        self.dual_std = np.sqrt(self.dual_inv_diag)
        
        # Edge endpoint arrays
        edges_arr = np.array(self.edges, dtype=np.int32)
        self.u_list = edges_arr[:, 0]
        self.v_list = edges_arr[:, 1]
        
        # Per-node s^2 values for implicit formula
        self.s_squared_nodes = self.s_nodes**2
    
    # --- GENERAL SAMPLERS ---
    def sample_primal(self, n_samples=1000, thinning=10, mean_only=False, random_init=False, scan="random"):
        """
        Sample from primal distribution (general implementation).
        
        CORRECTED: random_init now properly initializes from prior-scaled noise.

        scan: 'random' (default, matches the paper's theory), 'permutation',
              or 'fixed'. See _build_scan_indices for details.
        """
        # Default initialization: constant vector
        x = 10 * np.ones(self.num_nodes) / np.sqrt(self.num_nodes)
        
        if random_init:
            # Initialize from scaled random noise
            # Scale by marginal std dev (approx sqrt(diag(Q^{-1})))
            x = 2 * np.random.randn(self.num_nodes) * self.primal_std
            # Init at D_s
            # x = np.random.randn(self.num_nodes) * self.s_nodes
        
        # Pre-generate scan indices with larger buffer for long runs
        total_steps = n_samples * thinning * self.num_nodes
        # Increase buffer size to reduce recycling
        buffer_size = min(total_steps, 2000000)
        indices = _build_scan_indices(self.num_nodes, buffer_size, scan)
        
        return primal_sampling_kernel_general(
            x, self.primal_neighbors, self.primal_neighbor_weights,
            self.primal_inv_diag, self.primal_std,
            self.num_nodes, n_samples, thinning, indices, mean_only
        )

    def sample_dual(self, n_samples=1000, thinning=10, mean_only=False, random_init=False, scan="random"):
        """
        Sample from dual distribution using O(1) implicit algorithm.
        
        CORRECTED: random_init now properly initializes from prior-scaled noise.

        scan: 'random' (default, matches the paper's theory), 'permutation',
              or 'fixed'. See _build_scan_indices for details.
        """
        # Default initialization: constant vector
        x = 10 * np.ones(self.num_edges) / np.sqrt(self.num_edges)
        
        if random_init:
            # Initialize from scaled random noise
            # Scale by marginal std dev (approx sqrt(diag(R^{-1})))
            x = 2 * np.random.randn(self.num_edges) * self.dual_std
            # Init at D_s
            # x = np.zeros_like(x)

        # Initialize node sums
        S_nodes = self.B @ x
        
        # Pre-generate scan indices with larger buffer
        total_steps = n_samples * thinning * self.num_edges
        buffer_size = min(total_steps, 2000000)
        indices = _build_scan_indices(self.num_edges, buffer_size, scan)
        
        # Use O(1) implicit sampler for general graphs!
        return dual_sampling_kernel_general(
            x, S_nodes, self.u_list, self.v_list, indices,
            self.s_squared_nodes, self.dual_inv_diag, self.dual_std,
            n_samples, thinning, self.num_edges, mean_only
        )

# ==========================================
# CONCRETE GRAPH TYPES
# ==========================================
class GMRF_K_Regular(GMRF_General):
    """Homogeneous k-regular graph."""
    def __init__(self, n, k, s, sigma):
        self.n = n
        self.k = k
        if (n * k) % 2 != 0:
            raise ValueError("n * k must be even for k-regular graphs.")
        self.G = nx.random_regular_graph(k, n)
        super().__init__(self.G, s, sigma)


class GMRF_Heterogeneous_K_Regular(GMRF_General):
    """Heterogeneous k-regular graph with random variances."""
    def __init__(self, n, k, s_alpha=2, s_beta=5, sigma_alpha=2, sigma_beta=5, 
                 min_s=0.8, max_s=1.2, min_sigma=0.8, max_sigma=1.2):
        self.n = n
        self.k = k
        if (n * k) % 2 != 0:
            raise ValueError("n * k must be even for k-regular graphs.")
        self.G = nx.random_regular_graph(k, n)
        
        # Sample variances from Beta distributions
        s_nodes = min_s + (max_s - min_s) * np.random.beta(s_alpha, s_beta, n)
        sigma_edges = min_sigma + (max_sigma - min_sigma) * np.random.beta(
            sigma_alpha, sigma_beta, len(self.G.edges())
        )
        super().__init__(self.G, s_nodes, sigma_edges)


class GMRF_CompleteBipartite(GMRF_General):
    """Complete bipartite graph K_{n1,n2}."""
    def __init__(self, n1, n2, s, sigma):
        self.G = nx.complete_bipartite_graph(n1=n1, n2=n2)
        super().__init__(self.G, s, sigma)


class GMRF_Hypercube(GMRF_General):
    """n-dimensional hypercube graph."""
    def __init__(self, n, s, sigma):
        self.G = nx.hypercube_graph(n)
        super().__init__(self.G, s, sigma)


class GMRF_Torus(GMRF_General):
    """2D torus (grid with periodic boundaries)."""
    def __init__(self, n, s, sigma):
        G_2d = nx.grid_2d_graph(n, n, periodic=True)
        # remap (i,j) -> i*n + j so all downstream code sees integer indices
        mapping = {(i, j): i * n + j for i, j in G_2d.nodes()}
        self.G = nx.relabel_nodes(G_2d, mapping)
        self.n = n
        super().__init__(self.G, s, sigma)


class GMRF_Star(GMRF_General):
    """Star graph (one central node connected to n peripheral nodes)."""
    def __init__(self, n, s, sigma):
        self.G = nx.star_graph(n)
        super().__init__(self.G, s, sigma)


class GMRF_Heterogeneous_Star(GMRF_General):
    """Heterogeneous star graph with random variances."""
    def __init__(self, n, s_alpha=2, s_beta=5, sigma_alpha=2, sigma_beta=5, 
                 min_s=0.5, max_s=2, min_sigma=0.1, max_sigma=0.5):
        self.G = nx.star_graph(n)

        s_nodes = min_s + (max_s - min_s) * np.random.beta(
            s_alpha, s_beta, len(self.G.nodes())
        )
        sigma_edges = min_sigma + (max_sigma - min_sigma) * np.random.beta(
            sigma_alpha, sigma_beta, len(self.G.edges())
        )
        super().__init__(self.G, s_nodes, sigma_edges)


# ==========================================
# ADDITIONAL GRAPH TYPES
# ==========================================
class GMRF_Erdos_Renyi(GMRF_General):
    """Erdős-Rényi random graph G(n,p)."""
    def __init__(self, n, p, s, sigma, seed=None):
        self.n = n
        self.p = p
        self.G = nx.erdos_renyi_graph(n, p, seed=seed)
        
        # Ensure connected
        if not nx.is_connected(self.G):
            print("Warning: Generated graph is not connected. Using largest component.")
            self.G = self.G.subgraph(max(nx.connected_components(self.G), key=len)).copy()
        
        super().__init__(self.G, s, sigma)


class GMRF_Barabasi_Albert(GMRF_General):
    """Barabási-Albert preferential attachment graph."""
    def __init__(self, n, m, s, sigma, seed=None):
        """
        Args:
            n: number of nodes
            m: number of edges to attach from new node
            s, sigma: variances
        """
        self.n = n
        self.m = m
        self.G = nx.barabasi_albert_graph(n, m, seed=seed)
        super().__init__(self.G, s, sigma)


class GMRF_Watts_Strogatz(GMRF_General):
    """Watts-Strogatz small-world graph."""
    def __init__(self, n, k, p, s, sigma, seed=None):
        """
        Args:
            n: number of nodes
            k: each node connected to k nearest neighbors
            p: rewiring probability
            s, sigma: variances
        """
        self.n = n
        self.k = k
        self.p = p
        self.G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        super().__init__(self.G, s, sigma)
        
class GMRF_Watts_Strogatz_Hetero(GMRF_General):
    """Watts-Strogatz small-world graph."""
    def __init__(self, n, k, p,  s_alpha=1, s_beta=1, sigma_alpha=1, sigma_beta=1, 
                 min_s=0.8, max_s=1.2, min_sigma=0.8, max_sigma=1.2, seed=None):
        """
        Args:
            n: number of nodes
            k: each node connected to k nearest neighbors
            p: rewiring probability
            s, sigma: variances
        """
        self.n = n
        self.k = k
        self.p = p
        self.G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        s_nodes = min_s + (max_s - min_s) * np.random.beta(
            s_alpha, s_beta, len(self.G.nodes())
        )
        sigma_edges = min_sigma + (max_sigma - min_sigma) * np.random.beta(
            sigma_alpha, sigma_beta, len(self.G.edges())
        )
        super().__init__(self.G, s_nodes, sigma_edges)