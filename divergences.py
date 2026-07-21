import numpy as np
from scipy.linalg import sqrtm

# ==========================================
# CORRECTED: KL Divergence
# ==========================================
def emp_kl_div_gaussian(samples, Q):
    """
    Compute KL(Q||P) where Q=empirical distribution from samples, P=true Gaussian.
    
    This measures "how far are samples from truth" (correct direction for convergence).
    
    KL(Q||P) = 0.5 * [tr(Σ_p^{-1} Σ_q) + (μ_p - μ_q)^T Σ_p^{-1} (μ_p - μ_q) - d + ln(|Σ_p|/|Σ_q|)]
    
    Args:
        samples: (n_samples, d) array
        Q: (d, d) precision matrix (can be sparse)
    
    Returns:
        Scalar KL divergence
    """
    samp_mean = samples.mean(0)
    true_mean = np.zeros_like(samp_mean)
    samp_cov = np.cov(samples, rowvar=False)
    
    # Add small regularization for numerical stability
    d = samp_cov.shape[0]
    samp_cov = samp_cov + 1e-8 * np.eye(d)
    
    true_cov = np.linalg.inv(Q.toarray() if hasattr(Q, 'toarray') else Q)
    
    # Log determinants
    sign_true, logdet_true = np.linalg.slogdet(true_cov)
    sign_samp, logdet_samp = np.linalg.slogdet(samp_cov)
    
    if sign_true <= 0 or sign_samp <= 0:
        print(f"Warning: Non-positive definite covariance detected!")
        return np.nan
    
    # Mean difference
    diff = samp_mean - true_mean
    
    # For KL(sample||true), we need true_cov_inv
    true_cov_inv = Q.toarray() if hasattr(Q, 'toarray') else Q
    
    kl = 0.5 * (
        np.trace(true_cov_inv @ samp_cov) +
        diff @ true_cov_inv @ diff -
        d +
        logdet_true - logdet_samp
    )
    
    return kl


# ==========================================
# MMD with RBF Kernel
# ==========================================
def mmd_gaussian(samples, Q, sigma=1.0):
    """
    Compute MMD between samples and Gaussian target N(0, Q^{-1}).
    
    Args:
        samples: (n, d) array
        Q: (d, d) precision matrix
        sigma: RBF kernel bandwidth
    
    Returns:
        Scalar MMD value
    """
    n, d = samples.shape

    # Convert precision to covariance
    true_cov = np.linalg.inv(Q.toarray() if hasattr(Q, "toarray") else Q)

    # 1) E_{x,x' ~ N(0,Sigma)}[k(x,x')]
    A = np.eye(d) + 2 * true_cov / sigma**2
    term1 = 1.0 / np.sqrt(np.linalg.det(A))

    # 2) Empirical U-statistic for sample-sample term
    # Compute pairwise squared distances efficiently
    sq_dists = (
        np.sum(samples**2, axis=1, keepdims=True)
        - 2 * samples @ samples.T
        + np.sum(samples**2, axis=1)
    )
    K_yy = np.exp(-sq_dists / (2 * sigma**2))
    term2 = (np.sum(K_yy) - np.trace(K_yy)) / (n * (n - 1))

    # 3) Target-sample cross terms E_{x~N, y~empirical}[k(x,y)]
    B = true_cov + sigma**2 * np.eye(d)
    B_inv = np.linalg.inv(B)

    # Prefactor = 1/sqrt(det(I + Sigma/sigma^2))
    C = np.eye(d) + true_cov / sigma**2
    prefactor = 1.0 / np.sqrt(np.linalg.det(C))

    # Vector of squared Mahalanobis distances under B_inv
    sq_maha = np.sum(samples @ B_inv * samples, axis=1)
    term3 = prefactor * np.exp(-0.5 * sq_maha).mean()

    # Assemble MMD^2
    mmd2 = term1 + term2 - 2 * term3

    return np.sqrt(max(0, mmd2))


def mmd_gaussian_with_median_heuristic(samples, Q):
    """
    Compute MMD using median heuristic for bandwidth selection.
    
    Args:
        samples: (n, d) array
        Q: (d, d) precision matrix
    
    Returns:
        Scalar MMD value with optimal sigma
    """
    n, d = samples.shape

    # Compute pairwise distances for bandwidth selection
    sq_dists = (
        np.sum(samples**2, axis=1, keepdims=True)
        - 2 * samples @ samples.T
        + np.sum(samples**2, axis=1)
    )

    # Get distances between distinct pairs only
    i_upper = np.triu_indices(n, k=1)
    dist_vals = np.sqrt(sq_dists[i_upper])

    # Median heuristic: sigma = median of pairwise distances
    sigma = np.median(dist_vals)

    # Fallback if median is zero
    if sigma <= 0:
        sigma = 1.0

    # Compute MMD with selected bandwidth
    return mmd_gaussian(samples, Q, sigma=sigma)


# ==========================================
# Fréchet Inception Distance (FID)
# ==========================================
def fid_gaussian(samples, Q):
    """
    Compute Fréchet Inception Distance (FID) between samples and target Gaussian.
    
    For Gaussians, FID has a closed form:
    FID = ||μ_1 - μ_2||² + Tr(Σ_1 + Σ_2 - 2(Σ_1 Σ_2)^{1/2})
    
    Args:
        samples: (n, d) array
        Q: (d, d) precision matrix
    
    Returns:
        Scalar FID value
    """
    # Empirical statistics
    samp_mean = samples.mean(axis=0)
    samp_cov = np.cov(samples, rowvar=False)
    
    # Add small regularization for numerical stability
    d = samp_cov.shape[0]
    samp_cov = samp_cov + 1e-8 * np.eye(d)
    
    # True statistics (zero mean Gaussian)
    true_mean = np.zeros_like(samp_mean)
    true_cov = np.linalg.inv(Q.toarray() if hasattr(Q, "toarray") else Q)
    
    # Mean difference term
    mean_diff = samp_mean - true_mean
    mean_term = np.dot(mean_diff, mean_diff)
    
    # Covariance term: Tr(Σ_1 + Σ_2 - 2(Σ_1 Σ_2)^{1/2})
    # Use eigendecomposition for better numerical stability
    M = samp_cov @ true_cov
    
    # Check if matrix is nearly symmetric
    if not np.allclose(M, M.T):
        M = (M + M.T) / 2  # Symmetrize
    
    try:
        # Eigendecomposition is more stable than sqrtm for PSD matrices
        eigvals, eigvecs = np.linalg.eigh(M)
        # Clip small negative eigenvalues (numerical errors)
        eigvals = np.maximum(eigvals, 0)
        sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    except:
        # Fallback to sqrtm if eigendecomposition fails
        sqrt_product = sqrtm(M)
        if np.iscomplexobj(sqrt_product):
            sqrt_product = sqrt_product.real
    
    # Trace term
    cov_term = np.trace(samp_cov + true_cov - 2 * sqrt_product)
    
    # FID
    fid = mean_term + cov_term
    
    # Ensure non-negative (numerical stability)
    return max(0.0, fid)


# ==========================================
# Wasserstein-2 Distance (for Gaussians)
# ==========================================
def wasserstein2_gaussian(samples, Q):
    """
    Compute squared Wasserstein-2 distance between samples and target Gaussian.
    
    For Gaussians, W_2^2 has the same formula as FID.
    
    Args:
        samples: (n, d) array
        Q: (d, d) precision matrix
    
    Returns:
        Scalar W_2^2 value
    """
    return fid_gaussian(samples, Q)