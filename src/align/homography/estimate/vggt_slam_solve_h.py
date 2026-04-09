#Code adapted from 'VGGT-SLAM: Dense RGB SLAM Optimized on the SL (4) Manifold}'
#  with minor modifications
# https://github.com/MIT-SPARK/VGGT-SLAM/blob/version1.0/vggt_slam/h_solve.py
# @article{maggio2025vggt-slam,
#  title={VGGT-SLAM: Dense RGB SLAM Optimized on the SL (4) Manifold},
#  author={Maggio, Dominic and Lim, Hyungtae and Carlone, Luca},
#  journal={Advances in Neural Information Processing Systems},
#  volume={39},
#  year={2025}
#}

import numpy as np
import torch
from scipy.linalg import null_space


def to_homogeneous(X):
    return np.hstack([X, np.ones((X.shape[0], 1))])


def apply_homography(H, X, debug=False):
    X_h = to_homogeneous(X)
    X_trans = (H @ X_h.T).T
    if debug:
        print(X_trans[:, 3])
    return X_trans[:, :3] / X_trans[:, 3:]


def apply_homography_batch(H_batch: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Efficiently apply batched 4x4 homographies to 3D points.
    
    Args:
        H_batch: Tensor of shape (B, 4, 4)
        X:       Tensor of shape (N, 3)
    Returns:
        Transformed points: Tensor of shape (B, N, 3)
    """
    B = H_batch.shape[0]
    N = X.shape[0]
    
    # Append 1 to each point: (N, 4)
    ones = torch.ones((N, 1), dtype=X.dtype, device=X.device)
    X_h = torch.cat([X, ones], dim=1)  # (N, 4)

    # Apply homographies: (B, 4, 4) x (N, 4)^T → (B, 4, N)
    X_h = X_h.T.unsqueeze(0).expand(B, 4, N)  # (B, 4, N)
    X_trans = torch.bmm(H_batch, X_h)  # (B, 4, N)

    # Perspective divide
    X_trans = X_trans[:, :3, :] / X_trans[:, 3:4, :]  # (B, 3, N)
    
    # Transpose to (B, N, 3)
    return X_trans.permute(0, 2, 1)


def estimate_3D_homography(X_src_batch, X_dst_batch):
    """
    Estimate batch of 3D Homography using Batched SVD.
    """
    B, N, _ = X_src_batch.shape
    ones = np.ones((B, N))

    x, y, z = X_src_batch[:, :, 0], X_src_batch[:, :, 1], X_src_batch[:, :, 2]
    xp, yp, zp = X_dst_batch[:, :, 0], X_dst_batch[:, :, 1], X_dst_batch[:, :, 2]

    # Prepare matrices
    A = np.zeros((B, 3 * N, 16))
    stacked_X = np.stack([x, y, z, ones], axis=2)

    A[:, 0::3, 0:4] = -stacked_X
    A[:, 0::3, 12:16] = np.stack([x * xp, y * xp, z * xp, xp], axis=2)

    A[:, 1::3, 4:8] = -stacked_X
    A[:, 1::3, 12:16] = np.stack([x * yp, y * yp, z * yp, yp], axis=2)

    A[:, 2::3, 8:12] = -stacked_X
    A[:, 2::3, 12:16] = np.stack([x * zp, y * zp, z * zp, zp], axis=2)

    # Batched SVD: A is (B, 3N, 16). Vh is (B, 16, 16)
    # The last row of Vh corresponds to the smallest singular value
    _, _, Vh = np.linalg.svd(A)
    H_batch = Vh[:, -1, :].reshape(B, 4, 4)

    # Vectorized normalization and SL(4) projection
    for i in range(B):
        H = H_batch[i]
        
        # Avoid division by zero
        if abs(H[3, 3]) < 1e-8:
            H_batch[i] = np.eye(4)
            continue
            
        H = H / H[3, 3]
        det = np.linalg.det(H)
        
        # Check for invalid determinants (reflections or singular matrices)
        if np.isnan(det) or det < 0.0001:
            H_batch[i] = np.eye(4)
        else:
            H_batch[i] = H / (det**0.25)

    return torch.tensor(H_batch, dtype=torch.float32)


def is_planar(X, threshold=5e-2):
    X_centered = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(X_centered)
    normal_strength = S[-1] / S[0]
    return normal_strength < threshold


def scale(X):
    centroid = X.mean(axis=0)
    X_centered = X - centroid  # move centroid to origin

    # Compute average distance to the origin after centering
    avg_norm = np.linalg.norm(X_centered, axis=1).mean()

    # Desired average distance is sqrt(3)
    desired_avg_norm = np.sqrt(3)

    # Compute the uniform scaling factor
    scale = desired_avg_norm / avg_norm

    # Construct the 4x4 similarity transform matrix
    T = np.eye(4)
    T[:3, :3] *= scale  # apply scaling
    T[:3, 3] = -scale * centroid  # apply translation

    X_h = np.hstack([X, np.ones((X.shape[0], 1))])  # shape: (N, 4)

    # Step 2: Apply the transform
    X_transformed_h = (T @ X_h.T).T  # shape: (N, 4)

    # Step 3: Convert back to 3D (drop the homogeneous coordinate)
    X_transformed = X_transformed_h[:, :3]

    return T, X_transformed


def ransac_projective(X1_np, X2_np, threshold=0.01, max_iter=300, sample_size=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to torch tensors on GPU
    X1 = torch.tensor(X1_np, dtype=torch.float32, device=device)
    X2 = torch.tensor(X2_np, dtype=torch.float32, device=device)
    N = X1.shape[0]

    # Sample indices for each hypothesis.
    indices = torch.randint(0, N, (max_iter, sample_size), device=device)

    # Gather sampled point sets efficiently using advanced indexing.
    X1_samples = X1[indices]  # (max_iter, sample_size, 3)
    X2_samples = X2[indices]  # (max_iter, sample_size, 3)

    # Estimate homographies.
    H_ests = estimate_3D_homography(X1_samples.cpu().numpy(), X2_samples.cpu().numpy())
    H_ests = H_ests.to(device) # FIX: Assign the tensor back to H_ests

    # Apply homographies to all points.
    X2_preds = apply_homography_batch(H_ests, X1)

    # Compute Euclidean error.
    errors = torch.norm(X2_preds - X2[None, :, :], dim=2)

    # Compute inlier masks and counts.
    inlier_masks = errors < threshold  # (max_iter, N)
    inlier_counts = inlier_masks.sum(dim=1)

    # Select best hypothesis
    best_idx = torch.argmax(inlier_counts)
    
    # Optional but highly recommended: Re-estimate using ALL inliers of the best model
    best_inlier_mask = inlier_masks[best_idx]
    
    # Only refine if we actually found more than the minimum sample size
    if best_inlier_mask.sum() > sample_size:
        best_X1_inliers = X1[best_inlier_mask].unsqueeze(0).cpu().numpy() # Shape: (1, Num_Inliers, 3)
        best_X2_inliers = X2[best_inlier_mask].unsqueeze(0).cpu().numpy() # Shape: (1, Num_Inliers, 3)
        
        # Re-run DLT on the full consensus set
        refined_H = estimate_3D_homography(best_X1_inliers, best_X2_inliers)
        best_H = refined_H[0].numpy()
    else:
        # Fallback to the original minimal sample estimate
        best_H = H_ests[best_idx].cpu().numpy()

    return best_H
