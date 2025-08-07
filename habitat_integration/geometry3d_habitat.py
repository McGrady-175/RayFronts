"""
Habitat-optimized version of rayfronts geometry3d functions

This module contains adapted versions of key geometry3d functions
optimized for use with Habitat simulation environment.
"""

import torch
import numpy as np
import torch_scatter
from typing import Tuple


def pts_to_homogen(pts: torch.Tensor) -> torch.Tensor:
    """Convert 3D points to homogeneous coordinates"""
    if pts.shape[-1] != 3:
        raise ValueError(f"Invalid points tensor shape {pts.shape}. Last dim should have length 3")
    return torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)


def mat_3x3_to_4x4(mat: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 matrix to 4x4 transformation matrix"""
    zeros = torch.zeros(size=(*mat.shape[:-2], 3, 1), device=mat.device)
    mat = torch.cat((mat, zeros), dim=-1)
    row = torch.tensor([[0, 0, 0, 1]], device=mat.device, dtype=mat.dtype)
    mat = torch.cat((mat, row.repeat(*mat.shape[:-2], 1, 1)), axis=-2)
    return mat


def transform_points(points: torch.Tensor, transform_mat: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transformation matrix to 3D points"""
    # Handle 3x3 matrices
    if transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 3:
        transform_mat = mat_3x3_to_4x4(transform_mat)
    elif transform_mat.shape[-2] == 3 and transform_mat.shape[-1] == 4:
        row = torch.tensor([[0, 0, 0, 1]], device=transform_mat.device, dtype=transform_mat.dtype)
        transform_mat = torch.cat((transform_mat, row.repeat(*transform_mat.shape[:-2], 1, 1)), axis=-2)
    
    # Convert to homogeneous coordinates if needed
    if points.shape[-1] == 3:
        points_homo = pts_to_homogen(points)
    else:
        points_homo = points
    
    # Apply transformation
    transformed = points_homo @ torch.transpose(transform_mat, -2, -1)
    
    # Convert back to 3D
    return transformed[..., :3]


def pointcloud_to_sparse_voxels(xyz_pc: torch.Tensor, 
                               vox_size: float, 
                               feat_pc: torch.Tensor = None,
                               aggregation: str = "mean", 
                               return_counts: bool = False):
    """Convert point cloud to sparse voxels with feature aggregation
    
    Optimized version for Habitat integration with better memory management.
    """
    device = xyz_pc.device
    
    # Voxelize coordinates
    xyz_vx = torch.round(xyz_pc / vox_size).type(torch.int64)
    
    if feat_pc is None:
        xyz_vx, count_vx = torch.unique(xyz_vx, return_counts=True, dim=0)
        xyz_vx = xyz_vx.type(torch.float) * vox_size
        count_vx = count_vx.type(torch.float).unsqueeze(-1)
        
        if return_counts:
            return xyz_vx, count_vx
        else:
            return xyz_vx
    
    # Handle features
    xyz_vx, reduce_ind, counts_vx = torch.unique(xyz_vx, return_inverse=True, 
                                                return_counts=True, dim=0)
    feat_vx = torch.zeros((xyz_vx.shape[0], feat_pc.shape[-1]), 
                         device=device, dtype=feat_pc.dtype)
    
    # Use torch_scatter for efficient aggregation
    torch_scatter.scatter(src=feat_pc, index=reduce_ind, out=feat_vx, 
                         reduce=aggregation, dim=0)
    
    xyz_vx = xyz_vx.type(torch.float) * vox_size
    counts_vx = counts_vx.type(torch.float).unsqueeze(-1)
    
    if return_counts:
        return xyz_vx, feat_vx, counts_vx
    else:
        return xyz_vx, feat_vx


def habitat_depth_to_sparse_occupancy_voxels(depth_img: torch.FloatTensor,
                                            pose_4x4: torch.FloatTensor,
                                            intrinsics_3x3: torch.FloatTensor,
                                            vox_size: float,
                                            conf_map: torch.FloatTensor = None,
                                            max_num_pts: int = 2000,  # Reduced for Habitat
                                            max_num_empty_pts: int = 2000,  # Reduced for Habitat  
                                            max_depth_sensing: float = 10.0,  # Habitat typical range
                                            occ_thickness: int = 1,
                                            algorithm: str = "frustum_culling",
                                            return_pc: bool = False):
    """
    Habitat-optimized version of depth_to_sparse_occupancy_voxels
    
    Key optimizations for Habitat:
    - Reduced default point limits for better performance
    - Habitat-specific depth range handling
    - Optimized memory usage for real-time mapping
    """
    
    B, _, H, W = depth_img.shape
    device = depth_img.device
    valid_depth_mask = torch.logical_and(torch.isfinite(depth_img), depth_img > 0)
    
    min_depth = vox_size / 2
    if max_depth_sensing > 0:
        max_depth = max_depth_sensing
    else:
        try:
            max_depth = depth_img[valid_depth_mask].max()
            # Clamp to reasonable range for Habitat
            max_depth = min(max_depth.item(), 20.0)
        except RuntimeError:
            max_depth = 10.0  # Default for Habitat environments
    
    # Create image plane points (optimized for Habitat resolution)
    img_xi, img_yi = torch.meshgrid(torch.arange(W, device=device),
                                   torch.arange(H, device=device),
                                   indexing="xy")
    img_xi = img_xi.tile((B, 1, 1))
    img_yi = img_yi.tile((B, 1, 1))
    
    img_plane_pts = torch.stack([
        img_xi.flatten(-2),
        img_yi.flatten(-2),
        torch.ones(B, H*W, device=device),
    ], axis=-1)
    
    unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))
    
    if max_depth <= 0:
        # Return robot position as single occupied voxel
        world_empty_pts_xyz = pose_4x4[:, :3, -1]
        xyz_vx = pointcloud_to_sparse_voxels(world_empty_pts_xyz, vox_size=vox_size)
        occupancy_vx = torch.zeros(1, 1, device=device)
        
        if return_pc:
            world_occ_pts_xyz = torch.empty(0, 3, device=device)
            occ_pts_img_indices = torch.empty(0, dtype=torch.long, device=device)
            return xyz_vx, occupancy_vx, world_occ_pts_xyz, occ_pts_img_indices
        return xyz_vx, occupancy_vx
    
    # Use frustum culling algorithm (optimized for Habitat)
    if algorithm == "frustum_culling":
        bbox_mn, bbox_mx = get_update_bbox_habitat(
            pose_4x4, intrinsics_3x3, resolution=(H, W),
            near=0, far=max_depth
        )
        
        # Create voxel grid
        xx = torch.arange(bbox_mn[0] - vox_size, bbox_mx[0] + vox_size * occ_thickness,
                         vox_size, device=device)
        yy = torch.arange(bbox_mn[1] - vox_size, bbox_mx[1] + vox_size * occ_thickness,
                         vox_size, device=device)
        zz = torch.arange(bbox_mn[2] - vox_size, bbox_mx[2] + vox_size * occ_thickness,
                         vox_size, device=device)
        
        world_bbox_pts = torch.stack(torch.meshgrid(xx, yy, zz, indexing="xy"),
                                   dim=-1).reshape(-1, 3)
        world_bbox_pts = world_bbox_pts.unsqueeze(0).tile(B, 1, 1)
        
        # Transform to camera coordinates
        cam_bbox_pts = transform_points(world_bbox_pts, torch.inverse(pose_4x4))
        
        # Project to image plane
        img_plane_bbox_pts = transform_points(
            cam_bbox_pts, intrinsics_3x3.unsqueeze(0).tile(B, 1, 1))
        img_plane_bbox_pts = img_plane_bbox_pts[..., :2] / img_plane_bbox_pts[..., -1:]
        img_plane_bbox_pts = torch.round(img_plane_bbox_pts).long()
        
        # Frustum culling
        mask = ((img_plane_bbox_pts[..., 0] >= 0) &
                (img_plane_bbox_pts[..., 0] < W) &
                (img_plane_bbox_pts[..., 1] >= 0) &
                (img_plane_bbox_pts[..., 1] < H))
        
        # Convert to flat indices
        img_bbox_indices = W * img_plane_bbox_pts[..., 1] + img_plane_bbox_pts[..., 0]
        img_bbox_indices = img_bbox_indices + (torch.arange(0, B, device=device) * H * W).unsqueeze(-1)
        
        in_frustum_indices = img_bbox_indices[mask]
        world_bbox_pts = world_bbox_pts[mask]
        world_bbox_pts_assoc_depth = depth_img.flatten()[in_frustum_indices]
        world_bbox_pts_actual_depth = cam_bbox_pts[..., -1][mask]
        
        # Determine occupancy
        occ_mask = torch.abs(world_bbox_pts_actual_depth - world_bbox_pts_assoc_depth) < vox_size / 2 * occ_thickness
        
        world_occ_pts_xyz = world_bbox_pts[occ_mask]
        occ_pts_img_indices = in_frustum_indices[occ_mask]
        
        world_observed_pts_xyz = world_bbox_pts[
            (world_bbox_pts_actual_depth < world_bbox_pts_assoc_depth - vox_size) &
            (world_bbox_pts_actual_depth < max_depth - vox_size) &
            (world_bbox_pts_actual_depth > min_depth)]
        
    else:
        raise ValueError(f"Algorithm '{algorithm}' not supported in Habitat version")
    
    # Sample points if too many
    max_num_pts *= B
    if max_num_pts > 0 and max_num_pts < len(occ_pts_img_indices):
        if conf_map is None:
            indices_indices = torch.randperm(len(occ_pts_img_indices), device=device)
        else:
            conf_flat = conf_map.flatten()
            indices_indices = torch.argsort(conf_flat[occ_pts_img_indices], descending=True)
        
        occ_pts_img_indices = occ_pts_img_indices[indices_indices[:max_num_pts]]
        world_occ_pts_xyz = world_occ_pts_xyz[indices_indices[:max_num_pts], :]
    
    max_num_empty_pts *= B
    if max_num_empty_pts > 0 and max_num_empty_pts < world_observed_pts_xyz.shape[0]:
        all_indices = torch.arange(0, world_observed_pts_xyz.shape[0], device=device)
        indices_indices = torch.randperm(len(all_indices), device=device)
        selected_indices = all_indices[indices_indices[:max_num_empty_pts]]
        world_observed_pts_xyz = world_observed_pts_xyz[selected_indices]
    
    # Voxelize
    flat_world_occ_pts_xyz = world_occ_pts_xyz.reshape(-1, 3)
    occupancy_pts = torch.vstack([
        torch.ones_like(flat_world_occ_pts_xyz[..., -1:]),
        torch.zeros_like(world_observed_pts_xyz[..., -1:])
    ])
    
    xyz_pts = torch.vstack([flat_world_occ_pts_xyz, world_observed_pts_xyz])
    xyz_vx, occupancy_vx = pointcloud_to_sparse_voxels(
        xyz_pts, vox_size, feat_pc=occupancy_pts, aggregation="sum")
    
    occupancy_vx = torch.clamp(occupancy_vx, max=1)
    
    if return_pc:
        return xyz_vx, occupancy_vx, world_occ_pts_xyz, occ_pts_img_indices
    return xyz_vx, occupancy_vx


def get_update_bbox_habitat(pose_4x4: torch.FloatTensor,
                           intrinsics_3x3: torch.FloatTensor,
                           resolution: Tuple[int, int],
                           far: float,
                           near: float = 0):
    """Habitat-optimized version of get_update_bbox"""
    H, W = resolution
    B = pose_4x4.shape[0]
    device = pose_4x4.device
    
    plane_pts = lambda d: torch.tensor([
        [0.0, 0.0, 1],
        [0.0, H, 1],
        [W, 0.0, 1],
        [W, H, 1]
    ], dtype=torch.float, device=device) * d
    
    near_plane_pts = plane_pts(near)
    far_plane_pts = plane_pts(far)
    
    all_plane_pts = torch.stack([near_plane_pts, far_plane_pts], dim=0)
    
    unproj_mat = pose_4x4 @ mat_3x3_to_4x4(torch.inverse(intrinsics_3x3))
    all_plane_pts = transform_points(
        all_plane_pts.reshape(1, -1, 3).tile(B, 1, 1), unproj_mat
    ).reshape(B, 8, 3)
    
    mn = all_plane_pts.reshape(-1, 3).min(dim=0).values
    mx = all_plane_pts.reshape(-1, 3).max(dim=0).values
    
    return mn, mx