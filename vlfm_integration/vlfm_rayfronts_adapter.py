"""
VLFM-Rayfronts Integration Adapter

This module provides a direct integration of rayfronts occupancy mapping
into the VLFM (Vision-Language Frontier Maps) framework for enhanced 
semantic navigation with detailed 3D occupancy understanding.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import habitat_sim
from habitat_sim.utils.common import quat_to_matrix

# Import VLFM components (assuming VLFM is installed)
try:
    from vlfm.mapping.frontier_map import FrontierMap
    from vlfm.mapping.obstacle_map import ObstacleMap
    from vlfm.policy.frontier_exploration_policy import FrontierExplorationPolicy
    VLFM_AVAILABLE = True
except ImportError:
    print("VLFM not found. Please install VLFM first.")
    VLFM_AVAILABLE = False

# Import our rayfronts components
from habitat_integration.geometry3d_habitat import (
    habitat_depth_to_sparse_occupancy_voxels,
    pointcloud_to_sparse_voxels
)


class VLFMRayfrontsMapper:
    """
    Enhanced VLFM mapper that integrates rayfronts 3D occupancy mapping
    with VLFM's frontier-based exploration and language-grounded navigation.
    """
    
    def __init__(self,
                 # VLFM parameters
                 map_size_cm: int = 2400,
                 map_resolution: int = 5,  # cm per pixel
                 
                 # Rayfronts 3D mapping parameters
                 vox_size: float = 0.1,
                 max_pts_per_frame: int = 1000,
                 max_empty_pts_per_frame: int = 1000,
                 max_depth_sensing: float = 8.0,
                 
                 # Integration parameters
                 height_thresh: Tuple[float, float] = (-0.5, 2.0),  # Agent-relative height bounds
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            map_size_cm: Size of the VLFM map in centimeters
            map_resolution: Map resolution in cm per pixel
            vox_size: 3D voxel size for rayfronts mapping
            max_pts_per_frame: Maximum points per frame for rayfronts
            max_empty_pts_per_frame: Maximum empty points per frame
            max_depth_sensing: Maximum depth sensing range
            height_thresh: Height bounds for projecting 3D voxels to 2D map
            device: PyTorch device
        """
        self.device = device
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.map_size_pixels = map_size_cm // map_resolution
        
        # Rayfronts 3D mapping parameters
        self.vox_size = vox_size
        self.max_pts_per_frame = max_pts_per_frame
        self.max_empty_pts_per_frame = max_empty_pts_per_frame
        self.max_depth_sensing = max_depth_sensing
        self.height_thresh = height_thresh
        
        # VLFM components
        if VLFM_AVAILABLE:
            self.obstacle_map = ObstacleMap(
                size=self.map_size_pixels,
                resolution=map_resolution
            )
            self.frontier_map = FrontierMap(
                size=self.map_size_pixels,
                resolution=map_resolution
            )
        else:
            # Fallback to our own implementations
            self.obstacle_map = self._create_obstacle_map_fallback()
            self.frontier_map = self._create_frontier_map_fallback()
        
        # 3D occupancy storage
        self.global_vox_xyz: Optional[torch.Tensor] = None
        self.global_vox_occ: Optional[torch.Tensor] = None
        
        # Temporary accumulation for 3D mapping
        self._tmp_vox_xyz = []
        self._tmp_vox_occ = []
        self._vox_accum_cnt = 0
        self.vox_accum_period = 1
        
        # Agent tracking
        self.agent_height = 1.5  # Default agent height
        self.current_pose = None
        
    def update_map(self, 
                   rgb_obs: np.ndarray,
                   depth_obs: np.ndarray, 
                   agent_pose: Dict[str, Any],
                   camera_intrinsics: np.ndarray) -> Dict[str, Any]:
        """
        Update both 2D VLFM maps and 3D rayfronts occupancy from observations
        
        Args:
            rgb_obs: RGB observation [H, W, 3]
            depth_obs: Depth observation [H, W]
            agent_pose: Agent pose {'position': [x,y,z], 'rotation': quaternion}
            camera_intrinsics: Camera intrinsics matrix [3, 3]
            
        Returns:
            Dictionary with update information
        """
        update_info = {}
        
        # Convert inputs to tensors
        rgb_tensor = torch.from_numpy(rgb_obs).float().to(self.device)
        depth_tensor = torch.from_numpy(depth_obs).float().to(self.device)
        intrinsics_tensor = torch.from_numpy(camera_intrinsics).float().to(self.device)
        
        # Format for rayfronts processing
        rgb_batch = rgb_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, 3, H, W]
        depth_batch = depth_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Create pose matrix
        pose_matrix = self._agent_pose_to_matrix(agent_pose)
        pose_4x4 = torch.from_numpy(pose_matrix).float().unsqueeze(0).to(self.device)
        
        # Update 3D occupancy using rayfronts
        vox_xyz, vox_occ = habitat_depth_to_sparse_occupancy_voxels(
            depth_img=depth_batch,
            pose_4x4=pose_4x4,
            intrinsics_3x3=intrinsics_tensor,
            vox_size=self.vox_size,
            max_num_pts=self.max_pts_per_frame,
            max_num_empty_pts=self.max_empty_pts_per_frame,
            max_depth_sensing=self.max_depth_sensing,
            algorithm="frustum_culling"
        )
        
        # Accumulate 3D voxels
        self._tmp_vox_xyz.append(vox_xyz)
        self._tmp_vox_occ.append(vox_occ)
        self._vox_accum_cnt += 1
        
        if self._vox_accum_cnt >= self.vox_accum_period:
            self._accumulate_3d_voxels()
            self._vox_accum_cnt = 0
        
        # Project 3D occupancy to 2D for VLFM integration
        if self.global_vox_xyz is not None:
            obstacle_map_2d = self._project_3d_to_2d_obstacles()
            update_info['obstacle_projection'] = True
        else:
            # Fallback to direct depth projection
            obstacle_map_2d = self._depth_to_2d_obstacles(depth_obs, agent_pose)
            update_info['obstacle_projection'] = False
        
        # Update VLFM obstacle map
        self._update_vlfm_obstacle_map(obstacle_map_2d, agent_pose)
        
        # Update frontier map based on new obstacles
        frontier_map_2d = self._compute_frontiers(obstacle_map_2d)
        self._update_vlfm_frontier_map(frontier_map_2d)
        
        # Store current pose for next iteration
        self.current_pose = agent_pose
        
        update_info.update({
            'new_3d_voxels': len(vox_xyz) if vox_xyz is not None else 0,
            'total_3d_voxels': len(self.global_vox_xyz) if self.global_vox_xyz is not None else 0,
            'frontiers_detected': frontier_map_2d.sum().item() if isinstance(frontier_map_2d, torch.Tensor) else np.sum(frontier_map_2d)
        })
        
        return update_info
    
    def get_frontier_values_for_vlm(self, 
                                   text_queries: List[str],
                                   vlm_features: torch.Tensor) -> torch.Tensor:
        """
        Compute language-grounded frontier values for VLFM exploration
        
        Args:
            text_queries: List of natural language queries (e.g., ["find a chair"])
            vlm_features: Vision-language model features for current view
            
        Returns:
            Frontier value map with language grounding
        """
        if not VLFM_AVAILABLE:
            return self._fallback_frontier_values()
        
        # Get current frontier map
        frontier_map = self.frontier_map.get_frontier_map()
        
        # Language-ground the frontiers using VLM
        # This would typically involve:
        # 1. Extracting visual features at frontier locations
        # 2. Computing similarity with text queries
        # 3. Weighting frontiers by semantic relevance
        
        # Placeholder implementation - would need actual VLM integration
        frontier_values = self._compute_semantic_frontier_values(
            frontier_map, text_queries, vlm_features
        )
        
        return frontier_values
    
    def _accumulate_3d_voxels(self):
        """Accumulate 3D voxels into global map"""
        if len(self._tmp_vox_xyz) == 0:
            return
        
        # Include existing global voxels
        if self.global_vox_xyz is not None:
            self._tmp_vox_xyz.append(self.global_vox_xyz)
            self._tmp_vox_occ.append(self.global_vox_occ)
        
        # Concatenate and voxelize
        pts_xyz = torch.cat(self._tmp_vox_xyz, dim=0)
        pts_occ = torch.cat(self._tmp_vox_occ, dim=0)
        
        self.global_vox_xyz, self.global_vox_occ = pointcloud_to_sparse_voxels(
            pts_xyz, feat_pc=pts_occ, vox_size=self.vox_size, aggregation="sum"
        )
        
        # Clamp occupancy to [0, 1]
        self.global_vox_occ = torch.clamp(self.global_vox_occ, 0, 1)
        
        # Clear temporary storage
        self._tmp_vox_xyz.clear()
        self._tmp_vox_occ.clear()
    
    def _project_3d_to_2d_obstacles(self) -> torch.Tensor:
        """Project 3D occupancy voxels to 2D obstacle map"""
        if self.global_vox_xyz is None:
            return torch.zeros(self.map_size_pixels, self.map_size_pixels, device=self.device)
        
        # Filter voxels by height relative to agent
        agent_height = self.current_pose['position'][1] if self.current_pose else self.agent_height
        voxel_heights = self.global_vox_xyz[:, 1] - agent_height
        
        height_mask = ((voxel_heights >= self.height_thresh[0]) & 
                      (voxel_heights <= self.height_thresh[1]))
        
        if not height_mask.any():
            return torch.zeros(self.map_size_pixels, self.map_size_pixels, device=self.device)
        
        # Get occupancy probabilities for height-filtered voxels
        filtered_xyz = self.global_vox_xyz[height_mask]
        filtered_occ = torch.sigmoid(self.global_vox_occ[height_mask])  # Convert to probabilities
        
        # Project to 2D grid
        map_center = self.map_size_pixels // 2
        
        # Convert world coordinates to map indices
        x_indices = (filtered_xyz[:, 0] / (self.map_resolution / 100.0) + map_center).long()
        z_indices = (filtered_xyz[:, 2] / (self.map_resolution / 100.0) + map_center).long()
        
        # Clip to valid range
        valid_mask = ((x_indices >= 0) & (x_indices < self.map_size_pixels) &
                     (z_indices >= 0) & (z_indices < self.map_size_pixels))
        
        if not valid_mask.any():
            return torch.zeros(self.map_size_pixels, self.map_size_pixels, device=self.device)
        
        x_indices = x_indices[valid_mask]
        z_indices = z_indices[valid_mask]
        occupancy_vals = filtered_occ[valid_mask].squeeze()
        
        # Create 2D obstacle map
        obstacle_map = torch.zeros(self.map_size_pixels, self.map_size_pixels, device=self.device)
        obstacle_map[z_indices, x_indices] = occupancy_vals
        
        return obstacle_map
    
    def _depth_to_2d_obstacles(self, depth_obs: np.ndarray, agent_pose: Dict) -> np.ndarray:
        """Fallback: Direct projection of depth observations to 2D"""
        # Simple projection - replace with more sophisticated method
        H, W = depth_obs.shape
        obstacle_map = np.zeros((self.map_size_pixels, self.map_size_pixels))
        
        # Project depth points to world coordinates and then to map
        # This is a simplified implementation
        center_pixel = self.map_size_pixels // 2
        for v in range(0, H, 10):  # Subsample for efficiency
            for u in range(0, W, 10):
                depth = depth_obs[v, u]
                if 0.1 < depth < self.max_depth_sensing:
                    # Simple forward projection
                    x_world = (u - W//2) * depth * 0.001  # Approximate
                    z_world = depth
                    
                    x_map = int(x_world / (self.map_resolution / 100.0) + center_pixel)
                    z_map = int(z_world / (self.map_resolution / 100.0) + center_pixel)
                    
                    if 0 <= x_map < self.map_size_pixels and 0 <= z_map < self.map_size_pixels:
                        obstacle_map[z_map, x_map] = 1.0
        
        return obstacle_map
    
    def _compute_frontiers(self, obstacle_map: torch.Tensor) -> torch.Tensor:
        """Compute frontier map from obstacle map"""
        if VLFM_AVAILABLE:
            return self.frontier_map.update_frontier_map(obstacle_map)
        else:
            return self._compute_frontiers_fallback(obstacle_map)
    
    def _compute_frontiers_fallback(self, obstacle_map: torch.Tensor) -> torch.Tensor:
        """Fallback frontier computation"""
        # Simple frontier detection: edges between free and unknown space
        if isinstance(obstacle_map, np.ndarray):
            obstacle_map = torch.from_numpy(obstacle_map).to(self.device)
        
        # Create free space map (1 - obstacle_map for known free areas)
        free_map = (obstacle_map < 0.5).float()
        
        # Find frontiers as boundaries between free and unknown space
        frontier_map = torch.zeros_like(obstacle_map)
        
        # Simple edge detection
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                             dtype=torch.float, device=self.device)
        kernel = kernel.view(1, 1, 3, 3)
        
        free_map_batch = free_map.unsqueeze(0).unsqueeze(0)
        edges = torch.nn.functional.conv2d(free_map_batch, kernel, padding=1)
        frontier_map = (edges.squeeze() > 0.5).float()
        
        return frontier_map
    
    def _agent_pose_to_matrix(self, agent_pose: Dict) -> np.ndarray:
        """Convert agent pose to 4x4 transformation matrix"""
        position = np.array(agent_pose['position'])
        rotation = agent_pose['rotation']  # quaternion
        
        # Convert quaternion to rotation matrix
        if hasattr(rotation, 'components'):  # habitat_sim quaternion
            rot_matrix = quat_to_matrix(rotation)
        else:  # assume it's already a numpy array
            # Simple conversion - may need adjustment based on format
            rot_matrix = np.eye(3)
        
        # Create 4x4 transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = position
        
        return pose_matrix
    
    def _update_vlfm_obstacle_map(self, obstacle_map_2d: torch.Tensor, agent_pose: Dict):
        """Update VLFM's obstacle map"""
        if VLFM_AVAILABLE:
            self.obstacle_map.update(obstacle_map_2d, agent_pose)
    
    def _update_vlfm_frontier_map(self, frontier_map_2d: torch.Tensor):
        """Update VLFM's frontier map"""
        if VLFM_AVAILABLE:
            self.frontier_map.update(frontier_map_2d)
    
    def _compute_semantic_frontier_values(self, 
                                        frontier_map: torch.Tensor,
                                        text_queries: List[str],
                                        vlm_features: torch.Tensor) -> torch.Tensor:
        """Compute semantic values for frontiers"""
        # Placeholder for VLM-based semantic frontier evaluation
        # In a real implementation, this would:
        # 1. Extract visual features at frontier locations
        # 2. Compute text-image similarity scores
        # 3. Weight frontiers by semantic relevance
        
        semantic_values = torch.ones_like(frontier_map) * 0.5
        return semantic_values
    
    def _create_obstacle_map_fallback(self):
        """Create fallback obstacle map when VLFM not available"""
        class FallbackObstacleMap:
            def __init__(self, size, resolution):
                self.size = size
                self.resolution = resolution
                self.map = np.zeros((size, size))
            
            def update(self, obstacle_data, pose):
                pass
        
        return FallbackObstacleMap(self.map_size_pixels, self.map_resolution)
    
    def _create_frontier_map_fallback(self):
        """Create fallback frontier map when VLFM not available"""
        class FallbackFrontierMap:
            def __init__(self, size, resolution):
                self.size = size
                self.resolution = resolution
                self.map = np.zeros((size, size))
            
            def update(self, frontier_data):
                pass
            
            def get_frontier_map(self):
                return self.map
        
        return FallbackFrontierMap(self.map_size_pixels, self.map_resolution)
    
    def _fallback_frontier_values(self) -> torch.Tensor:
        """Fallback frontier values when VLFM not available"""
        return torch.ones(self.map_size_pixels, self.map_size_pixels, device=self.device) * 0.5
    
    def get_exploration_policy(self) -> Dict[str, Any]:
        """Get the current exploration policy state"""
        return {
            'obstacle_map': self.obstacle_map.map if hasattr(self.obstacle_map, 'map') else None,
            'frontier_map': self.frontier_map.get_frontier_map() if hasattr(self.frontier_map, 'get_frontier_map') else None,
            'voxel_count': len(self.global_vox_xyz) if self.global_vox_xyz is not None else 0
        }
    
    def save_maps(self, filepath: str):
        """Save both 2D and 3D maps"""
        save_data = {
            '3d_voxels': {
                'xyz': self.global_vox_xyz.cpu() if self.global_vox_xyz is not None else None,
                'occupancy': self.global_vox_occ.cpu() if self.global_vox_occ is not None else None,
                'vox_size': self.vox_size
            },
            '2d_maps': {
                'obstacle_map': self.obstacle_map.map if hasattr(self.obstacle_map, 'map') else None,
                'frontier_map': self.frontier_map.get_frontier_map() if hasattr(self.frontier_map, 'get_frontier_map') else None,
                'map_size_cm': self.map_size_cm,
                'map_resolution': self.map_resolution
            }
        }
        
        torch.save(save_data, filepath)
        print(f"VLFM-Rayfronts maps saved to {filepath}")


class VLFMRayfrontsPolicy:
    """
    Enhanced VLFM exploration policy that leverages both 2D frontier maps
    and 3D occupancy information for better navigation decisions.
    """
    
    def __init__(self, mapper: VLFMRayfrontsMapper):
        self.mapper = mapper
        
    def select_frontier(self, 
                       text_query: str,
                       vlm_features: torch.Tensor,
                       current_position: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Select the best frontier for exploration based on language query
        and 3D occupancy information.
        
        Args:
            text_query: Natural language description of target
            vlm_features: Current vision-language features
            current_position: Agent's current position [x, y, z]
            
        Returns:
            (target_position, confidence_score)
        """
        # Get language-grounded frontier values
        frontier_values = self.mapper.get_frontier_values_for_vlm(
            [text_query], vlm_features
        )
        
        # Enhanced frontier selection using 3D information
        enhanced_values = self._enhance_frontiers_with_3d(frontier_values)
        
        # Select best frontier
        best_idx = torch.argmax(enhanced_values)
        best_coords = np.unravel_index(best_idx.cpu().numpy(), enhanced_values.shape)
        
        # Convert map coordinates to world coordinates
        target_position = self._map_to_world_coords(best_coords)
        confidence_score = enhanced_values[best_coords].item()
        
        return target_position, confidence_score
    
    def _enhance_frontiers_with_3d(self, frontier_values: torch.Tensor) -> torch.Tensor:
        """Enhance frontier values using 3D occupancy information"""
        enhanced = frontier_values.clone()
        
        # Add 3D-based enhancements:
        # 1. Prefer frontiers with good 3D visibility
        # 2. Avoid frontiers near complex 3D obstacles
        # 3. Weight by 3D exploration completeness
        
        if self.mapper.global_vox_xyz is not None:
            # Example enhancement: boost frontiers in less explored 3D regions
            exploration_bonus = self._compute_3d_exploration_bonus(frontier_values)
            enhanced = enhanced + 0.2 * exploration_bonus
        
        return enhanced
    
    def _compute_3d_exploration_bonus(self, frontier_values: torch.Tensor) -> torch.Tensor:
        """Compute exploration bonus based on 3D occupancy density"""
        bonus = torch.zeros_like(frontier_values)
        
        # Simple implementation: lower bonus in densely mapped areas
        if self.mapper.global_vox_xyz is not None:
            # Project 3D voxel density to 2D
            # Areas with fewer voxels get higher exploration bonus
            voxel_density_2d = self._project_voxel_density_to_2d()
            bonus = 1.0 / (1.0 + voxel_density_2d)
        
        return bonus
    
    def _project_voxel_density_to_2d(self) -> torch.Tensor:
        """Project 3D voxel density to 2D map"""
        density_map = torch.zeros(self.mapper.map_size_pixels, self.mapper.map_size_pixels, 
                                 device=self.mapper.device)
        
        if self.mapper.global_vox_xyz is None:
            return density_map
        
        # Convert voxel coordinates to map indices
        map_center = self.mapper.map_size_pixels // 2
        x_indices = (self.mapper.global_vox_xyz[:, 0] / (self.mapper.map_resolution / 100.0) + map_center).long()
        z_indices = (self.mapper.global_vox_xyz[:, 2] / (self.mapper.map_resolution / 100.0) + map_center).long()
        
        # Clip to valid range
        valid_mask = ((x_indices >= 0) & (x_indices < self.mapper.map_size_pixels) &
                     (z_indices >= 0) & (z_indices < self.mapper.map_size_pixels))
        
        if valid_mask.any():
            x_indices = x_indices[valid_mask]
            z_indices = z_indices[valid_mask]
            
            # Count voxels per map cell
            for i in range(len(x_indices)):
                density_map[z_indices[i], x_indices[i]] += 1
        
        return density_map
    
    def _map_to_world_coords(self, map_coords: Tuple[int, int]) -> np.ndarray:
        """Convert map coordinates to world coordinates"""
        map_center = self.mapper.map_size_pixels // 2
        resolution_m = self.mapper.map_resolution / 100.0
        
        x_world = (map_coords[1] - map_center) * resolution_m
        z_world = (map_coords[0] - map_center) * resolution_m
        y_world = self.mapper.agent_height  # Assume ground level
        
        return np.array([x_world, y_world, z_world])