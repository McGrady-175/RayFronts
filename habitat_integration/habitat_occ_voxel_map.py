"""
Habitat-compatible Occupancy Voxel Map

This module provides a Habitat-optimized version of rayfronts OccVoxelMap
that works seamlessly with Habitat simulation environments.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod

from habitat_adapter import HabitatEnvironmentWrapper
from geometry3d_habitat import (
    habitat_depth_to_sparse_occupancy_voxels,
    pointcloud_to_sparse_voxels
)


class HabitatMappingBase(ABC):
    """Base class for Habitat-compatible mapping systems"""
    
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 clip_bbox: Optional[Tuple[Tuple[float, float, float], 
                                          Tuple[float, float, float]]] = None):
        """
        Args:
            device: PyTorch device for computations
            clip_bbox: Optional bounding box to limit mapping region
        """
        self.device = device
        self.clip_bbox = clip_bbox
        
        if self.clip_bbox is not None:
            self.clip_bbox = torch.tensor(self.clip_bbox, device=device)
    
    @abstractmethod
    def process_habitat_obs(self, observations: Dict[str, torch.FloatTensor]) -> Dict[str, Any]:
        """Process Habitat observations and update map"""
        pass
    
    def _clip_points(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clip points to bounding box if specified"""
        if self.clip_bbox is None:
            return xyz, features
            
        min_bound, max_bound = self.clip_bbox[0], self.clip_bbox[1]
        
        # Create mask for points within bounds
        mask = torch.all(xyz >= min_bound, dim=-1) & torch.all(xyz <= max_bound, dim=-1)
        
        return xyz[mask], features[mask]


class HabitatOccVoxelMap(HabitatMappingBase):
    """Habitat-compatible Occupancy Voxel Map
    
    This class provides the same functionality as rayfronts OccVoxelMap
    but optimized for Habitat environments.
    """
    
    def __init__(self,
                 vox_size: float = 0.1,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 clip_bbox: Optional[Tuple[Tuple[float, float, float], 
                                          Tuple[float, float, float]]] = None,
                 max_pts_per_frame: int = 1000,
                 max_empty_pts_per_frame: int = 1000,
                 max_depth_sensing: float = 10.0,
                 max_empty_cnt: float = -3.0,
                 max_occ_cnt: float = 3.0,
                 occ_observ_weight: float = 5.0,
                 occ_thickness: int = 1,
                 vox_accum_period: int = 1):
        """
        Args:
            vox_size: Size of each voxel in meters
            device: PyTorch device for computations
            clip_bbox: Optional bounding box [(min_x,min_y,min_z), (max_x,max_y,max_z)]
            max_pts_per_frame: Maximum occupied points per frame
            max_empty_pts_per_frame: Maximum empty points per frame
            max_depth_sensing: Maximum sensing range in meters
            max_empty_cnt: Maximum log-odds for empty voxels
            max_occ_cnt: Maximum log-odds for occupied voxels
            occ_observ_weight: Weight for occupied observations
            occ_thickness: Thickness of occupied surfaces in voxels
            vox_accum_period: Frames to accumulate before voxelization
        """
        super().__init__(device, clip_bbox)
        
        # Mapping parameters
        self.vox_size = vox_size
        self.max_pts_per_frame = max_pts_per_frame
        self.max_empty_pts_per_frame = max_empty_pts_per_frame
        self.max_depth_sensing = max_depth_sensing
        self.max_empty_cnt = max_empty_cnt
        self.max_occ_cnt = max_occ_cnt
        self.occ_observ_weight = occ_observ_weight
        self.occ_thickness = occ_thickness
        self.vox_accum_period = vox_accum_period
        
        # Map state
        self.global_vox_xyz: Optional[torch.Tensor] = None
        self.global_vox_occ: Optional[torch.Tensor] = None
        
        # Temporary accumulation buffers
        self._tmp_vox_xyz = []
        self._tmp_vox_occ = []
        self._vox_accum_cnt = 0
        
        # Statistics
        self.total_voxels = 0
        self.frames_processed = 0
    
    def process_habitat_obs(self, observations: Dict[str, torch.FloatTensor]) -> Dict[str, Any]:
        """Process Habitat observations and update occupancy map
        
        Args:
            observations: Dictionary containing:
                - 'rgb_img': RGB image [B, 3, H, W]
                - 'depth_img': Depth image [B, 1, H, W]
                - 'pose_4x4': Camera pose [B, 4, 4]
                - 'intrinsics_3x3': Camera intrinsics [3, 3]
                - 'conf_map': Optional confidence map [B, 1, H, W]
                
        Returns:
            Dictionary with update information
        """
        update_info = {}
        
        # Extract required data
        rgb_img = observations['rgb_img']
        depth_img = observations['depth_img']
        pose_4x4 = observations['pose_4x4']
        intrinsics_3x3 = observations['intrinsics_3x3']
        conf_map = observations.get('conf_map', None)
        
        # Generate occupancy voxels
        vox_xyz, vox_occ = habitat_depth_to_sparse_occupancy_voxels(
            depth_img=depth_img,
            pose_4x4=pose_4x4,
            intrinsics_3x3=intrinsics_3x3,
            vox_size=self.vox_size,
            conf_map=conf_map,
            max_num_pts=self.max_pts_per_frame,
            max_num_empty_pts=self.max_empty_pts_per_frame,
            max_depth_sensing=self.max_depth_sensing,
            occ_thickness=self.occ_thickness,
            algorithm="frustum_culling"
        )
        
        # Clip to bounding box if specified
        vox_xyz, vox_occ = self._clip_points(vox_xyz, vox_occ)
        
        # Convert occupancy from [0,1] to log-odds [-1, occ_observ_weight]
        vox_occ = vox_occ * self.occ_observ_weight - 1
        
        # Accumulate voxels
        B = depth_img.shape[0]
        self._vox_accum_cnt += B
        self._tmp_vox_xyz.append(vox_xyz)
        self._tmp_vox_occ.append(vox_occ)
        
        # Update global map if accumulation period reached
        if self._vox_accum_cnt >= self.vox_accum_period:
            self._vox_accum_cnt = 0
            self._accumulate_voxels()
            
        self.frames_processed += B
        
        update_info.update({
            'new_voxels': len(vox_xyz),
            'total_voxels': self.total_voxels,
            'frames_processed': self.frames_processed
        })
        
        return update_info
    
    def _accumulate_voxels(self) -> None:
        """Accumulate temporarily stored voxels into global map"""
        if len(self._tmp_vox_xyz) == 0:
            return
            
        # Include existing global voxels if they exist
        if self.global_vox_xyz is not None:
            self._tmp_vox_xyz.append(self.global_vox_xyz)
            self._tmp_vox_occ.append(self.global_vox_occ)
            
        # Concatenate all voxels
        pts_xyz = torch.cat(self._tmp_vox_xyz, dim=0)
        pts_occ = torch.cat(self._tmp_vox_occ, dim=0)
        
        # Clear temporary buffers
        self._tmp_vox_xyz.clear()
        self._tmp_vox_occ.clear()
        
        # Voxelize with log-odds aggregation
        self.global_vox_xyz, self.global_vox_occ = pointcloud_to_sparse_voxels(
            pts_xyz, feat_pc=pts_occ, vox_size=self.vox_size, aggregation="sum"
        )
        
        # Clamp log-odds to prevent overflow
        self.global_vox_occ = torch.clamp(
            self.global_vox_occ, 
            min=self.max_empty_cnt, 
            max=self.max_occ_cnt
        )
        
        self.total_voxels = len(self.global_vox_xyz)
    
    def get_occupancy_map(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current occupancy map
        
        Returns:
            Tuple of (voxel_centers, occupancy_log_odds)
        """
        return self.global_vox_xyz, self.global_vox_occ
    
    def get_occupancy_probabilities(self) -> Optional[torch.Tensor]:
        """Convert log-odds to probabilities
        
        Returns:
            Occupancy probabilities in range [0, 1]
        """
        if self.global_vox_occ is None:
            return None
            
        # Convert log-odds to probabilities: p = exp(log_odds) / (1 + exp(log_odds))
        probabilities = torch.sigmoid(self.global_vox_occ)
        return probabilities
    
    def save_map(self, filepath: str) -> None:
        """Save occupancy map to file"""
        if self.global_vox_xyz is None:
            print("Warning: No map data to save")
            return
            
        save_data = {
            'vox_xyz': self.global_vox_xyz.cpu(),
            'vox_occ': self.global_vox_occ.cpu(),
            'vox_size': self.vox_size,
            'total_voxels': self.total_voxels,
            'frames_processed': self.frames_processed,
            'config': {
                'max_pts_per_frame': self.max_pts_per_frame,
                'max_empty_pts_per_frame': self.max_empty_pts_per_frame,
                'max_depth_sensing': self.max_depth_sensing,
                'max_empty_cnt': self.max_empty_cnt,
                'max_occ_cnt': self.max_occ_cnt,
                'occ_observ_weight': self.occ_observ_weight,
                'occ_thickness': self.occ_thickness
            }
        }
        
        torch.save(save_data, filepath)
        print(f"Map saved to {filepath}")
    
    def load_map(self, filepath: str) -> None:
        """Load occupancy map from file"""
        save_data = torch.load(filepath, map_location=self.device)
        
        self.global_vox_xyz = save_data['vox_xyz'].to(self.device)
        self.global_vox_occ = save_data['vox_occ'].to(self.device)
        self.vox_size = save_data['vox_size']
        self.total_voxels = save_data['total_voxels']
        self.frames_processed = save_data['frames_processed']
        
        print(f"Map loaded from {filepath}")
        print(f"Loaded {self.total_voxels} voxels from {self.frames_processed} frames")
    
    def reset_map(self) -> None:
        """Reset/clear the occupancy map"""
        self.global_vox_xyz = None
        self.global_vox_occ = None
        self._tmp_vox_xyz.clear()
        self._tmp_vox_occ.clear()
        self._vox_accum_cnt = 0
        self.total_voxels = 0
        self.frames_processed = 0
    
    def is_empty(self) -> bool:
        """Check if map is empty"""
        return self.global_vox_xyz is None or self.global_vox_xyz.shape[0] == 0
    
    def get_map_stats(self) -> Dict[str, Any]:
        """Get mapping statistics"""
        stats = {
            'total_voxels': self.total_voxels,
            'frames_processed': self.frames_processed,
            'vox_size': self.vox_size,
            'is_empty': self.is_empty()
        }
        
        if not self.is_empty():
            # Compute map bounds
            min_bounds = self.global_vox_xyz.min(dim=0).values
            max_bounds = self.global_vox_xyz.max(dim=0).values
            
            stats.update({
                'map_bounds_min': min_bounds.cpu().tolist(),
                'map_bounds_max': max_bounds.cpu().tolist(),
                'map_volume': ((max_bounds - min_bounds) * self.vox_size).prod().item(),
                'occupancy_stats': {
                    'min_log_odds': self.global_vox_occ.min().item(),
                    'max_log_odds': self.global_vox_occ.max().item(),
                    'mean_log_odds': self.global_vox_occ.mean().item()
                }
            })
            
        return stats


class HabitatMappingPipeline:
    """Complete mapping pipeline for Habitat environments"""
    
    def __init__(self, 
                 habitat_config_path: str,
                 mapper_config: Dict[str, Any] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            habitat_config_path: Path to Habitat environment configuration
            mapper_config: Configuration for the occupancy mapper
            device: PyTorch device
        """
        self.device = device
        
        # Initialize Habitat environment wrapper
        self.env_wrapper = HabitatEnvironmentWrapper(habitat_config_path, device)
        
        # Initialize mapper with default or provided config
        mapper_config = mapper_config or {}
        self.mapper = HabitatOccVoxelMap(device=device, **mapper_config)
        
        # Pipeline state
        self.is_running = False
        self.step_count = 0
    
    def start_mapping(self) -> None:
        """Start the mapping pipeline"""
        self.is_running = True
        self.step_count = 0
        
        # Reset environment and mapper
        observations = self.env_wrapper.reset()
        self.mapper.reset_map()
        
        # Process initial observation
        update_info = self.mapper.process_habitat_obs(observations)
        print(f"Mapping started. Initial voxels: {update_info['new_voxels']}")
    
    def step_mapping(self, action: int) -> Dict[str, Any]:
        """Execute one mapping step
        
        Args:
            action: Habitat action to execute
            
        Returns:
            Combined information from environment and mapper
        """
        if not self.is_running:
            raise RuntimeError("Mapping pipeline not started. Call start_mapping() first.")
        
        # Step environment
        observations = self.env_wrapper.step(action)
        
        # Update map
        update_info = self.mapper.process_habitat_obs(observations)
        
        self.step_count += 1
        
        # Combine environment and mapping info
        step_info = {
            'step': self.step_count,
            'mapping': update_info,
            'observations': {k: v for k, v in observations.items() if k != 'habitat_obs'}
        }
        
        return step_info
    
    def stop_mapping(self) -> None:
        """Stop mapping and close environment"""
        self.is_running = False
        self.env_wrapper.close()
        print(f"Mapping stopped after {self.step_count} steps")
    
    def get_map(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current occupancy map"""
        return self.mapper.get_occupancy_map()
    
    def save_map(self, filepath: str) -> None:
        """Save current map"""
        self.mapper.save_map(filepath)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        mapper_stats = self.mapper.get_map_stats()
        pipeline_stats = {
            'pipeline_steps': self.step_count,
            'is_running': self.is_running
        }
        
        return {**mapper_stats, **pipeline_stats}