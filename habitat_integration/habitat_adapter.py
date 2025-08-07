"""
Habitat Adapter for Rayfronts OccVoxelMap Integration

This module provides adapters to integrate rayfronts occupancy mapping
functionality with Habitat simulation environment.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Any
import habitat_sim
import habitat_lab
from habitat_sim.utils.common import quat_from_angle_axis


class HabitatRGBDAdapter:
    """Adapter to convert Habitat observations to rayfronts format"""
    
    def __init__(self, 
                 habitat_config: habitat_lab.Config,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            habitat_config: Habitat environment configuration
            device: PyTorch device for tensor operations
        """
        self.device = device
        self.habitat_config = habitat_config
        
        # Cache for camera intrinsics
        self._intrinsics_3x3 = None
        
    def get_camera_intrinsics(self, sim: habitat_sim.Simulator) -> torch.FloatTensor:
        """Extract camera intrinsics from Habitat simulator
        
        Returns:
            3x3 camera intrinsics matrix in PyTorch format
        """
        if self._intrinsics_3x3 is None:
            # Get camera sensor specification
            for agent_id in sim.config.agents:
                agent_config = sim.config.agents[agent_id]
                for sensor_id in agent_config.sensor_specifications:
                    sensor_spec = agent_config.sensor_specifications[sensor_id]
                    if sensor_spec.sensor_type == habitat_sim.SensorType.COLOR:
                        # Extract intrinsics from sensor spec
                        width = sensor_spec.resolution[0]
                        height = sensor_spec.resolution[1]
                        hfov = sensor_spec.hfov
                        
                        # Convert HFOV to focal length
                        focal_length = width / (2.0 * np.tan(hfov / 2.0))
                        
                        # Create intrinsics matrix
                        intrinsics = np.array([
                            [focal_length, 0.0, width / 2.0],
                            [0.0, focal_length, height / 2.0], 
                            [0.0, 0.0, 1.0]
                        ], dtype=np.float32)
                        
                        self._intrinsics_3x3 = torch.from_numpy(intrinsics).to(self.device)
                        break
                        
        return self._intrinsics_3x3
    
    def habitat_obs_to_rgbd(self, 
                           observations: Dict[str, Any]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Convert Habitat observations to rayfronts RGBD format
        
        Args:
            observations: Habitat environment observations
            
        Returns:
            rgb_img: RGB image tensor [1, 3, H, W] 
            depth_img: Depth image tensor [1, 1, H, W]
        """
        # Extract RGB
        rgb = observations.get("rgb", observations.get("color_sensor", None))
        if rgb is None:
            raise ValueError("No RGB data found in observations")
            
        # Extract depth
        depth = observations.get("depth", observations.get("depth_sensor", None))
        if depth is None:
            raise ValueError("No depth data found in observations")
            
        # Convert to PyTorch tensors
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).to(self.device)
        if isinstance(depth, np.ndarray):
            depth = torch.from_numpy(depth).to(self.device)
            
        # Ensure correct format: HWC -> CHW and add batch dimension
        if rgb.dim() == 3:  # HWC
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # -> 1CHW
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
            
        # Handle depth format
        if depth.dim() == 2:  # HW
            depth = depth.unsqueeze(0).unsqueeze(0)  # -> 11HW
        elif depth.dim() == 3:  # HW1 or 1HW
            if depth.shape[-1] == 1:  # HW1
                depth = depth.permute(2, 0, 1).unsqueeze(0)  # -> 11HW
            else:  # 1HW
                depth = depth.unsqueeze(0)  # -> 11HW
                
        return rgb, depth
    
    def habitat_pose_to_4x4(self, 
                           agent_state: habitat_sim.AgentState) -> torch.FloatTensor:
        """Convert Habitat agent state to 4x4 pose matrix
        
        Args:
            agent_state: Habitat agent state containing position and rotation
            
        Returns:
            4x4 pose transformation matrix
        """
        position = agent_state.position
        rotation = agent_state.rotation  # Quaternion
        
        # Convert quaternion to rotation matrix
        rot_matrix = habitat_sim.utils.common.quat_to_matrix(rotation)
        
        # Create 4x4 transformation matrix
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :3] = rot_matrix
        pose_4x4[:3, 3] = position
        
        return torch.from_numpy(pose_4x4).unsqueeze(0).to(self.device)  # Add batch dim


class HabitatCoordinateConverter:
    """Handle coordinate system conversion between Habitat and rayfronts"""
    
    @staticmethod
    def habitat_to_rayfronts_pose(habitat_pose: torch.FloatTensor) -> torch.FloatTensor:
        """Convert Habitat coordinate system to rayfronts coordinate system
        
        Habitat typically uses: +Y up, +Z forward, +X right
        Rayfronts (OpenCV): +Y down, +Z forward, +X right
        
        Args:
            habitat_pose: 4x4 pose matrix in Habitat coordinates
            
        Returns:
            4x4 pose matrix in rayfronts coordinates
        """
        # Create conversion matrix from Habitat to OpenCV coordinates
        # This may need adjustment based on actual coordinate conventions
        conversion_matrix = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=habitat_pose.dtype, device=habitat_pose.device)
        
        # Apply conversion
        converted_pose = conversion_matrix @ habitat_pose @ conversion_matrix.T
        return converted_pose
    
    @staticmethod
    def habitat_depth_to_rayfronts(depth: torch.FloatTensor) -> torch.FloatTensor:
        """Convert Habitat depth format to rayfronts expected format
        
        Args:
            depth: Depth image from Habitat
            
        Returns:
            Depth image in rayfronts format
        """
        # Habitat depth is typically in meters, which is what rayfronts expects
        # Handle special values if needed
        depth = depth.clone()
        
        # Convert any invalid depths to appropriate values
        depth[depth <= 0] = float('nan')  # Invalid depths
        depth[torch.isnan(depth)] = float('nan')  # Explicit NaN handling
        
        return depth


class HabitatEnvironmentWrapper:
    """High-level wrapper for using rayfronts mapping with Habitat environment"""
    
    def __init__(self, 
                 habitat_config_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Args:
            habitat_config_path: Path to Habitat environment configuration
            device: PyTorch device
        """
        self.device = device
        
        # Initialize Habitat environment
        self.config = habitat_lab.get_config(habitat_config_path)
        self.env = habitat_lab.make_env(config=self.config)
        
        # Initialize adapters
        self.rgbd_adapter = HabitatRGBDAdapter(self.config, device)
        self.coord_converter = HabitatCoordinateConverter()
        
        # Cache
        self._intrinsics = None
        
    def reset(self) -> Dict[str, torch.FloatTensor]:
        """Reset environment and return initial observations in rayfronts format"""
        habitat_obs = self.env.reset()
        return self._convert_observations(habitat_obs)
        
    def step(self, action: int) -> Dict[str, torch.FloatTensor]:
        """Step environment and return observations in rayfronts format"""
        habitat_obs = self.env.step(action)
        return self._convert_observations(habitat_obs)
        
    def _convert_observations(self, habitat_obs: Dict[str, Any]) -> Dict[str, torch.FloatTensor]:
        """Convert Habitat observations to rayfronts format"""
        # Get RGBD data
        rgb_img, depth_img = self.rgbd_adapter.habitat_obs_to_rgbd(habitat_obs)
        
        # Get agent pose
        agent_state = self.env.sim.get_agent_state()
        pose_4x4 = self.rgbd_adapter.habitat_pose_to_4x4(agent_state)
        
        # Convert coordinates if needed
        pose_4x4 = self.coord_converter.habitat_to_rayfronts_pose(pose_4x4)
        depth_img = self.coord_converter.habitat_depth_to_rayfronts(depth_img)
        
        # Get camera intrinsics
        if self._intrinsics is None:
            self._intrinsics = self.rgbd_adapter.get_camera_intrinsics(self.env.sim)
            
        return {
            'rgb_img': rgb_img,
            'depth_img': depth_img,
            'pose_4x4': pose_4x4,
            'intrinsics_3x3': self._intrinsics,
            'habitat_obs': habitat_obs  # Keep original for reference
        }
        
    def get_action_space(self):
        """Get available actions"""
        return self.env.action_space
        
    def close(self):
        """Close environment"""
        self.env.close()