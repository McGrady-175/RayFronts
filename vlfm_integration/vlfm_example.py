"""
VLFM-Rayfronts Integration Example

This example demonstrates how to directly integrate rayfronts occupancy mapping
into VLFM for enhanced semantic navigation with 3D spatial understanding.
"""

import numpy as np
import torch
import habitat
import habitat_sim
from typing import Dict, Any, List
import argparse
import os

# VLFM imports
try:
    from vlfm.run import VLFMRunner
    from vlfm.mapping.frontier_map import FrontierMap
    from vlfm.policy.frontier_exploration_policy import FrontierExplorationPolicy
    VLFM_AVAILABLE = True
except ImportError:
    print("VLFM not available. Please install it first.")
    VLFM_AVAILABLE = False

# Our integration components
from vlfm_rayfronts_adapter import VLFMRayfrontsMapper, VLFMRayfrontsPolicy


class EnhancedVLFMRunner:
    """
    Enhanced VLFM runner that integrates rayfronts 3D occupancy mapping
    for improved semantic navigation performance.
    """
    
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """
        Args:
            config_path: Path to VLFM configuration file
            checkpoint_path: Path to pre-trained VLFM checkpoint
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Initialize VLFM runner if available
        if VLFM_AVAILABLE:
            self.vlfm_runner = VLFMRunner(config_path)
            if checkpoint_path:
                self.vlfm_runner.load_checkpoint(checkpoint_path)
        else:
            self.vlfm_runner = None
            print("Warning: VLFM not available, using fallback mode")
        
        # Initialize our enhanced mapper
        self.rayfronts_mapper = VLFMRayfrontsMapper(
            map_size_cm=2400,  # 24m x 24m map
            map_resolution=5,   # 5cm per pixel
            vox_size=0.1,      # 10cm voxels
            max_pts_per_frame=1000,
            max_depth_sensing=8.0
        )
        
        # Initialize enhanced policy
        self.enhanced_policy = VLFMRayfrontsPolicy(self.rayfronts_mapper)
        
        # Episode tracking
        self.episode_count = 0
        self.success_count = 0
        self.total_spl = 0.0
        
    def run_episode(self, 
                   env: habitat.Env, 
                   target_object: str,
                   max_steps: int = 500) -> Dict[str, Any]:
        """
        Run a single navigation episode with enhanced VLFM + rayfronts
        
        Args:
            env: Habitat environment
            target_object: Target object to find (e.g., "chair", "table")
            max_steps: Maximum steps per episode
            
        Returns:
            Episode results dictionary
        """
        print(f"\n=== Episode {self.episode_count + 1}: Find '{target_object}' ===")
        
        # Reset environment and get initial observations
        obs = env.reset()
        self.rayfronts_mapper.global_vox_xyz = None
        self.rayfronts_mapper.global_vox_occ = None
        
        # Episode state
        step_count = 0
        episode_reward = 0
        done = False
        path_length = 0
        success = False
        
        # Get initial agent state
        agent_state = env.sim.get_agent_state()
        initial_position = agent_state.position
        
        while not done and step_count < max_steps:
            # Get current observations
            rgb_obs = obs.get('rgb', obs.get('color_sensor'))
            depth_obs = obs.get('depth', obs.get('depth_sensor'))
            
            if rgb_obs is None or depth_obs is None:
                print("Warning: Missing RGB or depth observations")
                break
            
            # Get agent pose
            agent_state = env.sim.get_agent_state()
            agent_pose = {
                'position': agent_state.position,
                'rotation': agent_state.rotation
            }
            
            # Get camera intrinsics (simplified)
            camera_intrinsics = self._get_camera_intrinsics(env)
            
            # Update enhanced mapping
            update_info = self.rayfronts_mapper.update_map(
                rgb_obs=rgb_obs,
                depth_obs=depth_obs,
                agent_pose=agent_pose,
                camera_intrinsics=camera_intrinsics
            )
            
            # Get VLM features (if VLFM available)
            if self.vlfm_runner:
                vlm_features = self._get_vlm_features(rgb_obs, target_object)
            else:
                vlm_features = torch.zeros(512)  # Dummy features
            
            # Enhanced action selection using 3D occupancy + VLM
            action = self._select_enhanced_action(
                obs=obs,
                target_object=target_object,
                vlm_features=vlm_features,
                agent_pose=agent_pose
            )
            
            # Execute action
            prev_position = agent_state.position
            obs = env.step(action)
            
            # Update metrics
            current_position = env.sim.get_agent_state().position
            step_distance = np.linalg.norm(current_position - prev_position)
            path_length += step_distance
            
            # Check if episode is done
            done = obs.get('done', False)
            if done:
                success = obs.get('success', False)
            
            step_count += 1
            
            # Print progress
            if step_count % 50 == 0:
                print(f"Step {step_count}: {update_info['total_3d_voxels']} voxels, "
                      f"{update_info['frontiers_detected']} frontiers")
        
        # Calculate metrics
        episode_metrics = self._calculate_episode_metrics(
            success=success,
            path_length=path_length,
            initial_position=initial_position,
            final_position=env.sim.get_agent_state().position,
            step_count=step_count
        )
        
        # Update overall statistics
        self.episode_count += 1
        if success:
            self.success_count += 1
        self.total_spl += episode_metrics['spl']
        
        print(f"Episode completed: Success={success}, SPL={episode_metrics['spl']:.3f}, "
              f"Steps={step_count}, Path Length={path_length:.2f}m")
        
        return episode_metrics
    
    def _select_enhanced_action(self, 
                               obs: Dict[str, Any],
                               target_object: str,
                               vlm_features: torch.Tensor,
                               agent_pose: Dict[str, Any]) -> int:
        """
        Select action using enhanced VLFM + 3D occupancy information
        """
        # Use VLFM if available
        if self.vlfm_runner:
            try:
                # Get VLFM action
                vlfm_action = self.vlfm_runner.select_action(obs, target_object)
            except Exception as e:
                print(f"VLFM action selection failed: {e}")
                vlfm_action = 0  # Forward as fallback
        else:
            vlfm_action = 0
        
        # Enhance with 3D occupancy information
        current_position = np.array(agent_pose['position'])
        
        # Get enhanced frontier selection
        try:
            target_position, confidence = self.enhanced_policy.select_frontier(
                text_query=f"find a {target_object}",
                vlm_features=vlm_features,
                current_position=current_position
            )
            
            # If we have high confidence in enhanced selection, use it
            if confidence > 0.7:
                enhanced_action = self._position_to_action(current_position, target_position)
                return enhanced_action
        except Exception as e:
            print(f"Enhanced action selection failed: {e}")
        
        # Fallback to VLFM action or simple exploration
        return vlfm_action if vlfm_action is not None else self._exploration_action()
    
    def _position_to_action(self, current_pos: np.ndarray, target_pos: np.ndarray) -> int:
        """Convert target position to habitat action"""
        # Calculate direction to target
        direction = target_pos - current_pos
        direction_2d = np.array([direction[0], direction[2]])  # x, z
        
        if np.linalg.norm(direction_2d) < 0.1:
            return 0  # Move forward if very close
        
        # Normalize direction
        direction_2d = direction_2d / np.linalg.norm(direction_2d)
        
        # Simple action selection based on direction
        # This is simplified - in practice you'd want more sophisticated path planning
        angle = np.arctan2(direction_2d[1], direction_2d[0])
        
        if abs(angle) < np.pi/6:  # Forward
            return 0
        elif angle > 0:  # Turn left
            return 1
        else:  # Turn right
            return 2
    
    def _exploration_action(self) -> int:
        """Simple exploration action when all else fails"""
        return np.random.choice([0, 0, 0, 1, 2])  # Bias toward forward movement
    
    def _get_vlm_features(self, rgb_obs: np.ndarray, target_object: str) -> torch.Tensor:
        """Extract VLM features for current observation"""
        if not self.vlfm_runner:
            return torch.zeros(512)
        
        try:
            # This would depend on VLFM's actual API
            features = self.vlfm_runner.extract_features(rgb_obs, target_object)
            return features
        except Exception as e:
            print(f"VLM feature extraction failed: {e}")
            return torch.zeros(512)
    
    def _get_camera_intrinsics(self, env: habitat.Env) -> np.ndarray:
        """Get camera intrinsics from environment"""
        # Get sensor specifications
        sensor_spec = None
        for sensor_uuid, spec in env.sim.config.agents[0].sensor_specifications.items():
            if 'rgb' in sensor_uuid.lower() or 'color' in sensor_uuid.lower():
                sensor_spec = spec
                break
        
        if sensor_spec is None:
            # Default intrinsics for 640x480 with 90 degree HFOV
            return np.array([
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
        
        # Calculate intrinsics from sensor spec
        width, height = sensor_spec.resolution
        hfov = sensor_spec.hfov
        focal_length = width / (2.0 * np.tan(hfov / 2.0))
        
        intrinsics = np.array([
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        
        return intrinsics
    
    def _calculate_episode_metrics(self, 
                                  success: bool,
                                  path_length: float,
                                  initial_position: np.ndarray,
                                  final_position: np.ndarray,
                                  step_count: int) -> Dict[str, float]:
        """Calculate episode performance metrics"""
        # Geodesic distance (simplified as Euclidean for now)
        geodesic_distance = np.linalg.norm(final_position - initial_position)
        
        # SPL (Success weighted by Path Length)
        if success and path_length > 0:
            spl = min(geodesic_distance / path_length, 1.0)
        else:
            spl = 0.0
        
        return {
            'success': success,
            'spl': spl,
            'path_length': path_length,
            'geodesic_distance': geodesic_distance,
            'step_count': step_count
        }
    
    def run_evaluation(self, 
                      env: habitat.Env,
                      episode_list: List[str],
                      num_episodes: int = None) -> Dict[str, float]:
        """
        Run evaluation over multiple episodes
        
        Args:
            env: Habitat environment
            episode_list: List of target objects for episodes
            num_episodes: Number of episodes to run (None for all)
            
        Returns:
            Overall evaluation metrics
        """
        if num_episodes:
            episode_list = episode_list[:num_episodes]
        
        print(f"\n=== Starting Enhanced VLFM Evaluation: {len(episode_list)} episodes ===")
        
        results = []
        for i, target_object in enumerate(episode_list):
            episode_result = self.run_episode(env, target_object)
            results.append(episode_result)
            
            # Print intermediate results
            current_success_rate = self.success_count / self.episode_count
            current_avg_spl = self.total_spl / self.episode_count
            print(f"Progress: {i+1}/{len(episode_list)} episodes, "
                  f"Success Rate: {current_success_rate:.3f}, "
                  f"Average SPL: {current_avg_spl:.3f}")
        
        # Calculate final metrics
        final_metrics = {
            'success_rate': self.success_count / self.episode_count,
            'average_spl': self.total_spl / self.episode_count,
            'total_episodes': self.episode_count,
            'successful_episodes': self.success_count
        }
        
        print(f"\n=== Final Evaluation Results ===")
        print(f"Success Rate: {final_metrics['success_rate']:.3f}")
        print(f"Average SPL: {final_metrics['average_spl']:.3f}")
        print(f"Total Episodes: {final_metrics['total_episodes']}")
        
        return final_metrics
    
    def save_results(self, filepath: str):
        """Save evaluation results and maps"""
        # Save enhanced maps
        map_filepath = filepath.replace('.json', '_maps.pt')
        self.rayfronts_mapper.save_maps(map_filepath)
        
        # Save evaluation metrics
        results = {
            'success_rate': self.success_count / max(self.episode_count, 1),
            'average_spl': self.total_spl / max(self.episode_count, 1),
            'total_episodes': self.episode_count,
            'config': {
                'vox_size': self.rayfronts_mapper.vox_size,
                'map_size_cm': self.rayfronts_mapper.map_size_cm,
                'map_resolution': self.rayfronts_mapper.map_resolution
            }
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")


def create_enhanced_vlfm_config():
    """Create configuration for enhanced VLFM"""
    config = {
        'habitat': {
            'simulator': {
                'type': 'Sim-v0',
                'action_space_config': 'v0',
                'forward_step_size': 0.25,
                'scene': 'data/scene_datasets/hm3d/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb',
                'agents': {
                    'main_agent': {
                        'height': 1.5,
                        'radius': 0.1,
                        'sim_sensors': {
                            'rgb_sensor': {
                                'type': 'HabitatSimRGBSensor',
                                'height': 480,
                                'width': 640,
                                'hfov': 90
                            },
                            'depth_sensor': {
                                'type': 'HabitatSimDepthSensor',
                                'height': 480,
                                'width': 640,
                                'hfov': 90,
                                'min_depth': 0.0,
                                'max_depth': 10.0
                            }
                        }
                    }
                }
            },
            'task': {
                'type': 'ObjectNav-v1',
                'actions': {
                    'move_forward': {'type': 'MoveForwardAction'},
                    'turn_left': {'type': 'TurnLeftAction'},
                    'turn_right': {'type': 'TurnRightAction'},
                    'stop': {'type': 'StopAction'}
                }
            }
        },
        'vlfm': {
            'model_path': 'data/models/vlfm_model.pth',
            'value_map_size': 480,
            'frontier_dilate_occ': 3,
            'frontier_min_size': 10
        },
        'rayfronts': {
            'vox_size': 0.1,
            'max_pts_per_frame': 1000,
            'max_depth_sensing': 8.0,
            'map_size_cm': 2400,
            'map_resolution': 5
        }
    }
    
    return config


def main():
    """Main function for running enhanced VLFM evaluation"""
    parser = argparse.ArgumentParser(description="Enhanced VLFM with Rayfronts Integration")
    parser.add_argument("--config", type=str, default="config/enhanced_vlfm.yaml",
                       help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to VLFM checkpoint")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to run")
    parser.add_argument("--output", type=str, default="results/enhanced_vlfm_results.json",
                       help="Output file for results")
    parser.add_argument("--scene", type=str, 
                       default="data/scene_datasets/hm3d/train/00009-vLpv2VX547B/vLpv2VX547B.basis.glb",
                       help="Scene file path")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create enhanced VLFM runner
    runner = EnhancedVLFMRunner(args.config, args.checkpoint)
    
    # Create Habitat environment
    config = habitat.get_config()
    config.habitat.simulator.scene = args.scene
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = 480
    config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = 640
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = 480
    config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = 640
    
    env = habitat.Env(config=config)
    
    # Define target objects for evaluation
    target_objects = [
        "chair", "table", "sofa", "bed", "toilet", 
        "television", "refrigerator", "book", "clock", "vase"
    ] * (args.num_episodes // 10 + 1)
    target_objects = target_objects[:args.num_episodes]
    
    try:
        # Run evaluation
        results = runner.run_evaluation(env, target_objects, args.num_episodes)
        
        # Save results
        runner.save_results(args.output)
        
        print(f"\nEvaluation completed successfully!")
        print(f"Final Success Rate: {results['success_rate']:.3f}")
        print(f"Final Average SPL: {results['average_spl']:.3f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()


if __name__ == "__main__":
    main()