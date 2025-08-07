"""
Example usage of Habitat-integrated OccVoxelMap

This script demonstrates how to use the migrated rayfronts OccVoxelMap
with Habitat simulation environment for real-time occupancy mapping.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import time

# Import our Habitat integration modules
from habitat_occ_voxel_map import HabitatMappingPipeline, HabitatOccVoxelMap
from habitat_adapter import HabitatEnvironmentWrapper

# Habitat imports
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis


def create_habitat_config():
    """Create a basic Habitat configuration for testing"""
    
    # Create simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    sim_cfg.enable_physics = False
    
    # Create sensor specifications
    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [480, 640]  # H, W
    color_sensor_spec.position = [0.0, 0.0, 0.0]
    color_sensor_spec.hfov = 90.0  # Horizontal field of view in degrees
    
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]  # H, W
    depth_sensor_spec.position = [0.0, 0.0, 0.0]
    depth_sensor_spec.hfov = 90.0
    
    # Create agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [color_sensor_spec, depth_sensor_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def simple_navigation_example():
    """Simple example: Manual navigation and mapping"""
    
    print("=== Simple Navigation and Mapping Example ===")
    
    # Create Habitat configuration
    habitat_config = create_habitat_config()
    
    # Create simulator directly for this example
    sim = habitat_sim.Simulator(habitat_config)
    
    # Initialize our occupancy mapper
    mapper_config = {
        'vox_size': 0.1,  # 10cm voxels
        'max_pts_per_frame': 1000,
        'max_empty_pts_per_frame': 1000,
        'max_depth_sensing': 8.0,  # 8 meter range
        'clip_bbox': ((-10, -3, -10), (10, 3, 10))  # Limit mapping area
    }
    
    mapper = HabitatOccVoxelMap(**mapper_config)
    
    # Helper function to get observations in our format
    def get_observations(sim):
        habitat_obs = sim.get_sensor_observations()
        
        # Convert to torch tensors
        rgb = torch.from_numpy(habitat_obs["color_sensor"]).float() / 255.0
        depth = torch.from_numpy(habitat_obs["depth_sensor"]).float()
        
        # Reshape for batch processing
        rgb = rgb.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
        depth = depth.unsqueeze(0).unsqueeze(0)  # HW -> B1HW
        
        # Get agent pose
        agent_state = sim.get_agent_state()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = habitat_sim.utils.common.quat_to_matrix(agent_state.rotation)
        pose_matrix[:3, 3] = agent_state.position
        pose_4x4 = torch.from_numpy(pose_matrix).float().unsqueeze(0)
        
        # Create intrinsics (simplified)
        H, W = 480, 640
        hfov_rad = np.radians(90.0)
        focal_length = W / (2.0 * np.tan(hfov_rad / 2.0))
        intrinsics = torch.tensor([
            [focal_length, 0.0, W/2.0],
            [0.0, focal_length, H/2.0],
            [0.0, 0.0, 1.0]
        ]).float()
        
        return {
            'rgb_img': rgb,
            'depth_img': depth,
            'pose_4x4': pose_4x4,
            'intrinsics_3x3': intrinsics
        }
    
    # Reset environment
    sim.reset()
    
    # Define a simple exploration sequence
    actions = ["move_forward"] * 5 + ["turn_left"] + ["move_forward"] * 3 + \
              ["turn_right"] + ["move_forward"] * 4 + ["turn_left"] + ["move_forward"] * 2
    
    print(f"Executing {len(actions)} actions for mapping...")
    
    for i, action in enumerate(actions):
        # Get current observations
        obs = get_observations(sim)
        
        # Update map
        update_info = mapper.process_habitat_obs(obs)
        
        # Execute action
        sim.step(action)
        
        # Print progress
        if i % 5 == 0:
            print(f"Step {i+1}/{len(actions)}: {update_info['total_voxels']} total voxels")
    
    # Final statistics
    stats = mapper.get_map_stats()
    print("\n=== Mapping Results ===")
    print(f"Total voxels: {stats['total_voxels']}")
    print(f"Frames processed: {stats['frames_processed']}")
    print(f"Map bounds: {stats.get('map_bounds_min', 'N/A')} to {stats.get('map_bounds_max', 'N/A')}")
    
    # Save map
    output_path = "simple_navigation_map.pt"
    mapper.save_map(output_path)
    print(f"Map saved to {output_path}")
    
    sim.close()
    return mapper


def random_exploration_example():
    """Example: Random exploration with mapping pipeline"""
    
    print("\n=== Random Exploration Example ===")
    
    # Note: This example assumes you have a proper Habitat-Lab config file
    # You'll need to create this config file or use an existing one
    config_path = "habitat_config.yaml"  # You need to create this
    
    try:
        # Create mapping pipeline
        mapper_config = {
            'vox_size': 0.15,  # Larger voxels for faster processing
            'max_pts_per_frame': 800,
            'max_empty_pts_per_frame': 800,
            'max_depth_sensing': 6.0,
        }
        
        pipeline = HabitatMappingPipeline(config_path, mapper_config)
        
        # Start mapping
        pipeline.start_mapping()
        
        # Random exploration
        actions = [0, 1, 2]  # Assuming these are forward, turn_left, turn_right
        num_steps = 50
        
        print(f"Performing {num_steps} random exploration steps...")
        
        for step in range(num_steps):
            # Choose random action
            action = np.random.choice(actions)
            
            # Execute step
            step_info = pipeline.step_mapping(action)
            
            # Print periodic updates
            if step % 10 == 0:
                mapping_info = step_info['mapping']
                print(f"Step {step}: {mapping_info['total_voxels']} voxels")
        
        # Save final map
        pipeline.save_map("random_exploration_map.pt")
        
        # Print final statistics
        stats = pipeline.get_stats()
        print("\n=== Final Results ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        pipeline.stop_mapping()
        
    except Exception as e:
        print(f"Pipeline example failed: {e}")
        print("This likely means you need to set up proper Habitat-Lab config files")


def visualization_example():
    """Example: Visualizing the occupancy map"""
    
    print("\n=== Visualization Example ===")
    
    # Load a previously saved map
    map_files = ["simple_navigation_map.pt", "random_exploration_map.pt"]
    
    for map_file in map_files:
        if Path(map_file).exists():
            print(f"\nVisualizing {map_file}...")
            
            # Create mapper and load map
            mapper = HabitatOccVoxelMap()
            mapper.load_map(map_file)
            
            # Get map data
            vox_xyz, vox_occ = mapper.get_occupancy_map()
            
            if vox_xyz is not None:
                # Convert to numpy for visualization
                xyz = vox_xyz.cpu().numpy()
                probs = mapper.get_occupancy_probabilities().cpu().numpy().flatten()
                
                # Create 3D visualization
                fig = plt.figure(figsize=(12, 8))
                
                # Top-down view
                ax1 = fig.add_subplot(121)
                occupied_mask = probs > 0.5
                free_mask = probs < 0.5
                
                ax1.scatter(xyz[occupied_mask, 0], xyz[occupied_mask, 1], 
                           c='red', s=20, alpha=0.7, label='Occupied')
                ax1.scatter(xyz[free_mask, 0], xyz[free_mask, 1], 
                           c='blue', s=5, alpha=0.3, label='Free')
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_title(f'Top-down View: {map_file}')
                ax1.legend()
                ax1.grid(True)
                ax1.axis('equal')
                
                # Side view
                ax2 = fig.add_subplot(122)
                ax2.scatter(xyz[occupied_mask, 0], xyz[occupied_mask, 2], 
                           c='red', s=20, alpha=0.7, label='Occupied')
                ax2.scatter(xyz[free_mask, 0], xyz[free_mask, 2], 
                           c='blue', s=5, alpha=0.3, label='Free')
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Z (m)')
                ax2.set_title(f'Side View: {map_file}')
                ax2.legend()
                ax2.grid(True)
                
                plt.tight_layout()
                
                # Save visualization
                vis_file = map_file.replace('.pt', '_visualization.png')
                plt.savefig(vis_file, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {vis_file}")
                
                plt.show()
                
            else:
                print(f"No map data found in {map_file}")
        else:
            print(f"Map file {map_file} not found")


def performance_benchmark():
    """Benchmark the mapping performance"""
    
    print("\n=== Performance Benchmark ===")
    
    # Create test configuration
    habitat_config = create_habitat_config()
    sim = habitat_sim.Simulator(habitat_config)
    
    # Test different voxel sizes
    voxel_sizes = [0.05, 0.1, 0.2, 0.3]
    point_limits = [500, 1000, 2000]
    
    results = {}
    
    for vox_size in voxel_sizes:
        for max_pts in point_limits:
            print(f"\nTesting vox_size={vox_size}, max_pts={max_pts}")
            
            # Create mapper
            mapper = HabitatOccVoxelMap(
                vox_size=vox_size,
                max_pts_per_frame=max_pts,
                max_empty_pts_per_frame=max_pts
            )
            
            # Helper to get observations
            def get_test_obs():
                habitat_obs = sim.get_sensor_observations()
                rgb = torch.from_numpy(habitat_obs["color_sensor"]).float() / 255.0
                depth = torch.from_numpy(habitat_obs["depth_sensor"]).float()
                rgb = rgb.permute(2, 0, 1).unsqueeze(0)
                depth = depth.unsqueeze(0).unsqueeze(0)
                
                agent_state = sim.get_agent_state()
                pose_matrix = np.eye(4)
                pose_matrix[:3, :3] = habitat_sim.utils.common.quat_to_matrix(agent_state.rotation)
                pose_matrix[:3, 3] = agent_state.position
                pose_4x4 = torch.from_numpy(pose_matrix).float().unsqueeze(0)
                
                H, W = 480, 640
                focal_length = W / (2.0 * np.tan(np.radians(90.0) / 2.0))
                intrinsics = torch.tensor([
                    [focal_length, 0.0, W/2.0],
                    [0.0, focal_length, H/2.0],
                    [0.0, 0.0, 1.0]
                ]).float()
                
                return {
                    'rgb_img': rgb,
                    'depth_img': depth,
                    'pose_4x4': pose_4x4,
                    'intrinsics_3x3': intrinsics
                }
            
            # Benchmark processing time
            num_frames = 10
            times = []
            
            for i in range(num_frames):
                obs = get_test_obs()
                
                start_time = time.time()
                update_info = mapper.process_habitat_obs(obs)
                end_time = time.time()
                
                times.append(end_time - start_time)
                
                # Move forward for next frame
                sim.step("move_forward")
            
            avg_time = np.mean(times)
            fps = 1.0 / avg_time if avg_time > 0 else float('inf')
            
            final_stats = mapper.get_map_stats()
            
            results[(vox_size, max_pts)] = {
                'avg_time_ms': avg_time * 1000,
                'fps': fps,
                'total_voxels': final_stats['total_voxels']
            }
            
            print(f"  Avg time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}, Voxels: {final_stats['total_voxels']}")
    
    # Print summary
    print("\n=== Performance Summary ===")
    print("Vox Size | Max Pts | Avg Time (ms) | FPS   | Voxels")
    print("-" * 55)
    for (vox_size, max_pts), result in results.items():
        print(f"{vox_size:8.2f} | {max_pts:7d} | {result['avg_time_ms']:13.2f} | {result['fps']:5.1f} | {result['total_voxels']:6d}")
    
    sim.close()


def main():
    """Main function with command line interface"""
    
    parser = argparse.ArgumentParser(description="Habitat OccVoxelMap Integration Examples")
    parser.add_argument("--example", type=str, default="simple",
                       choices=["simple", "random", "visualization", "benchmark", "all"],
                       help="Which example to run")
    parser.add_argument("--device", type=str, default="auto",
                       help="PyTorch device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set default tensor type
    if device == "cuda":
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    try:
        if args.example == "simple" or args.example == "all":
            simple_navigation_example()
            
        if args.example == "random" or args.example == "all":
            random_exploration_example()
            
        if args.example == "visualization" or args.example == "all":
            visualization_example()
            
        if args.example == "benchmark" or args.example == "all":
            performance_benchmark()
            
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()