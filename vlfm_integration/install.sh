#!/bin/bash

# VLFM-Rayfronts Integration Installation Script
# This script sets up the complete environment for enhanced VLFM with rayfronts

set -e  # Exit on any error

echo "🚀 Starting VLFM-Rayfronts Integration Installation..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        exit 1
    fi
    print_success "Conda is available"
}

# Create conda environment
create_environment() {
    print_status "Creating conda environment 'enhanced_vlfm'..."
    
    if conda info --envs | grep -q enhanced_vlfm; then
        print_warning "Environment 'enhanced_vlfm' already exists. Removing it..."
        conda env remove -n enhanced_vlfm -y
    fi
    
    conda create -n enhanced_vlfm python=3.9 -y
    print_success "Conda environment created"
}

# Activate environment and install basic dependencies
install_basic_deps() {
    print_status "Installing basic dependencies..."
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate enhanced_vlfm
    
    # Install PyTorch
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
    pip install torch-scatter
    
    # Install other basic deps
    pip install numpy opencv-python matplotlib plotly
    pip install pyyaml tqdm
    
    print_success "Basic dependencies installed"
}

# Install Habitat
install_habitat() {
    print_status "Installing Habitat-Sim and Habitat-Lab..."
    
    # Install Habitat-Sim
    conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
    
    # Install Habitat-Lab
    pip install habitat-lab
    
    print_success "Habitat installed"
}

# Clone and install VLFM
install_vlfm() {
    print_status "Installing VLFM..."
    
    # Create workspace directory
    mkdir -p ~/enhanced_vlfm_workspace
    cd ~/enhanced_vlfm_workspace
    
    # Clone VLFM
    if [ -d "vlfm" ]; then
        print_warning "VLFM directory already exists. Removing it..."
        rm -rf vlfm
    fi
    
    git clone https://github.com/bdaiinstitute/vlfm.git
    cd vlfm
    
    # Install VLFM dependencies
    pip install -e .[habitat]
    pip install git+https://github.com/IDEA-Research/GroundingDINO.git@eeba084341aaa454ce13cb32fa7fd9282fc73a67
    pip install salesforce-lavis==1.0.2
    
    # Clone YOLOv7 (required by VLFM)
    git clone https://github.com/WongKinYiu/yolov7.git
    
    print_success "VLFM installed"
}

# Install our integration code
install_integration() {
    print_status "Installing rayfronts integration..."
    
    # Copy our integration files to VLFM directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
    
    cp "$SCRIPT_DIR/vlfm_rayfronts_adapter.py" vlfm/
    cp "$SCRIPT_DIR/vlfm_example.py" vlfm/
    mkdir -p vlfm/config
    cp "$SCRIPT_DIR/enhanced_vlfm_config.yaml" vlfm/config/
    
    print_success "Integration code installed"
}

# Download required models and data
download_models() {
    print_status "Downloading required models..."
    
    # Create data directory
    mkdir -p data/models
    
    # Download VLFM models (if available)
    # Note: You'll need to follow VLFM's instructions for downloading their models
    print_warning "Please follow VLFM's official instructions to download the required models:"
    print_warning "1. MobileSAM: https://github.com/ChaoningZhang/MobileSAM"
    print_warning "2. GroundingDINO: https://github.com/IDEA-Research/GroundingDINO"
    print_warning "3. YOLOv7: https://github.com/WongKinYiu/yolov7"
    print_warning "4. PointNav weights: included in VLFM data subdirectory"
    
    print_success "Model download instructions provided"
}

# Download sample data
download_sample_data() {
    print_status "Downloading sample data..."
    
    # Create data directories
    mkdir -p data/scene_datasets
    mkdir -p data/datasets/objectnav
    
    # Download habitat test scenes
    python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
    
    print_success "Sample data downloaded"
}

# Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test script for VLFM-Rayfronts integration"""

import sys
import torch

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import habitat
        import habitat_sim
        print("✓ Habitat imports successful")
    except ImportError as e:
        print(f"✗ Habitat import failed: {e}")
        return False
    
    try:
        import torch_scatter
        print("✓ torch_scatter import successful")
    except ImportError as e:
        print(f"✗ torch_scatter import failed: {e}")
        return False
    
    try:
        # Test our integration
        from vlfm_rayfronts_adapter import VLFMRayfrontsMapper, VLFMRayfrontsPolicy
        print("✓ Rayfronts integration imports successful")
    except ImportError as e:
        print(f"✗ Rayfronts integration import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from vlfm_rayfronts_adapter import VLFMRayfrontsMapper
        
        # Create mapper
        mapper = VLFMRayfrontsMapper(
            map_size_cm=1200,
            map_resolution=10,
            vox_size=0.2,
            device="cpu"  # Use CPU for testing
        )
        print("✓ Mapper creation successful")
        
        # Test basic functionality
        stats = mapper.get_exploration_policy()
        print("✓ Basic functionality test passed")
        
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    print("=== VLFM-Rayfronts Integration Test ===")
    
    # Test CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed!")
        sys.exit(1)
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Functionality tests failed!")
        sys.exit(1)
    
    print("\n🎉 All tests passed! Installation successful!")
    print("\nNext steps:")
    print("1. Download VLFM models following the official instructions")
    print("2. Download HM3D dataset if you want to run full evaluations")
    print("3. Run: python vlfm_example.py --config config/enhanced_vlfm_config.yaml")

if __name__ == "__main__":
    main()
EOF

    print_success "Test script created"
}

# Main installation function
main() {
    print_status "Starting VLFM-Rayfronts Integration Installation"
    
    # Check prerequisites
    check_conda
    
    # Installation steps
    create_environment
    install_basic_deps
    install_habitat
    install_vlfm
    install_integration
    download_models
    download_sample_data
    create_test_script
    
    print_success "Installation completed!"
    
    echo ""
    echo "🎉 VLFM-Rayfronts Integration installed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Activate the environment: conda activate enhanced_vlfm"
    echo "2. Go to workspace: cd ~/enhanced_vlfm_workspace/vlfm"
    echo "3. Run test: python test_installation.py"
    echo "4. Download required models (see README for instructions)"
    echo "5. Start exploring: python vlfm_example.py --help"
    echo ""
    echo "📖 For detailed usage instructions, see the README.md file"
    echo ""
    
    # Optional: run test
    read -p "🧪 Would you like to run the installation test now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Running installation test..."
        cd ~/enhanced_vlfm_workspace/vlfm
        eval "$(conda shell.bash hook)"
        conda activate enhanced_vlfm
        python test_installation.py
    fi
}

# Handle script interruption
trap 'print_error "Installation interrupted by user"; exit 1' INT

# Run main function
main "$@"