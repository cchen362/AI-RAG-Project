#!/bin/bash
# NVIDIA Container Toolkit Installation Script for Pop!_OS 22.04
# Comprehensive setup for GPU Docker deployment on POP servers
# 
# Based on 2025 best practices and research findings
# CRITICAL: Uses NVIDIA official repo (not Pop!_OS packages which have bugs)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        error "Run as regular user - script will use sudo when needed"
        exit 1
    fi
}

# Verify system requirements
verify_system() {
    log "Verifying system requirements..."
    
    # Check if Pop!_OS
    if ! grep -q "Pop!_OS" /etc/os-release; then
        error "This script is designed for Pop!_OS 22.04"
        exit 1
    fi
    
    # Check Ubuntu version (Pop!_OS 22.04 is based on Ubuntu 22.04)
    UBUNTU_VERSION=$(lsb_release -rs)
    if [[ "$UBUNTU_VERSION" != "22.04" ]]; then
        warning "This script is optimized for Pop!_OS 22.04 (Ubuntu 22.04 base)"
        warning "Current version: $UBUNTU_VERSION - proceed with caution"
    fi
    
    # Check for NVIDIA GPU
    if ! lspci | grep -i nvidia > /dev/null; then
        error "No NVIDIA GPU detected!"
        error "Please ensure NVIDIA GPU is properly installed"
        exit 1
    fi
    
    # Check NVIDIA driver
    if ! nvidia-smi > /dev/null 2>&1; then
        error "NVIDIA driver not properly installed or not working"
        error "Please install NVIDIA drivers first: sudo apt install system76-driver-nvidia"
        exit 1
    fi
    
    success "System requirements verified"
    log "NVIDIA GPU Status:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
}

# Remove existing problematic packages
cleanup_existing() {
    log "Cleaning up existing nvidia-container packages..."
    
    # Remove Pop!_OS nvidia-container packages (known to have bugs)
    packages_to_remove=(
        "nvidia-container-toolkit"
        "nvidia-container-runtime"
        "nvidia-docker2"
        "libnvidia-container-tools"
        "libnvidia-container1"
    )
    
    for package in "${packages_to_remove[@]}"; do
        if dpkg -l | grep -q "^ii.*$package"; then
            log "Removing problematic package: $package"
            sudo apt-get remove -y "$package" || true
        fi
    done
    
    # Clean up any remaining configurations
    sudo apt-get autoremove -y
    sudo apt-get autoclean
    
    success "Cleanup completed"
}

# Install Docker if not present
setup_docker() {
    log "Setting up Docker..."
    
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        sudo apt-get update
        sudo apt-get install -y ca-certificates curl gnupg lsb-release
        
        # Add Docker's official GPG key
        sudo mkdir -p /etc/apt/keyrings
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
        
        # Add Docker repository
        echo \
            "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
            $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
        
        success "Docker installed"
    else
        success "Docker already installed"
    fi
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    log "User $USER added to docker group (requires logout/login to take effect)"
}

# Install NVIDIA Container Toolkit from official NVIDIA repository
install_nvidia_toolkit() {
    log "Installing NVIDIA Container Toolkit from official NVIDIA repository..."
    
    # Configure the production repository
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # CRITICAL: Pin NVIDIA repository to prevent Pop!_OS conflicts
    log "Configuring repository priorities to use NVIDIA packages..."
    cat << 'EOF' | sudo tee /etc/apt/preferences.d/nvidia-container-toolkit
Package: nvidia-container-toolkit nvidia-container-runtime libnvidia-container-tools libnvidia-container1
Pin: origin nvidia.github.io
Pin-Priority: 1001

Package: nvidia-container-toolkit nvidia-container-runtime libnvidia-container-tools libnvidia-container1
Pin: origin ppa.launchpadcontent.net
Pin-Priority: 100
EOF
    
    # Update package list and install
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    success "NVIDIA Container Toolkit installed from official repository"
}

# Configure Docker to use NVIDIA runtime
configure_docker_runtime() {
    log "Configuring Docker runtime for NVIDIA containers..."
    
    # Configure the container runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    
    # Restart Docker service
    sudo systemctl restart docker
    
    # Verify Docker daemon configuration
    if [[ -f /etc/docker/daemon.json ]]; then
        log "Docker daemon.json configuration:"
        cat /etc/docker/daemon.json
    fi
    
    success "Docker runtime configured"
}

# Test GPU container functionality
test_gpu_containers() {
    log "Testing GPU container functionality..."
    
    # Test basic CUDA container
    log "Testing basic CUDA access..."
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi; then
        success "Basic CUDA container test passed"
    else
        error "Basic CUDA container test failed"
        return 1
    fi
    
    # Test AI workload container
    log "Testing AI workload container..."
    if docker run --rm --gpus all pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"; then
        success "AI workload container test passed"
    else
        warning "AI workload container test failed - may need custom image"
    fi
    
    success "GPU container testing completed"
}

# Main execution
main() {
    log "🚀 Starting NVIDIA Container Toolkit setup for Pop!_OS 22.04"
    log "Optimized for AI-RAG-Project GPU-only deployment"
    echo
    
    check_root
    verify_system
    cleanup_existing
    setup_docker
    install_nvidia_toolkit
    configure_docker_runtime
    test_gpu_containers
    
    echo
    success "🎉 NVIDIA Container Toolkit setup completed successfully!"
    echo
    log "Next steps:"
    echo "1. Log out and log back in (or run 'newgrp docker') to activate docker group membership"
    echo "2. Run 'docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi' to verify setup"
    echo "3. Navigate to your AI-RAG-Project directory"
    echo "4. Run 'docker-compose --profile gpu up -d ai-rag-app-gpu' to deploy"
    echo
    log "For troubleshooting, check the logs at /var/log/nvidia-container-toolkit/"
    
    warning "IMPORTANT: You must log out and log back in before using Docker without sudo!"
}

# Run main function
main "$@"