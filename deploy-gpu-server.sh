#!/bin/bash

# =======================================================
# AI-RAG-Project GPU Server Deployment Script
# =======================================================
# Automated deployment script for GPU-enabled Linux servers
# Supports Ubuntu 20.04+, Debian 11+, CentOS 8+, RHEL 8+

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-rag-project"
DOCKER_COMPOSE_VERSION="2.21.0"
MIN_GPU_MEMORY=8  # GB
REQUIRED_DOCKER_VERSION="20.10.0"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Please run as a regular user with sudo privileges."
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
        log "Detected OS: $OS $VER"
    else
        error "Cannot detect OS. Please use Ubuntu 20.04+, Debian 11+, CentOS 8+, or RHEL 8+"
    fi
}

# Check CPU compatibility for AI libraries
check_cpu_compatibility() {
    log "Checking CPU compatibility for AI workloads..."
    
    # Get CPU information
    CPU_INFO=$(cat /proc/cpuinfo | grep "model name" | head -1 | cut -d: -f2 | xargs)
    log "CPU: $CPU_INFO"
    
    # Check for old problematic CPUs
    OLD_CPU_PATTERNS=("Phenom II" "Phenom" "Athlon II" "Athlon 64" "Core 2 Duo" "Core 2 Quad" "Pentium")
    CPU_COMPATIBLE=true
    
    for pattern in "${OLD_CPU_PATTERNS[@]}"; do
        if echo "$CPU_INFO" | grep -qi "$pattern"; then
            warn "OLD CPU DETECTED: $CPU_INFO"
            warn "This CPU may lack modern instruction sets (AVX, AVX2)"
            warn "AI libraries may fail with 'Illegal instruction' errors"
            warn "ENFORCING GPU-ONLY MODE for compatibility"
            CPU_COMPATIBLE=false
            export FORCE_GPU_ONLY=true
            break
        fi
    done
    
    # Check for AVX/AVX2 support
    if [ "$CPU_COMPATIBLE" = true ]; then
        if ! grep -q avx /proc/cpuinfo; then
            warn "CPU lacks AVX instruction set support"
            warn "Modern AI libraries may have compatibility issues"
            warn "ENFORCING GPU-ONLY MODE for safety"
            CPU_COMPATIBLE=false
            export FORCE_GPU_ONLY=true
        elif ! grep -q avx2 /proc/cpuinfo; then
            warn "CPU lacks AVX2 instruction set support"
            warn "Some AI libraries may have reduced performance"
            info "GPU-only mode recommended for optimal performance"
        else
            log "CPU supports modern instruction sets (AVX, AVX2) ✓"
        fi
    fi
    
    # Verify GPU availability if GPU-only mode is required
    if [ "$FORCE_GPU_ONLY" = true ]; then
        if ! command -v nvidia-smi &> /dev/null; then
            error "GPU-only mode required but NVIDIA GPU not available!"
        fi
        
        if ! nvidia-smi &> /dev/null; then
            error "GPU-only mode required but GPU not accessible!"
        fi
        
        log "GPU-only mode will be enforced due to CPU compatibility"
    fi
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check CPU compatibility first
    check_cpu_compatibility
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_MEM -lt 16 ]]; then
        warn "System has ${TOTAL_MEM}GB RAM. 16GB+ recommended for optimal performance."
    else
        log "System memory: ${TOTAL_MEM}GB ✓"
    fi
    
    # Check available disk space
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $AVAILABLE_SPACE -lt 50 ]]; then
        error "Insufficient disk space. Need at least 50GB available, found ${AVAILABLE_SPACE}GB"
    else
        log "Available disk space: ${AVAILABLE_SPACE}GB ✓"
    fi
    
    # Check for GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        GPU_MEMORY=$(echo "$GPU_INFO" | awk -F', ' '{print $2}' | head -1)
        GPU_MEMORY_GB=$((GPU_MEMORY / 1024))
        
        if [[ $GPU_MEMORY_GB -lt $MIN_GPU_MEMORY ]]; then
            warn "GPU has ${GPU_MEMORY_GB}GB memory. ${MIN_GPU_MEMORY}GB+ recommended for optimal performance."
        else
            log "GPU detected: $(echo "$GPU_INFO" | awk -F', ' '{print $1}' | head -1) (${GPU_MEMORY_GB}GB) ✓"
        fi
    else
        warn "NVIDIA GPU not detected or nvidia-smi not available. GPU features will be disabled."
    fi
}

# Install Docker
install_docker() {
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        log "Docker already installed: $DOCKER_VERSION"
        return
    fi
    
    log "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Update package index
    sudo apt-get update
    
    # Install prerequisites
    sudo apt-get install -y \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    log "Docker installed successfully"
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    if ! command -v nvidia-smi &> /dev/null; then
        warn "NVIDIA drivers not found. Skipping NVIDIA Container Toolkit installation."
        return
    fi
    
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log "NVIDIA Container Toolkit already configured ✓"
        return
    fi
    
    log "Installing NVIDIA Container Toolkit..."
    
    # Add NVIDIA package repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
        && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker daemon
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Test GPU access
    if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log "NVIDIA Container Toolkit installed and configured successfully ✓"
    else
        error "NVIDIA Container Toolkit installation failed"
    fi
}

# Install additional dependencies
install_dependencies() {
    log "Installing additional dependencies..."
    
    # System packages
    sudo apt-get update
    sudo apt-get install -y \
        git \
        curl \
        wget \
        unzip \
        htop \
        tmux \
        vim \
        poppler-utils \
        build-essential
    
    log "Dependencies installed successfully"
}

# Clone or update project
setup_project() {
    if [[ -d "$PROJECT_NAME" ]]; then
        log "Project directory exists. Updating..."
        cd "$PROJECT_NAME"
        git pull
    else
        log "Cloning project repository..."
        # Replace with your actual repository URL
        git clone https://github.com/your-username/ai-rag-project.git "$PROJECT_NAME"
        cd "$PROJECT_NAME"
    fi
    
    # Create necessary directories
    mkdir -p data cache logs models/cpu models/gpu
    
    # Set up environment file
    if [[ ! -f .env ]]; then
        log "Creating environment configuration..."
        cp .env.production .env
        
        info "Please edit .env file to configure your API keys and settings:"
        info "  - OPENAI_API_KEY (optional)"
        info "  - SALESFORCE_* credentials (optional)"
        info "  - MODEL_DEVICE (cuda for GPU, cpu for CPU-only)"
        
        read -p "Press Enter to continue after configuring .env file..."
    else
        log "Environment file already exists ✓"
    fi
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # If GPU-only mode is forced, only build GPU image
    if [ "$FORCE_GPU_ONLY" = true ]; then
        log "🎮 Building GPU-ONLY image for old CPU compatibility..."
        docker build --target gpu-production -t ai-rag-app:gpu .
        log "GPU-only Docker image built successfully ✓"
        
    # Check if GPU is available for GPU build (normal case)
    elif command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log "Building both CPU and GPU images..."
        docker build --target cpu-production -t ai-rag-app:cpu .
        docker build --target gpu-production -t ai-rag-app:gpu .
        log "Docker images built successfully ✓"
    else
        log "Building CPU image only..."
        docker build --target cpu-production -t ai-rag-app:cpu .
        log "CPU Docker image built successfully ✓"
    fi
}

# Deploy application
deploy_application() {
    log "Deploying application..."
    
    # Force GPU deployment if old CPU detected
    if [ "$FORCE_GPU_ONLY" = true ]; then
        log "🎮 ENFORCING GPU-ONLY DEPLOYMENT due to old CPU compatibility"
        log "Starting GPU-only deployment for AMD Phenom II X6 1090T compatibility..."
        
        # Set environment variables for GPU-only mode
        export MODEL_DEVICE=cuda
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
        
        docker-compose --profile gpu up -d ai-rag-app-gpu
        
        # Wait for health check
        log "Waiting for GPU-only application to start..."
        sleep 45  # Extra time for GPU model loading
        
        if curl -f http://localhost:8502/_stcore/health &> /dev/null; then
            log "🚀 GPU-ONLY deployment successful! Application available at: http://localhost:8502"
            log "✅ Old CPU compatibility ensured - all AI processing on RTX 3080"
        else
            warn "GPU-only deployment may have issues. Check logs: docker-compose logs ai-rag-app-gpu"
        fi
        
    # Check if GPU deployment is possible (normal case)
    elif command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        log "Starting GPU-accelerated deployment..."
        docker-compose --profile gpu up -d ai-rag-app-gpu
        
        # Wait for health check
        log "Waiting for application to start..."
        sleep 30
        
        if curl -f http://localhost:8502/_stcore/health &> /dev/null; then
            log "🚀 GPU deployment successful! Application available at: http://localhost:8502"
        else
            warn "GPU deployment may have issues. Check logs: docker-compose logs ai-rag-app-gpu"
        fi
    else
        log "Starting CPU deployment..."
        docker-compose up -d ai-rag-app
        
        # Wait for health check
        log "Waiting for application to start..."
        sleep 30
        
        if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
            log "🚀 CPU deployment successful! Application available at: http://localhost:8501"
        else
            warn "CPU deployment may have issues. Check logs: docker-compose logs ai-rag-app"
        fi
    fi
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    # Create monitoring script
    cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== AI-RAG-Project System Monitor ==="
echo "Date: $(date)"
echo ""

echo "=== Docker Containers ==="
docker-compose ps

echo ""
echo "=== System Resources ==="
echo "Memory Usage:"
free -h
echo ""
echo "Disk Usage:"
df -h . 

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi
fi

echo ""
echo "=== Application Logs (last 10 lines) ==="
docker-compose logs --tail=10 ai-rag-app 2>/dev/null || docker-compose logs --tail=10 ai-rag-app-gpu 2>/dev/null || echo "No logs available"
EOF
    
    chmod +x monitor.sh
    log "Monitoring script created: ./monitor.sh"
}

# Print deployment summary
print_summary() {
    echo ""
    log "==============================================="
    log "🎉 AI-RAG-Project Deployment Complete!"
    log "==============================================="
    echo ""
    
    if command -v nvidia-smi &> /dev/null && docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
        info "GPU-Accelerated Deployment:"
        info "  • Application URL: http://localhost:8502"
        info "  • Health Check: http://localhost:8502/_stcore/health"
        info "  • Container: ai-rag-gpu"
    else
        info "CPU-Optimized Deployment:"
        info "  • Application URL: http://localhost:8501"
        info "  • Health Check: http://localhost:8501/_stcore/health"
        info "  • Container: ai-rag-cpu"
    fi
    
    echo ""
    info "Management Commands:"
    info "  • View logs: docker-compose logs -f"
    info "  • Stop: docker-compose down"
    info "  • Restart: docker-compose restart"
    info "  • Monitor: ./monitor.sh"
    info "  • Update: git pull && docker-compose build && docker-compose up -d"
    
    echo ""
    info "Configuration Files:"
    info "  • Environment: .env"
    info "  • Docker: docker-compose.yml"
    info "  • Data: ./data/"
    info "  • Logs: ./logs/"
    
    echo ""
    log "Deployment completed successfully! 🚀"
}

# Main deployment function
main() {
    log "Starting AI-RAG-Project GPU Server Deployment"
    log "=============================================="
    
    check_root
    detect_os
    check_system_requirements
    install_docker
    install_nvidia_container_toolkit
    install_dependencies
    setup_project
    build_images
    deploy_application
    setup_monitoring
    print_summary
}

# Run main function
main "$@"