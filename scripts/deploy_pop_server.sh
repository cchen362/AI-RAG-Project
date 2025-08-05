#!/bin/bash
# Comprehensive AI-RAG-Project Deployment Script for POP Server
# GPU-Only Mode with 6-CPU Constraint Optimization
# 
# This script handles the complete deployment pipeline:
# 1. System verification and setup
# 2. NVIDIA Container Toolkit configuration  
# 3. Docker image building and deployment
# 4. Health monitoring and validation

set -e  # Exit on any error

# Configuration
PROJECT_NAME="ai-rag-project"
COMPOSE_SERVICE="ai-rag-app-gpu"
CONTAINER_NAME="ai-rag-gpu"
EXPOSE_PORT="8501"
NGINX_PORT="3015"  # For nginx reverse proxy

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
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

info() {
    echo -e "${PURPLE}ℹ️  $1${NC}"
}

# Display banner
show_banner() {
    echo -e "${PURPLE}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║              AI-RAG-Project POP Server Deployment            ║"
    echo "║                      GPU-Only Mode                           ║" 
    echo "║               6-CPU Optimized Configuration                   ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check if in project directory
    if [[ ! -f "docker-compose.yml" ]] || [[ ! -f "streamlit_rag_app.py" ]]; then
        error "Not in AI-RAG-Project directory or missing core files"
        error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        error "NVIDIA GPU not detected or drivers not installed"
        error "Please run setup_nvidia_toolkit_pop.sh first"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not installed"
        error "Please run setup_nvidia_toolkit_pop.sh first"
        exit 1
    fi
    
    # Check Docker group membership
    if ! groups | grep -q docker; then
        error "User not in docker group"
        error "Please run: sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi
    
    # Check nvidia-container-toolkit
    if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
        error "NVIDIA Container Toolkit not properly configured"
        error "Please run setup_nvidia_toolkit_pop.sh first"
        exit 1
    fi
    
    success "All prerequisites met"
}

# System resource verification
verify_system_resources() {
    log "Verifying system resources for 6-CPU constraint..."
    
    # CPU check
    CPU_COUNT=$(nproc)
    if [[ $CPU_COUNT -lt 6 ]]; then
        error "Insufficient CPU cores: $CPU_COUNT (minimum 6 required)"
        exit 1
    fi
    
    # Memory check (minimum 16GB for GPU-only processing)
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $MEMORY_GB -lt 16 ]]; then
        warning "Low system memory: ${MEMORY_GB}GB (recommended 16GB+)"
    fi
    
    # GPU memory check
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEMORY_GB=$((GPU_MEMORY / 1024))
    if [[ $GPU_MEMORY_GB -lt 8 ]]; then
        warning "Low GPU memory: ${GPU_MEMORY_GB}GB (recommended 8GB+)"
    fi
    
    success "System resources verified:"
    info "  CPUs: $CPU_COUNT"
    info "  System RAM: ${MEMORY_GB}GB"
    info "  GPU VRAM: ${GPU_MEMORY_GB}GB"
}

# Environment setup
setup_environment() {
    log "Setting up deployment environment..."
    
    # Create required directories
    mkdir -p data/documents cache/embeddings logs models/gpu
    
    # Set proper permissions
    chmod -R 755 data cache logs models
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        log "Creating default .env file..."
        cat > .env << 'EOF'
# AI-RAG-Project Configuration for POP Server
# GPU-Only Mode Environment Variables

# Core application settings
MODEL_DEVICE=cuda
FORCE_GPU_ONLY=true
DISABLE_CPU_FALLBACK=true

# API Keys (optional - local embeddings work without)
OPENAI_API_KEY=

# Salesforce integration (optional)
SALESFORCE_USERNAME=
SALESFORCE_PASSWORD=
SALESFORCE_SECURITY_TOKEN=

# Cross-platform poppler support
POPPLER_PATH=/usr/bin

# GPU optimization
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
EOF
        success "Default .env file created"
        warning "Please update .env with your API keys if needed"
    else
        success ".env file already exists"
    fi
    
    success "Environment setup completed"
}

# Stop existing containers
stop_existing_containers() {
    log "Stopping any existing containers..."
    
    # Stop Docker Compose services
    docker-compose down --remove-orphans || true
    
    # Stop individual containers if they exist
    if docker ps -q --filter "name=${CONTAINER_NAME}"; then
        docker stop ${CONTAINER_NAME} || true
        docker rm ${CONTAINER_NAME} || true
    fi
    
    # Clean up orphaned containers
    docker container prune -f || true
    
    success "Existing containers stopped and cleaned"
}

# Build Docker images
build_images() {
    log "Building GPU-optimized Docker images..."
    
    # Build GPU production image
    log "Building GPU production image (this may take 10-15 minutes)..."
    docker-compose build ai-rag-app-gpu
    
    # Verify image was built
    if ! docker images | grep -q "ai-rag-app.*gpu"; then
        error "Failed to build GPU image"
        exit 1
    fi
    
    success "Docker images built successfully"
    
    # Show image sizes
    log "Image information:"
    docker images | grep -E "(ai-rag-app|REPOSITORY)" | head -5
}

# Deploy application
deploy_application() {
    log "Deploying AI-RAG-Project in GPU-only mode..."
    
    # Deploy with GPU profile
    docker-compose --profile gpu up -d ai-rag-app-gpu
    
    # Wait for container to start
    log "Waiting for container to initialize..."
    sleep 10
    
    # Check if container is running
    if ! docker ps | grep -q ${CONTAINER_NAME}; then
        error "Container failed to start"
        log "Container logs:"
        docker-compose logs ai-rag-app-gpu
        exit 1
    fi
    
    success "Application deployed successfully"
}

# Health checks and validation
run_health_checks() {
    log "Running comprehensive health checks..."
    
    # Container health check
    log "Checking container health status..."
    HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' ${CONTAINER_NAME} 2>/dev/null || echo "no-health-check")
    
    if [[ "$HEALTH_STATUS" == "healthy" ]]; then
        success "Container health check: HEALTHY"
    elif [[ "$HEALTH_STATUS" == "starting" ]]; then
        warning "Container health check: STARTING (waiting...)"
        sleep 30
        HEALTH_STATUS=$(docker inspect --format='{{.State.Health.Status}}' ${CONTAINER_NAME} 2>/dev/null || echo "no-health-check")
        if [[ "$HEALTH_STATUS" == "healthy" ]]; then
            success "Container health check: HEALTHY"
        else
            warning "Container health check: $HEALTH_STATUS"
        fi
    else
        warning "Container health check: $HEALTH_STATUS"
    fi
    
    # GPU utilization check
    log "Checking GPU utilization in container..."
    if docker exec ${CONTAINER_NAME} nvidia-smi > /dev/null 2>&1; then
        success "GPU access verified in container"
        docker exec ${CONTAINER_NAME} nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
    else
        error "GPU access failed in container"
    fi
    
    # Application endpoint check
    log "Checking application endpoint..."
    for i in {1..10}; do
        if curl -s http://localhost:${EXPOSE_PORT} > /dev/null; then
            success "Application endpoint responding"
            break
        else
            if [[ $i -eq 10 ]]; then
                warning "Application endpoint not responding (may still be initializing)"
            else
                log "Waiting for application to initialize... (attempt $i/10)"
                sleep 15
            fi
        fi
    done
    
    success "Health checks completed"
}

# Display deployment information
show_deployment_info() {
    echo
    success "🎉 AI-RAG-Project successfully deployed on POP Server!"
    echo
    info "Deployment Information:"
    echo "  📝 Service Name: ${COMPOSE_SERVICE}"
    echo "  🐳 Container Name: ${CONTAINER_NAME}"
    echo "  🌐 Local Access: http://localhost:${EXPOSE_PORT}"
    echo "  🔧 GPU Mode: ENABLED (GPU-Only)"
    echo "  💾 CPU Limit: 6.0 CPUs (POP Server Optimized)"
    echo "  🧠 Memory Limit: 20GB"
    echo
    info "Management Commands:"
    echo "  View logs: docker-compose logs -f ai-rag-app-gpu"
    echo "  Stop service: docker-compose down"
    echo "  Restart service: docker-compose restart ai-rag-app-gpu"
    echo "  Container shell: docker exec -it ${CONTAINER_NAME} bash"
    echo
    info "For nginx reverse proxy configuration:"
    echo "  Proxy target: http://localhost:${EXPOSE_PORT}"
    echo "  Recommended external port: ${NGINX_PORT}"
    echo
    info "GPU Monitoring:"
    echo "  System: nvidia-smi -l 1"
    echo "  Container: docker exec ${CONTAINER_NAME} nvidia-smi"
    echo
    warning "Note: First-time model loading may take 2-3 minutes"
    warning "Monitor logs with: docker-compose logs -f ai-rag-app-gpu"
}

# Cleanup function for interruption
cleanup() {
    echo
    warning "Deployment interrupted. Cleaning up..."
    docker-compose down --remove-orphans || true
    exit 1
}

# Main execution
main() {
    trap cleanup INT TERM
    
    show_banner
    
    log "🚀 Starting AI-RAG-Project deployment for POP Server"
    log "Target configuration: GPU-Only mode with 6-CPU optimization"
    echo
    
    check_prerequisites
    verify_system_resources
    setup_environment
    stop_existing_containers
    build_images
    deploy_application
    run_health_checks
    show_deployment_info
    
    success "Deployment completed successfully! 🎉"
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "AI-RAG-Project POP Server Deployment Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Prerequisites:"
    echo "  1. Run setup_nvidia_toolkit_pop.sh first"
    echo "  2. Ensure you're in the project root directory"
    echo "  3. Have at least 6 CPU cores and 16GB RAM"
    echo
    echo "This script will:"
    echo "  - Verify system requirements and GPU access"
    echo "  - Build GPU-optimized Docker images"
    echo "  - Deploy in GPU-only mode with 6-CPU constraint"
    echo "  - Perform comprehensive health checks"
    echo "  - Configure proper resource limits"
    echo
    echo "Environment variables can be configured in .env file"
    exit 0
fi

# Run main function
main "$@"