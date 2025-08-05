# POP Server Deployment Troubleshooting Guide
**AI-RAG-Project GPU-Only Mode for Old CPU Compatibility**

This comprehensive troubleshooting guide addresses common issues when deploying AI-RAG-Project on Pop!_OS servers with old CPUs and 6-CPU constraints.

---

## 🚨 **CRITICAL ISSUES & SOLUTIONS**

### **Issue 1: Container Creation Fails - CPU Limit Error**
```
ERROR: range of CPUs is from 0.01 to 6.00, as there are only 6 CPUs available
```

**Root Cause**: Docker Compose configured for 8.0 CPUs but server only has 6 CPUs.

**Solution**:
```bash
# Fixed in docker-compose.yml:
# GPU service CPU limit: 8.0 → 6.0
# GPU service CPU reservation: 4.0 → 3.0
docker-compose --profile gpu up -d ai-rag-app-gpu
```

**Prevention**: Always check CPU count with `nproc` before deployment.

---

### **Issue 2: nvidia-container-toolkit Installation Fails**
```
nvidia-container-toolkit from pop_os distribution does not work
```

**Root Cause**: Pop!_OS packages have known bugs - must use NVIDIA's official repository.

**Solution**:
```bash
# Run the provided setup script
chmod +x scripts/setup_nvidia_toolkit_pop.sh
./scripts/setup_nvidia_toolkit_pop.sh

# Manual fix if needed:
sudo apt-get remove nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Test**: `docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi`

---

### **Issue 3: Old CPU "Illegal Instruction" Errors**
```
Illegal instruction (core dumped)
RuntimeError: AVX instruction set required
```

**Root Cause**: AMD Phenom II X6 lacks modern instruction sets (AVX, AVX2).

**Solution**: GPU-only mode is already configured:
```bash
# Verify GPU-only environment variables are set:
MODEL_DEVICE=cuda
FORCE_GPU_ONLY=true
DISABLE_CPU_FALLBACK=true
SENTENCE_TRANSFORMERS_DEVICE=cuda
TRANSFORMERS_DEVICE=cuda
```

**Verification**: Check with health check script:
```bash
docker exec ai-rag-gpu python scripts/health_check.py
```

---

## 🔧 **COMMON DEPLOYMENT ISSUES**

### **Docker Build Issues**

#### **Long Build Times (10+ minutes)**
- **Expected**: Model pre-loading takes time
- **Monitor**: Use `docker-compose build --progress=plain` for verbose output
- **Optimization**: Models are cached in container layers

#### **Build Fails - Out of Space**
```bash
# Clean up Docker
docker system prune -af
docker volume prune -f

# Check disk space
df -h
```

#### **Build Fails - Memory Issues**
```bash
# Reduce parallel build processes
export DOCKER_BUILDKIT=0
docker-compose build --no-cache ai-rag-app-gpu
```

### **Container Runtime Issues**

#### **Container Starts but Health Check Fails**
```bash
# Check container logs
docker-compose logs -f ai-rag-app-gpu

# Check health check output
docker exec ai-rag-gpu python scripts/health_check.py

# Common issues:
# 1. GPU not accessible → verify nvidia-container-toolkit
# 2. Models not loaded → check model cache
# 3. Port conflicts → verify port 8501 is free
```

#### **GPU Not Accessible in Container**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi

# If fails, check:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify Docker daemon.json
cat /etc/docker/daemon.json
```

#### **Application Won't Start - Port Conflicts**
```bash
# Check what's using port 8501
sudo netstat -tulpn | grep :8501
sudo lsof -i :8501

# Kill conflicting processes
sudo pkill -f streamlit
```

### **Performance Issues**

#### **Slow Model Loading (5+ minutes)**
- **Check**: GPU utilization with `nvidia-smi -l 1`
- **Verify**: GPU-only mode is enforced
- **Monitor**: Container logs during startup

#### **Out of Memory Errors**
```bash
# Check GPU memory
nvidia-smi

# Reduce batch sizes in environment
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8

# Check system memory
free -h
```

---

## 🔍 **DIAGNOSTIC COMMANDS**

### **System Verification**
```bash
# Check CPU info
cat /proc/cpuinfo | grep "model name" | head -1
nproc

# Check memory
free -h

# Check GPU
nvidia-smi
lspci | grep -i nvidia

# Check OS version
cat /etc/os-release
```

### **Docker & Container Diagnostics**
```bash
# Docker status
docker --version
docker compose version
systemctl status docker

# Container status
docker ps -a | grep ai-rag
docker stats ai-rag-gpu

# Network inspection
docker network ls
docker network inspect bridge | grep Gateway

# Image inspection
docker images | grep ai-rag
docker history ai-rag-app:gpu
```

### **Application Diagnostics**
```bash
# Health check
docker exec ai-rag-gpu python scripts/health_check.py

# Application logs
docker-compose logs -f ai-rag-app-gpu

# Resource usage inside container
docker exec ai-rag-gpu ps aux
docker exec ai-rag-gpu free -h
docker exec ai-rag-gpu nvidia-smi
```

---

## 🚀 **DEPLOYMENT WORKFLOW**

### **Step 1: Pre-deployment Checks**
```bash
# Verify system meets requirements
nproc  # Should show 6+
free -h  # Should show 16GB+
nvidia-smi  # Should show GPU
```

### **Step 2: Environment Setup**
```bash
# Run NVIDIA toolkit setup
./scripts/setup_nvidia_toolkit_pop.sh

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### **Step 3: Application Deployment**
```bash
# Run comprehensive deployment
./scripts/deploy_pop_server.sh

# Monitor deployment
docker-compose logs -f ai-rag-app-gpu
```

### **Step 4: Post-deployment Validation**
```bash
# Run health checks
docker exec ai-rag-gpu python scripts/health_check.py

# Test application access
curl http://localhost:8501/_stcore/health

# Monitor performance
nvidia-smi -l 5
```

---

## 📊 **PERFORMANCE BENCHMARKS**

### **Expected Performance (POP Server 6-CPU + GPU)**
- **Container Startup**: 10-30 seconds
- **Model Loading**: 1-3 minutes (first time)
- **Health Check**: All 8 checks should pass
- **Memory Usage**: 
  - System: 8-12GB
  - GPU: 4-8GB VRAM
- **CPU Usage**: 20-40% during inference

### **Performance Red Flags**
- Startup > 5 minutes → Check model loading
- GPU memory > 90% → Reduce batch sizes
- CPU usage > 80% → Verify GPU-only mode
- Health checks failing → Check configuration

---

## 🛠️ **ADVANCED TROUBLESHOOTING**

### **Deep Container Inspection**
```bash
# Enter container for debugging
docker exec -it ai-rag-gpu bash

# Check Python environment
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import sentence_transformers; print('Sentence transformers OK')"

# Check model files
ls -la /app/models/
ls -la /app/models/transformers/
ls -la /app/models/huggingface/

# Check environment variables
env | grep -E "(CUDA|MODEL|GPU|TRANSFORM)"
```

### **Network & Proxy Issues**
```bash
# For nginx reverse proxy
sudo nginx -t
sudo systemctl status nginx

# Find Docker gateway IP
docker network inspect bridge | grep Gateway

# Test internal connectivity
curl http://172.19.0.1:8501/_stcore/health
```

### **Model Loading Issues**
```bash
# Check model manifest
docker exec ai-rag-gpu cat /app/models/model_manifest.json

# Re-run model warm-up
docker exec ai-rag-gpu python scripts/warm_up_models.py

# Check disk space for models
docker exec ai-rag-gpu df -h /app/models/
```

---

## 📞 **ESCALATION PROCEDURES**

### **When to Escalate**
1. Health checks consistently failing after following all solutions
2. GPU not accessible despite proper nvidia-container-toolkit setup
3. Application crashes with "Illegal instruction" in GPU-only mode
4. Performance significantly below benchmarks

### **Information to Collect**
```bash
# System info
uname -a
cat /etc/os-release
nvidia-smi
docker --version

# Container logs
docker-compose logs ai-rag-app-gpu > deployment-logs.txt

# Health check output
docker exec ai-rag-gpu python scripts/health_check.py > health-check.txt

# Resource usage
nvidia-smi > gpu-status.txt
free -h > memory-status.txt
docker stats --no-stream > container-stats.txt
```

---

## 🔄 **RECOVERY PROCEDURES**

### **Complete Reset**
```bash
# Stop all containers
docker-compose down --remove-orphans

# Clean up Docker
docker system prune -af
docker volume prune -f

# Remove images
docker rmi ai-rag-app:gpu

# Re-run deployment
./scripts/deploy_pop_server.sh
```

### **Partial Reset - Keep Models**
```bash
# Stop container
docker-compose stop ai-rag-app-gpu

# Remove container but keep images
docker-compose rm -f ai-rag-app-gpu

# Restart
docker-compose --profile gpu up -d ai-rag-app-gpu
```

---

**Last Updated**: August 2025  
**Tested On**: Pop!_OS 22.04 with AMD Phenom II X6 + NVIDIA GPU  
**Version**: AI-RAG-Project v2.0 GPU-Only Mode