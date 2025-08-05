"""
GPU Configuration and Memory Management for Old CPU Compatibility

This module provides centralized GPU configuration and memory management
for systems with old CPUs that lack modern instruction sets (AVX, AVX2).
"""

import os
import logging
import torch
import platform
import cpuinfo

logger = logging.getLogger(__name__)

class GPUConfigManager:
    """Centralized GPU configuration and memory management."""
    
    def __init__(self):
        self.gpu_only_mode = False
        self.gpu_available = False
        self.cpu_compatible = True
        self._configured = False
        
        # Perform compatibility check
        self._check_system_compatibility()
        
        # Configure GPU settings if needed
        if self.gpu_only_mode:
            self._configure_gpu_environment()
    
    def _check_system_compatibility(self):
        """Check CPU compatibility and determine if GPU-only mode is needed."""
        try:
            # Check environment override first
            if os.getenv('MODEL_DEVICE') == 'cuda':
                self.gpu_only_mode = True
                logger.info("🚀 GPU-only mode enforced by MODEL_DEVICE=cuda")
                return
            
            # Get CPU information
            cpu_info = cpuinfo.get_cpu_info()
            cpu_brand = cpu_info.get('brand_raw', '').lower()
            
            # Check for known problematic old CPUs
            old_cpu_patterns = [
                'phenom ii x6 1090t',  # User's specific CPU
                'phenom ii',
                'phenom',
                'athlon ii',
                'athlon 64',
                'core 2 duo',
                'core 2 quad',
                'pentium'
            ]
            
            for pattern in old_cpu_patterns:
                if pattern in cpu_brand:
                    self.cpu_compatible = False
                    self.gpu_only_mode = True
                    logger.warning(f"⚠️ OLD CPU DETECTED: {cpu_brand}")
                    logger.warning("   This CPU lacks modern instruction sets (AVX, AVX2)")
                    logger.warning("   Enforcing GPU-only mode to avoid 'Illegal instruction' errors")
                    return
            
            # Check CPU flags for modern instruction sets
            cpu_flags = cpu_info.get('flags', [])
            required_instructions = ['avx', 'avx2']
            missing_instructions = [inst for inst in required_instructions if inst not in cpu_flags]
            
            if missing_instructions:
                self.cpu_compatible = False
                self.gpu_only_mode = True
                logger.warning(f"⚠️ CPU missing modern instructions: {missing_instructions}")
                logger.warning("   Modern AI libraries may fail with 'Illegal instruction'")
                logger.warning("   Enforcing GPU-only mode for compatibility")
                return
            
            # CPU is compatible - check GPU availability
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"✅ Modern CPU with GPU available: {gpu_name}")
                logger.info("   GPU-only mode available but not required")
            else:
                logger.info("✅ Modern CPU detected, no GPU-only enforcement needed")
                
        except Exception as e:
            logger.warning(f"⚠️ CPU compatibility check failed: {e}")
            logger.warning("   Defaulting to GPU-only mode for safety")
            self.gpu_only_mode = True
    
    def _configure_gpu_environment(self):
        """Configure environment variables for GPU-only execution."""
        if self._configured:
            return
            
        logger.info("🔧 Configuring GPU-only execution environment...")
        
        # Verify GPU availability
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-only mode required but CUDA not available!")
        
        # Set environment variables for GPU-only execution
        gpu_env_vars = {
            'CUDA_VISIBLE_DEVICES': '0',  # Use first GPU only
            'MODEL_DEVICE': 'cuda',
            
            # PyTorch GPU memory optimization for old systems
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,garbage_collection_threshold:0.6',
            
            # Force GPU-only for AI libraries
            'SENTENCE_TRANSFORMERS_DEVICE': 'cuda',
            'TRANSFORMERS_DEVICE': 'cuda',
            
            # Disable CPU optimizations that might cause issues
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'NUMEXPR_NUM_THREADS': '1',
            
            # Disable CPU-specific optimizations
            'OPENBLAS_NUM_THREADS': '1',
            'VECLIB_MAXIMUM_THREADS': '1',
        }
        
        for env_var, value in gpu_env_vars.items():
            os.environ[env_var] = value
            logger.debug(f"   Set {env_var}={value}")
        
        # GPU memory status
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"🎮 Using GPU: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        
        # Perform initial GPU memory cleanup
        torch.cuda.empty_cache()
        current_memory = torch.cuda.memory_allocated(0) / 1024**3
        logger.info(f"🧹 Initial GPU memory: {current_memory:.2f}GB")
        
        self._configured = True
        logger.info("✅ GPU-only environment configured successfully")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory."""
        if self.gpu_available and torch.cuda.is_available():
            torch.cuda.empty_cache()
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"🧹 GPU memory after cleanup: {current_memory:.2f}GB")
    
    def get_device(self) -> str:
        """Get the appropriate device for model loading."""
        if self.gpu_only_mode:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU-only mode required but CUDA not available!")
            return 'cuda'
        else:
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_config_for_component(self, component_name: str) -> dict:
        """Get configuration for a specific component."""
        base_config = {
            'device': self.get_device(),
            'gpu_only_mode': self.gpu_only_mode,
            'gpu_available': self.gpu_available,
            'cpu_compatible': self.cpu_compatible
        }
        
        # Component-specific configurations
        if component_name == 'colpali':
            base_config.update({
                'torch_dtype': 'bfloat16' if self.gpu_only_mode else 'float32',
                'device_map': 'cuda' if self.gpu_only_mode else 'auto'
            })
        elif component_name == 'cross_encoder':
            base_config.update({
                'device': 'cuda' if self.gpu_only_mode else None
            })
        
        return base_config

# Global GPU configuration manager instance
_gpu_config = None

def get_gpu_config() -> GPUConfigManager:
    """Get the global GPU configuration manager instance."""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfigManager()
    return _gpu_config

def configure_gpu_for_old_cpu():
    """Configure GPU settings for old CPU compatibility."""
    return get_gpu_config()