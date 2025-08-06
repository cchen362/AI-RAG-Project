"""
Memory Optimization Utilities for ColPali RAG System

This module provides comprehensive memory monitoring, optimization, and quality assurance
for the ColPali visual document processing system on memory-constrained GPUs.
"""

import logging
import torch
import psutil
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import gc

logger = logging.getLogger(__name__)

class MemoryStrategy(Enum):
    """Memory optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"

@dataclass
class MemoryProfile:
    """Memory usage profile for optimization decisions."""
    total_gpu_memory: float
    available_gpu_memory: float
    used_gpu_memory: float
    cpu_memory_percent: float
    recommended_batch_size: int
    strategy: MemoryStrategy

@dataclass
class QualityMetrics:
    """Quality tracking metrics for adaptive processing."""
    processing_strategy: str
    visual_complexity_score: float
    query_intent_score: float
    success_rate: float
    processing_time: float
    memory_efficiency: float
    fallback_used: bool

class MemoryOptimizer:
    """
    Comprehensive memory optimization and quality assurance system.
    
    Provides:
    1. Real-time memory monitoring
    2. Adaptive strategy selection
    3. Quality tracking and reporting
    4. Performance optimization
    5. Graceful degradation
    """
    
    def __init__(self, gpu_limit_gb: float = 5.5):
        self.gpu_limit_gb = gpu_limit_gb
        self.gpu_warning_threshold = gpu_limit_gb * 0.8  # 80% warning
        self.gpu_critical_threshold = gpu_limit_gb * 0.9  # 90% critical
        
        # Quality tracking
        self.quality_metrics: List[QualityMetrics] = []
        self.success_count = 0
        self.failure_count = 0
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.memory_peaks: List[float] = []
        
        # Strategy tracking
        self.strategy_usage = {
            "visual_priority": 0,
            "hybrid": 0,
            "text_fallback": 0,
            "memory_aware_visual": 0
        }
        
        logger.info(f"🔧 MemoryOptimizer initialized (GPU limit: {gpu_limit_gb:.1f}GB)")
    
    def get_memory_profile(self) -> MemoryProfile:
        """Get current memory profile and recommendations."""
        try:
            # GPU memory analysis
            if torch.cuda.is_available():
                total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
                used_gpu = torch.cuda.memory_allocated(0) / 1024**3
                available_gpu = total_gpu - used_gpu
                
                # Ensure we don't exceed our limit
                available_gpu = min(available_gpu, self.gpu_limit_gb - used_gpu)
            else:
                total_gpu = 0
                used_gpu = 0
                available_gpu = 0
            
            # CPU memory analysis
            cpu_memory = psutil.virtual_memory()
            cpu_percent = cpu_memory.percent
            
            # Determine recommended batch size
            if available_gpu > 1.5:
                recommended_batch_size = 4
                strategy = MemoryStrategy.CONSERVATIVE
            elif available_gpu > 1.0:
                recommended_batch_size = 2
                strategy = MemoryStrategy.BALANCED
            elif available_gpu > 0.5:
                recommended_batch_size = 1
                strategy = MemoryStrategy.AGGRESSIVE
            else:
                recommended_batch_size = 1
                strategy = MemoryStrategy.AGGRESSIVE
                logger.warning(f"⚠️ Critical memory situation: {available_gpu:.2f}GB available")
            
            return MemoryProfile(
                total_gpu_memory=total_gpu,
                available_gpu_memory=available_gpu,
                used_gpu_memory=used_gpu,
                cpu_memory_percent=cpu_percent,
                recommended_batch_size=recommended_batch_size,
                strategy=strategy
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory profile: {e}")
            # Return safe defaults
            return MemoryProfile(
                total_gpu_memory=6.0,
                available_gpu_memory=1.0,
                used_gpu_memory=5.0,
                cpu_memory_percent=50.0,
                recommended_batch_size=1,
                strategy=MemoryStrategy.AGGRESSIVE
            )
    
    def optimize_memory_usage(self, force: bool = False) -> bool:
        """Perform memory optimization cleanup."""
        try:
            initial_memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            
            if force or initial_memory > self.gpu_warning_threshold:
                logger.info(f"🧹 Memory optimization triggered: {initial_memory:.2f}GB")
                
                # PyTorch GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                
                # Python garbage collection
                gc.collect()
                
                # Final memory check
                final_memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                freed = initial_memory - final_memory
                
                logger.info(f"   Freed: {freed:.2f}GB, Current: {final_memory:.2f}GB")
                
                return freed > 0.1  # Return True if significant memory was freed
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory optimization failed: {e}")
            return False
    
    def monitor_processing_quality(self, 
                                 strategy: str, 
                                 visual_score: float, 
                                 intent_score: float, 
                                 processing_time: float, 
                                 success: bool,
                                 fallback_used: bool = False) -> None:
        """Track processing quality metrics."""
        try:
            # Update counters
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Update strategy usage
            if strategy in self.strategy_usage:
                self.strategy_usage[strategy] += 1
            
            # Calculate current success rate
            total_attempts = self.success_count + self.failure_count
            success_rate = self.success_count / total_attempts if total_attempts > 0 else 0.0
            
            # Calculate memory efficiency (processing time vs memory usage)
            current_memory = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
            memory_efficiency = processing_time / max(current_memory, 0.1)  # Time per GB
            
            # Create quality metric
            metric = QualityMetrics(
                processing_strategy=strategy,
                visual_complexity_score=visual_score,
                query_intent_score=intent_score,
                success_rate=success_rate,
                processing_time=processing_time,
                memory_efficiency=memory_efficiency,
                fallback_used=fallback_used
            )
            
            self.quality_metrics.append(metric)
            
            # Keep only recent metrics (last 100)
            if len(self.quality_metrics) > 100:
                self.quality_metrics = self.quality_metrics[-100:]
            
            # Track performance trends
            self.processing_times.append(processing_time)
            self.memory_peaks.append(current_memory)
            
            # Keep only recent performance data
            if len(self.processing_times) > 50:
                self.processing_times = self.processing_times[-50:]
                self.memory_peaks = self.memory_peaks[-50:]
            
            logger.info(f"📊 Quality tracking: {strategy} | Success: {success} | Rate: {success_rate:.2f}")
            
        except Exception as e:
            logger.warning(f"⚠️ Quality monitoring failed: {e}")
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality and performance report."""
        try:
            total_attempts = self.success_count + self.failure_count
            success_rate = self.success_count / total_attempts if total_attempts > 0 else 0.0
            
            # Strategy effectiveness analysis
            strategy_effectiveness = {}
            for strategy, usage_count in self.strategy_usage.items():
                if usage_count > 0:
                    # Calculate success rate for this strategy
                    strategy_metrics = [m for m in self.quality_metrics if m.processing_strategy == strategy]
                    strategy_success_rate = sum(1 for m in strategy_metrics if m.success_rate > 0.8) / len(strategy_metrics) if strategy_metrics else 0
                    
                    strategy_effectiveness[strategy] = {
                        'usage_count': usage_count,
                        'success_rate': strategy_success_rate,
                        'avg_processing_time': sum(m.processing_time for m in strategy_metrics) / len(strategy_metrics) if strategy_metrics else 0
                    }
            
            # Performance trends
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            avg_memory_usage = sum(self.memory_peaks) / len(self.memory_peaks) if self.memory_peaks else 0
            
            # Memory profile
            memory_profile = self.get_memory_profile()
            
            report = {
                'overall_performance': {
                    'total_attempts': total_attempts,
                    'success_count': self.success_count,
                    'failure_count': self.failure_count,
                    'success_rate': success_rate,
                    'avg_processing_time': avg_processing_time,
                    'avg_memory_usage': avg_memory_usage
                },
                'strategy_effectiveness': strategy_effectiveness,
                'memory_status': {
                    'current_usage': memory_profile.used_gpu_memory,
                    'available': memory_profile.available_gpu_memory,
                    'utilization_rate': memory_profile.used_gpu_memory / memory_profile.total_gpu_memory if memory_profile.total_gpu_memory > 0 else 0,
                    'recommended_strategy': memory_profile.strategy.value
                },
                'quality_trends': {
                    'recent_success_rate': self.quality_metrics[-10:] if len(self.quality_metrics) >= 10 else [],
                    'fallback_usage_rate': sum(1 for m in self.quality_metrics if m.fallback_used) / len(self.quality_metrics) if self.quality_metrics else 0
                },
                'recommendations': self._generate_recommendations(memory_profile, strategy_effectiveness)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate quality report: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, memory_profile: MemoryProfile, strategy_effectiveness: Dict) -> List[str]:
        """Generate actionable recommendations based on performance analysis."""
        recommendations = []
        
        # Memory-based recommendations
        if memory_profile.available_gpu_memory < 0.5:
            recommendations.append("Consider using CPU fallback for large documents")
            recommendations.append("Enable aggressive memory cleanup between operations")
        
        if memory_profile.used_gpu_memory > self.gpu_critical_threshold:
            recommendations.append("Critical memory usage detected - reduce batch sizes")
        
        # Strategy-based recommendations
        best_strategy = None
        best_success_rate = 0
        
        for strategy, metrics in strategy_effectiveness.items():
            if metrics['success_rate'] > best_success_rate:
                best_success_rate = metrics['success_rate']
                best_strategy = strategy
        
        if best_strategy and best_success_rate > 0.8:
            recommendations.append(f"'{best_strategy}' strategy shows best results ({best_success_rate:.1%} success)")
        
        # Performance recommendations
        if self.processing_times and sum(self.processing_times[-5:]) / 5 > 30:  # If avg processing time > 30s
            recommendations.append("Consider using smaller batch sizes for faster response times")
        
        return recommendations
    
    def log_performance_summary(self) -> None:
        """Log a comprehensive performance summary."""
        report = self.get_quality_report()
        
        logger.info("="*50)
        logger.info("🎯 COLPALI MEMORY OPTIMIZATION SUMMARY")
        logger.info("="*50)
        
        overall = report.get('overall_performance', {})
        logger.info(f"📊 Processing Stats:")
        logger.info(f"   Total Attempts: {overall.get('total_attempts', 0)}")
        logger.info(f"   Success Rate: {overall.get('success_rate', 0):.1%}")
        logger.info(f"   Avg Processing Time: {overall.get('avg_processing_time', 0):.2f}s")
        
        memory = report.get('memory_status', {})
        logger.info(f"🧠 Memory Status:")
        logger.info(f"   Current Usage: {memory.get('current_usage', 0):.2f}GB")
        logger.info(f"   Available: {memory.get('available', 0):.2f}GB")
        logger.info(f"   Utilization: {memory.get('utilization_rate', 0):.1%}")
        
        strategies = report.get('strategy_effectiveness', {})
        if strategies:
            logger.info(f"🎯 Strategy Effectiveness:")
            for strategy, metrics in strategies.items():
                logger.info(f"   {strategy}: {metrics['success_rate']:.1%} success, {metrics['usage_count']} uses")
        
        recommendations = report.get('recommendations', [])
        if recommendations:
            logger.info(f"💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        
        logger.info("="*50)

# Global memory optimizer instance
memory_optimizer = MemoryOptimizer()