#!/usr/bin/env python3
"""
Test script to validate Graph-R1 embedding fixes
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_embedding_fixes():
    """Test the visual embedding shape fixes"""
    try:
        from src.hypergraph_constructor import HypergraphBuilder
        
        logger.info("🧪 Testing Graph-R1 embedding fixes...")
        
        # Configuration
        config = {
            'unified_dimension': 512,
            'cross_modal_similarity_threshold': 0.4,
            'semantic_similarity_threshold': 0.7
        }
        
        # Test documents
        test_docs = {
            'text_documents': [
                'data/documents/Test Confluence and Jira Tickets.docx'
            ],
            'visual_documents': [
                'data/documents/RAG_ColPali_Visual_Diagram_Only.pdf'
            ],
            'salesforce_queries': [
                'artificial intelligence',
                'machine learning'
            ]
        }
        
        # Create hypergraph builder
        logger.info("🏗️ Creating hypergraph builder...")
        builder = HypergraphBuilder(config)
        
        # Build hypergraph
        logger.info("🔨 Building hypergraph with embedding fixes...")
        result = constructor.build_hypergraph(test_docs)
        
        # Validate results
        logger.info("✅ Hypergraph construction completed!")
        logger.info(f"📊 Nodes created: {result['nodes_created']}")
        logger.info(f"🔗 Edges created: {result['edges_created']}")
        logger.info(f"⏱️ Build time: {result['build_time']:.2f}s")
        
        # Check for shape mismatch errors
        shape_errors = [error for error in result.get('errors', []) if 'Shape mismatch' in error]
        
        if shape_errors:
            logger.error(f"❌ Found {len(shape_errors)} shape mismatch errors:")
            for error in shape_errors[:5]:  # Show first 5
                logger.error(f"  - {error}")
        else:
            logger.info("✅ No shape mismatch errors found!")
        
        # Check cross-modal connections
        if 'edges_by_type' in result['statistics']:
            cross_modal_edges = result['statistics']['edges_by_type'].get('cross_modal', 0)
            logger.info(f"🌉 Cross-modal edges: {cross_modal_edges}")
            
            if cross_modal_edges > 0:
                logger.info("✅ Cross-modal connections successfully established!")
            else:
                logger.warning("⚠️ No cross-modal connections created")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main test function"""
    logger.info("🚀 Starting Graph-R1 embedding fix validation...")
    
    # Check if required files exist
    required_files = [
        'data/documents/colpali_rag_test_table.pdf',
        'data/documents/RAG_ColPali_Visual_Diagram_Only.pdf'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logger.warning(f"⚠️ Missing test files: {missing_files}")
        logger.info("Continuing with available files...")
    
    # Run the test
    result = test_embedding_fixes()
    
    if result:
        logger.info("🎉 Embedding fix validation completed successfully!")
        return True
    else:
        logger.error("💥 Embedding fix validation failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)