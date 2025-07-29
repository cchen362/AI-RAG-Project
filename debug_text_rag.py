"""
Debug TEXT RAG Answer Generation

This app tests whether LLM-based answer synthesis produces better TEXT RAG
responses compared to the current basic sentence extraction approach.

The goal is to prove that LLM synthesis works before modifying the main TEXT RAG system.
"""

import os
import time
import logging
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class TextRAGTester:
    """Test LLM-based answer synthesis vs current sentence extraction."""
    
    def __init__(self):
        self.openai_client = None
        self.model = "gpt-4o-mini"  # Use cheaper model for text synthesis
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Test queries that showed poor results in current system
        self.test_queries = [
            "Summarize the acceptance criteria of AI travel policy checker.",
            "What are the risks identified for AI travel policy checker?",
            "What are the main components of the travel booking system?",
            "How does the policy validation engine work?",
            "What happens when a booking violates company policy?"
        ]
        
        # Mock search results (simulating what vector search would return)
        self.mock_search_results = [
            {
                'content': 'The policy validation engine is designed to handle the following scenario: when a traveler is making a booking when they select travel options (flights, hotels, car rentals) then the ai system should validate in real-time against applicable corporate policies given that a booking violates company policy when the violation is detected then the system should display specific policy violation details and suggest compliant alternatives given that a policy has multiple conditions, and specifically when the ai analyzes the booking, the system will it should consider all policy hierarchies (global, regional, department-specific).',
                'score': 0.668,
                'metadata': {
                    'filename': 'travel_policy_requirements.txt',
                    'page': 1,
                    'chunk_id': 'chunk_001'
                }
            },
            {
                'content': 'Implement AI-Powered Travel Policy Compliance Checker Priority: High Components: AI/ML, Policy Engine, Booking Platform Related Tickets: FEAT-298 (Policy Database Optimization), FEAT-303 (ML Model Training Pipeline) Epic Link: EPIC-40 (Smart Travel Management). Risk Assessment: Data privacy concerns with traveler information, Integration complexity with existing booking systems, Performance requirements for real-time validation, Compliance with various regulatory frameworks across regions.',
                'score': 0.583,
                'metadata': {
                    'filename': 'project_specs.txt',
                    'page': 2,
                    'chunk_id': 'chunk_002'
                }
            },
            {
                'content': 'Travel Policy Acceptance Criteria: 1. System must validate bookings within 2 seconds. 2. Must support multi-tier policy hierarchy (global, regional, department). 3. Real-time violation detection with specific error messages. 4. Integration with existing booking platforms (Concur, SAP Concur, Expensify). 5. Audit trail for all policy decisions. 6. Support for policy exceptions and approval workflows. 7. Mobile compatibility for on-the-go bookings.',
                'score': 0.721,
                'metadata': {
                    'filename': 'acceptance_criteria.txt',
                    'page': 1,
                    'chunk_id': 'chunk_003'
                }
            }
        ]
        
        logger.info("✅ TextRAGTester initialized")
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI client initialized")
                return True
            else:
                logger.error("❌ OpenAI API key not found")
                return False
        except ImportError:
            logger.error("❌ OpenAI not available")
            return False
    
    def generate_current_approach_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        Simulate current TEXT RAG approach: basic sentence extraction.
        This replicates the logic from src/rag_system.py _generate_enhanced_answer.
        """
        logger.info(f"🔄 CURRENT APPROACH (sentence extraction):")
        logger.info(f"   Query: '{query}'")
        
        try:
            # Simulate current approach from rag_system.py
            top_results = search_results[:3]  # Focus on top 3
            
            # Extract best content (current approach)
            best_content = top_results[0]['content']
            
            # Basic sentence extraction (current method)
            sentences = [s.strip() for s in best_content.split('.') if s.strip()]
            
            # Find "relevant" sentences using keyword matching
            query_keywords = set(query.lower().split()) - {'what', 'are', 'the', 'a', 'an', 'how', 'does', 'when', 'where', 'why'}
            
            relevant_sentences = []
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                if query_keywords & sentence_words:  # If there's keyword overlap
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                answer = ". ".join(relevant_sentences[:2])  # Take top 2 relevant sentences
                if not answer.endswith('.'):
                    answer += '.'
            else:
                # Fallback to first few sentences
                answer = ". ".join(sentences[:2])
                if not answer.endswith('.'):
                    answer += '.'
            
            logger.info(f"   Generated answer: {len(answer)} chars")
            logger.info(f"   Preview: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Current approach failed: {e}")
            return f"Current approach failed: {e}"
    
    def generate_llm_synthesis_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        Test new LLM-based synthesis approach.
        Similar to ColPali's _analyze_image_with_vlm but for text chunks.
        """
        logger.info(f"🎯 LLM SYNTHESIS APPROACH (proposed):")
        logger.info(f"   Query: '{query}'")
        
        start_time = time.time()
        
        try:
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:3], 1):
                content = result['content']
                filename = result['metadata'].get('filename', 'document')
                score = result.get('score', 0)
                
                context_parts.append(f"Source {i} (from {filename}, relevance: {score:.3f}):\n{content}")
            
            combined_context = "\n\n".join(context_parts)
            
            # Create query-specific system prompt (similar to ColPali approach)
            system_prompt = f"""You are an expert document analyst helping answer specific questions about AI travel policy systems.

CRITICAL INSTRUCTIONS for query: "{query}"
1. Read all provided text sources carefully
2. Extract specific facts, details, and information that directly answer the question
3. Synthesize information from multiple sources when relevant
4. Focus ONLY on information relevant to this specific query
5. Provide a clear, direct answer - not a generic summary
6. If sources contain lists or criteria, extract them specifically
7. Quote exact information from sources when appropriate

Your response should:
- Start with the specific information that answers "{query}"
- Be concise but complete
- Use natural language, not fragment chunks
- Cite specific details from the provided sources"""

            # Call OpenAI for synthesis
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": f"Based on the following sources, please answer this question: {query}\n\nSOURCES:\n{combined_context}\n\nPlease provide a clear, specific answer based on the information in these sources."
                    }
                ],
                max_tokens=600,
                temperature=0.1  # Low temperature for factual, consistent responses
            )
            
            call_time = time.time() - start_time
            answer = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            logger.info(f"   API call time: {call_time:.2f}s")
            logger.info(f"   Tokens used: {tokens_used}")
            logger.info(f"   Generated answer: {len(answer)} chars")
            logger.info(f"   Preview: {answer[:100]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            return f"LLM synthesis failed: {e}"
    
    def compare_approaches(self, query: str, search_results: List[Dict]):
        """Compare current approach vs LLM synthesis for a specific query."""
        logger.info(f"\n📊 COMPARING APPROACHES for: '{query}'")
        logger.info("="*80)
        
        # Generate answer with current approach
        current_answer = self.generate_current_approach_answer(query, search_results)
        
        # Generate answer with LLM synthesis
        llm_answer = self.generate_llm_synthesis_answer(query, search_results)
        
        # Analysis and comparison
        logger.info(f"\n📋 RESPONSE COMPARISON:")
        logger.info("-" * 60)
        
        logger.info(f"🔄 CURRENT (sentence extraction):")
        logger.info(f"   Length: {len(current_answer)} chars")
        logger.info(f"   Response: {current_answer}")
        logger.info("")
        
        logger.info(f"🎯 LLM SYNTHESIS (proposed):")
        logger.info(f"   Length: {len(llm_answer)} chars") 
        logger.info(f"   Response: {llm_answer}")
        logger.info("")
        
        # Quality analysis
        self._analyze_response_quality(query, current_answer, llm_answer)
        
        return {
            'query': query,
            'current_answer': current_answer,
            'llm_answer': llm_answer,
            'current_length': len(current_answer),
            'llm_length': len(llm_answer)
        }
    
    def _analyze_response_quality(self, query: str, current_answer: str, llm_answer: str):
        """Analyze response quality and relevance."""
        logger.info(f"🔍 QUALITY ANALYSIS:")
        
        # Extract key terms from query
        query_lower = query.lower()
        key_terms = []
        
        if 'acceptance criteria' in query_lower or 'criteria' in query_lower:
            key_terms.extend(['criteria', 'requirements', 'must', 'should', 'system'])
        elif 'risks' in query_lower or 'risk' in query_lower:
            key_terms.extend(['risk', 'concern', 'issue', 'problem', 'challenge'])
        elif 'components' in query_lower or 'component' in query_lower:
            key_terms.extend(['component', 'system', 'platform', 'engine', 'module'])
        elif 'how does' in query_lower or 'work' in query_lower:
            key_terms.extend(['process', 'validate', 'system', 'engine', 'workflow'])
        elif 'what happens' in query_lower:
            key_terms.extend(['when', 'violation', 'detected', 'system', 'display'])
        
        # Count relevant terms in each response
        current_relevance = sum(1 for term in key_terms if term in current_answer.lower())
        llm_relevance = sum(1 for term in key_terms if term in llm_answer.lower())
        
        logger.info(f"   Key terms expected: {key_terms}")
        logger.info(f"   Current response relevance: {current_relevance}/{len(key_terms)}")
        logger.info(f"   LLM response relevance: {llm_relevance}/{len(key_terms)}")
        
        # Check for specific answer indicators
        current_specific = self._check_answer_specificity(current_answer, query)
        llm_specific = self._check_answer_specificity(llm_answer, query)
        
        logger.info(f"   Current answer specificity: {current_specific}")
        logger.info(f"   LLM answer specificity: {llm_specific}")
        
        # Overall assessment
        if llm_relevance > current_relevance and llm_specific > current_specific:
            logger.info("   ✅ LLM synthesis shows clear improvement")
        elif llm_relevance > current_relevance or llm_specific > current_specific:
            logger.info("   ⚠️ LLM synthesis shows some improvement")
        else:
            logger.info("   ❌ LLM synthesis may not be better (needs investigation)")
    
    def _check_answer_specificity(self, answer: str, query: str) -> int:
        """Check how specific the answer is to the query."""
        specificity_score = 0
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Query-specific checks
        if 'acceptance criteria' in query_lower:
            if any(indicator in answer_lower for indicator in ['must', 'system must', 'criteria:', '1.', '2.', 'requirements']):
                specificity_score += 2
        
        if 'risks' in query_lower:
            if any(indicator in answer_lower for indicator in ['risk', 'concern', 'privacy', 'security', 'complexity']):
                specificity_score += 2
        
        if 'components' in query_lower:
            if any(indicator in answer_lower for indicator in ['component', 'includes', 'consists of', 'ai/ml', 'engine']):
                specificity_score += 2
        
        if 'how does' in query_lower:
            if any(indicator in answer_lower for indicator in ['process', 'validates', 'when', 'then', 'workflow']):
                specificity_score += 2
        
        if 'what happens' in query_lower:
            if any(indicator in answer_lower for indicator in ['when', 'violation', 'system will', 'display', 'detected']):
                specificity_score += 2
        
        # General quality indicators
        if len(answer.split('.')) > 1:  # Multiple sentences
            specificity_score += 1
        
        if any(word in answer_lower for word in ['specifically', 'includes', 'such as', 'for example']):
            specificity_score += 1
        
        return specificity_score
    
    def run_comprehensive_test(self):
        """Run comprehensive test comparing approaches across all queries."""
        logger.info("🚀 Starting Comprehensive TEXT RAG Test")
        logger.info("="*60)
        
        if not self.openai_client:
            logger.error("❌ Cannot run test - OpenAI client not available")
            return
        
        results = []
        total_improvement_count = 0
        
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"\n📋 Test {i}/{len(self.test_queries)}")
            
            # Compare approaches for this query
            comparison = self.compare_approaches(query, self.mock_search_results)
            results.append(comparison)
            
            # Simple improvement check
            if len(comparison['llm_answer']) > len(comparison['current_answer']) * 1.2:
                total_improvement_count += 1
            
            logger.info("\n" + "="*80)
        
        # Final summary
        self._print_final_summary(results, total_improvement_count)
        
        return results
    
    def _print_final_summary(self, results: List[Dict], improvement_count: int):
        """Print comprehensive test summary."""
        logger.info(f"\n📊 FINAL TEST SUMMARY")
        logger.info("="*60)
        
        # Response length comparison
        avg_current_length = sum(r['current_length'] for r in results) / len(results)
        avg_llm_length = sum(r['llm_length'] for r in results) / len(results)
        
        logger.info(f"📝 RESPONSE QUALITY:")
        logger.info(f"   Average current response length: {avg_current_length:.0f} chars")
        logger.info(f"   Average LLM response length: {avg_llm_length:.0f} chars")
        logger.info(f"   Queries showing improvement: {improvement_count}/{len(results)}")
        logger.info("")
        
        # Recommendations
        logger.info(f"💡 RECOMMENDATIONS:")
        if improvement_count >= len(results) * 0.8:  # 80% improvement
            logger.info("   ✅ PROCEED with LLM synthesis approach")
            logger.info("   ✅ LLM responses show significant improvement over sentence extraction")
            logger.info("   ✅ Quality should match ColPali's VLM-generated responses")
        elif improvement_count >= len(results) * 0.6:  # 60% improvement
            logger.info("   ⚠️ PROCEED with caution - mixed results")
            logger.info("   ⚠️ Consider prompt engineering improvements")
        else:
            logger.info("   ❌ Current approach may be sufficient")
            logger.info("   ❌ LLM synthesis not showing clear benefits")
        
        logger.info("")
        logger.info("🎯 NEXT STEPS:")
        logger.info("   1. Review individual query results above")
        logger.info("   2. If LLM synthesis shows improvement, integrate into main TEXT RAG")
        logger.info("   3. Test with real document chunks and queries")
        logger.info("   4. Ensure integration doesn't break existing functionality")

def main():
    """Main test function."""
    tester = TextRAGTester()
    
    if not tester.openai_client:
        logger.error("❌ Cannot run test - OpenAI client not available")
        return
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    logger.info("\n🎯 TEST COMPLETE")
    logger.info("Review the results above to determine if LLM synthesis")
    logger.info("produces better TEXT RAG responses compared to sentence extraction.")

if __name__ == "__main__":
    main()