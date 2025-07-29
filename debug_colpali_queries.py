"""
Debug ColPali Query-Specific Analysis

This app tests whether query-specific VLM analysis produces different,
relevant responses compared to the current generic precomputed approach.

The goal is to prove that live query-specific analysis works before
modifying the main application.
"""

import os
import time
import logging
import base64
from io import BytesIO
from PIL import Image
from pathlib import Path
import openai
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class QuerySpecificColPaliTester:
    """Test query-specific VLM analysis vs generic precomputed responses."""
    
    def __init__(self):
        self.vlm_client = None
        self.vlm_model = "gpt-4o"
        self.test_document = "data/documents/RAG_ColPali_Visual_Test.pdf"
        
        # Initialize VLM client
        self._init_vlm_client()
        
        # Test queries that should produce different responses
        self.test_queries = [
            "What's the retrieval time in a ColPali RAG pipeline?",
            "What are the different stages in the pipeline?",
            "Which stage takes the longest time?", 
            "What is the overall performance comparison?",
            "How does embedding generation compare to query processing?"
        ]
        
        logger.info("✅ QuerySpecificColPaliTester initialized")
    
    def _init_vlm_client(self):
        """Initialize OpenAI Vision client."""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.vlm_client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI GPT-4 Vision initialized")
                return True
            else:
                logger.error("❌ OpenAI API key not found")
                return False
        except ImportError:
            logger.error("❌ OpenAI not available")
            return False
    
    def load_test_document(self):
        """Load and convert test PDF to images."""
        try:
            if not os.path.exists(self.test_document):
                logger.error(f"❌ Test document not found: {self.test_document}")
                return None
                
            logger.info(f"📄 Loading test document: {self.test_document}")
            
            # Convert PDF to images
            images = convert_from_path(self.test_document, dpi=200)
            
            if images:
                logger.info(f"✅ Converted {len(images)} pages to images")
                return images
            else:
                logger.error("❌ No images extracted from PDF")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to load test document: {e}")
            return None
    
    def analyze_with_generic_query(self, image, page_number):
        """Simulate current generic precomputed approach."""
        generic_query = "What is this document about?"
        
        logger.info(f"🔍 GENERIC ANALYSIS (current approach):")
        logger.info(f"   Query: '{generic_query}'")
        
        try:
            response = self._call_vlm_api(generic_query, image, page_number, is_generic=True)
            return response
        except Exception as e:
            logger.error(f"❌ Generic analysis failed: {e}")
            return f"Generic analysis failed: {e}"
    
    def analyze_with_specific_query(self, query, image, page_number):
        """Test new query-specific approach."""
        logger.info(f"🎯 QUERY-SPECIFIC ANALYSIS (proposed approach):")
        logger.info(f"   Query: '{query}'")
        
        try:
            response = self._call_vlm_api(query, image, page_number, is_generic=False)
            return response
        except Exception as e:
            logger.error(f"❌ Query-specific analysis failed: {e}")
            return f"Query-specific analysis failed: {e}"
    
    def _call_vlm_api(self, query, image, page_number, is_generic=False):
        """Call OpenAI Vision API with given query."""
        start_time = time.time()
        
        try:
            # Convert PIL image to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create query-specific system prompt
            if is_generic:
                system_prompt = f"""You are analyzing page {page_number} of a document. 
                
Provide a general summary of what this document page contains."""
            else:
                system_prompt = f"""You are an expert document analyst examining page {page_number}.

CRITICAL INSTRUCTIONS for query: "{query}"
1. Look carefully at the image and read ALL visible text
2. Extract specific facts, numbers, dates, and details that answer the exact question
3. If you see tables or charts, transcribe relevant data points
4. If you see performance metrics, report the actual numbers
5. Quote exact text from the document when possible
6. Focus ONLY on information relevant to this specific query
7. DO NOT give generic summaries - answer the specific question

Your response should:
- Start with the specific information that answers "{query}"
- Include exact quotes, numbers, or data from the page
- Describe visual elements (charts, tables, diagrams) with specific details relevant to the query
- Be concrete and factual, not vague or general"""
            
            # Call OpenAI Vision API
            response = self.vlm_client.chat.completions.create(
                model=self.vlm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please analyze this document page to answer: {query}\n\nI need specific facts and details from what you can see in the image, not general descriptions."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=800,
                temperature=0.0  # Deterministic responses
            )
            
            call_time = time.time() - start_time
            answer = response.choices[0].message.content
            
            # Log performance metrics
            tokens_used = response.usage.total_tokens if response.usage else 0
            logger.info(f"   API call time: {call_time:.2f}s")
            logger.info(f"   Tokens used: {tokens_used}")
            logger.info(f"   Response length: {len(answer)} chars")
            
            return {
                'answer': answer,
                'call_time': call_time,
                'tokens_used': tokens_used,
                'query_type': 'generic' if is_generic else 'specific'
            }
            
        except Exception as e:
            logger.error(f"❌ VLM API call failed: {e}")
            return {
                'answer': f"API call failed: {e}",
                'call_time': time.time() - start_time,
                'tokens_used': 0,
                'query_type': 'error'
            }
    
    def compare_responses(self, generic_response, specific_response, query):
        """Compare generic vs specific responses."""
        logger.info(f"\n📊 RESPONSE COMPARISON for: '{query}'")
        logger.info("="*80)
        
        # Generic response
        generic_answer = generic_response.get('answer', 'No response')
        logger.info(f"🔄 GENERIC RESPONSE (current system):")
        logger.info(f"   Length: {len(generic_answer)} chars")
        logger.info(f"   Time: {generic_response.get('call_time', 0):.2f}s")
        logger.info(f"   Tokens: {generic_response.get('tokens_used', 0)}")
        logger.info(f"   Content: {generic_answer[:200]}{'...' if len(generic_answer) > 200 else ''}")
        logger.info("")
        
        # Specific response  
        specific_answer = specific_response.get('answer', 'No response')
        logger.info(f"🎯 QUERY-SPECIFIC RESPONSE (proposed system):")
        logger.info(f"   Length: {len(specific_answer)} chars")
        logger.info(f"   Time: {specific_response.get('call_time', 0):.2f}s")
        logger.info(f"   Tokens: {specific_response.get('tokens_used', 0)}")
        logger.info(f"   Content: {specific_answer[:200]}{'...' if len(specific_answer) > 200 else ''}")
        logger.info("")
        
        # Analysis
        self._analyze_response_quality(generic_answer, specific_answer, query)
        
        return {
            'query': query,
            'generic': generic_response,
            'specific': specific_response,
            'generic_length': len(generic_answer),
            'specific_length': len(specific_answer)
        }
    
    def _analyze_response_quality(self, generic_answer, specific_answer, query):
        """Analyze response quality and relevance."""
        logger.info(f"🔍 QUALITY ANALYSIS:")
        
        # Check for query-specific terms
        query_lower = query.lower()
        generic_lower = generic_answer.lower()
        specific_lower = specific_answer.lower()
        
        # Extract key terms from query
        key_terms = []
        if 'retrieval time' in query_lower:
            key_terms.extend(['time', 'retrieval', 'seconds', 'ms', 'performance'])
        elif 'stages' in query_lower:
            key_terms.extend(['stage', 'step', 'pipeline', 'process'])
        elif 'longest' in query_lower:
            key_terms.extend(['longest', 'slowest', 'maximum', 'highest'])
        elif 'comparison' in query_lower:
            key_terms.extend(['compare', 'versus', 'vs', 'difference'])
        elif 'embedding generation' in query_lower:
            key_terms.extend(['embedding', 'generation', 'encode', 'vector'])
        
        # Count relevant terms in each response
        generic_relevance = sum(1 for term in key_terms if term in generic_lower)
        specific_relevance = sum(1 for term in key_terms if term in specific_lower)
        
        logger.info(f"   Key terms in query: {key_terms}")
        logger.info(f"   Generic response relevance: {generic_relevance}/{len(key_terms)}")
        logger.info(f"   Specific response relevance: {specific_relevance}/{len(key_terms)}")
        
        # Check for numbers (indicates specific data)
        import re
        generic_numbers = len(re.findall(r'\d+\.?\d*', generic_answer))
        specific_numbers = len(re.findall(r'\d+\.?\d*', specific_answer))
        
        logger.info(f"   Numbers in generic response: {generic_numbers}")
        logger.info(f"   Numbers in specific response: {specific_numbers}")
        
        # Overall assessment
        if specific_relevance > generic_relevance:
            logger.info("   ✅ Specific response is more relevant to query")
        elif specific_relevance == generic_relevance:
            logger.info("   ⚠️ Responses have similar relevance")
        else:
            logger.info("   ❌ Generic response seems more relevant (unexpected)")
            
        if specific_numbers > generic_numbers:
            logger.info("   ✅ Specific response contains more data points")
        elif specific_numbers == generic_numbers:
            logger.info("   ⚠️ Responses contain similar amounts of data")
        else:
            logger.info("   ❌ Generic response contains more data (unexpected)")
    
    def run_comprehensive_test(self):
        """Run comprehensive test comparing generic vs query-specific analysis."""
        logger.info("🚀 Starting Comprehensive ColPali Query Test")
        logger.info("="*60)
        
        # Load test document
        images = self.load_test_document()
        if not images:
            logger.error("❌ Cannot run test - document loading failed")
            return
        
        # Use first page for testing (typically contains the main chart)
        test_page = images[0] 
        page_number = 1
        
        logger.info(f"🖼️ Testing with page {page_number} ({test_page.size} pixels)")
        logger.info("")
        
        # Get generic analysis once (simulates current precomputed approach)
        logger.info("📋 Step 1: Generate generic analysis (current system)")
        generic_response = self.analyze_with_generic_query(test_page, page_number)
        logger.info("")
        
        # Test each specific query
        results = []
        total_specific_time = 0
        total_specific_tokens = 0
        
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"📋 Step {i+1}: Test query-specific analysis")
            logger.info(f"Query {i}/{len(self.test_queries)}: '{query}'")
            
            # Get query-specific analysis
            specific_response = self.analyze_with_specific_query(query, test_page, page_number)
            
            # Compare responses
            comparison = self.compare_responses(generic_response, specific_response, query)
            results.append(comparison)
            
            # Accumulate metrics
            total_specific_time += specific_response.get('call_time', 0)
            total_specific_tokens += specific_response.get('tokens_used', 0)
            
            logger.info("\n" + "="*80 + "\n")
        
        # Final summary
        self._print_final_summary(results, generic_response, total_specific_time, total_specific_tokens)
        
        return results
    
    def _print_final_summary(self, results, generic_response, total_specific_time, total_specific_tokens):
        """Print comprehensive test summary."""
        logger.info("📊 FINAL TEST SUMMARY")
        logger.info("="*60)
        
        # Performance metrics
        generic_time = generic_response.get('call_time', 0)
        generic_tokens = generic_response.get('tokens_used', 0)
        
        logger.info(f"⏱️ PERFORMANCE COMPARISON:")
        logger.info(f"   Generic approach (1 call): {generic_time:.2f}s, {generic_tokens} tokens")
        logger.info(f"   Specific approach ({len(results)} calls): {total_specific_time:.2f}s, {total_specific_tokens} tokens")
        logger.info(f"   Average per query: {total_specific_time/len(results):.2f}s, {total_specific_tokens//len(results)} tokens")
        logger.info("")
        
        # Response diversity
        logger.info(f"📝 RESPONSE DIVERSITY:")
        generic_answer = generic_response.get('answer', '')
        
        unique_responses = 0
        for result in results:
            specific_answer = result['specific'].get('answer', '')
            # Simple similarity check - if responses differ significantly, count as unique
            if len(specific_answer) > 0 and abs(len(specific_answer) - len(generic_answer)) > 50:
                unique_responses += 1
        
        logger.info(f"   Generic response length: {len(generic_answer)} chars")
        logger.info(f"   Unique specific responses: {unique_responses}/{len(results)}")
        logger.info("")
        
        # Recommendations
        logger.info(f"💡 RECOMMENDATIONS:")
        if unique_responses >= len(results) * 0.8:  # 80% unique
            logger.info("   ✅ PROCEED with query-specific approach")
            logger.info("   ✅ Responses show good query specificity")
            logger.info("   ✅ Performance impact is acceptable")
        elif unique_responses >= len(results) * 0.6:  # 60% unique
            logger.info("   ⚠️ PROCEED with caution - some improvement seen")
            logger.info("   ⚠️ Consider prompt engineering improvements")
        else:
            logger.info("   ❌ Current approach may be better")
            logger.info("   ❌ Query-specific responses not sufficiently different")
        
        if total_specific_time / len(results) > 3.0:  # Average > 3 seconds
            logger.info("   ⚠️ Consider response caching for performance")
        else:
            logger.info("   ✅ Response times are acceptable")

def main():
    """Main test function."""
    tester = QuerySpecificColPaliTester()
    
    if not tester.vlm_client:
        logger.error("❌ Cannot run test - VLM client not available")
        return
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    logger.info("\n🎯 TEST COMPLETE")
    logger.info("Review the results above to determine if query-specific analysis")
    logger.info("produces sufficiently different and relevant responses compared") 
    logger.info("to the current generic precomputed approach.")

if __name__ == "__main__":
    main()