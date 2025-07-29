"""
Debug Salesforce Response Generation

This app tests whether LLM-based answer synthesis produces better Salesforce
responses compared to the current basic HTML cleaning and formatting approach.

The goal is to prove that LLM synthesis works before modifying the main Salesforce system.
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

class SalesforceResponseTester:
    """Test LLM-based answer synthesis vs current HTML formatting."""
    
    def __init__(self):
        self.openai_client = None
        self.model = "gpt-4o-mini"  # Use efficient model for text synthesis
        
        # Initialize OpenAI client
        self._init_openai_client()
        
        # Test queries from user's example and other scenarios
        self.test_queries = [
            "Tell me the steps to modify an air booking?",
            "How to cancel a hotel booking?",
            "What should I do if a guest arrives late for hotel check-in?",
            "How to handle car rental no-show?",
            "What are the air modification procedures?",
        ]
        
        # Mock Salesforce search results (simulating what the connector would return)
        self.mock_salesforce_results = [
            {
                'id': 'ka01234567',
                'title': 'Air Modification',
                'content': '''<p><strong>Purpose</strong></p><p>This article provides a step-by-step guide for modifying a flight booking, ensuring agents follow the correct process efficiently while assisting travelers.</p>

<p><strong>Flight Modification Eligibility</strong></p><p>Before proceeding, verify:</p><ul><li>The airline allows modifications (check fare rules in the GDS or airline website).</li><li>The ticket is within the validity period for changes.</li><li>Traveler agrees to fare differences, penalties, and airline policies.</li><li>Client's travel policy permits the requested modification.</li></ul>

<p><strong>Step-by-Step Process</strong></p><p><strong>1. Retrieve the Booking</strong></p><p>Access the traveler's PNR in the GDS (e.g., Sabre, Amadeus, Travelport).</p><p><strong>Action:</strong> Confirm ticket status (active, unused, partially used).</p>

<p><strong>2. Check Modification Options</strong></p><p><strong>Action:</strong> Review airline fare rules for change fees, fare differences, and restrictions. Determine if the new flight option is available. If a waiver is applicable, follow the airline's waiver process.</p>

<p><strong>3. Inform the Traveler</strong></p><p>Clearly communicate change fees, fare differences, and new flight details. Obtain traveler's confirmation before proceeding.</p>

<p><strong>4. Process the Change</strong></p><p>Modify the itinerary in the GDS per airline guidelines. Reprice the ticket and confirm the updated fare. Apply any applicable waiver codes. Issue a new ticket or revalidate, depending on airline policy.</p>

<p><strong>5. Update Records & Notify Traveler</strong></p><p>Confirm successful modification in the GDS. Update the internal CRM (Salesforce) with new flight details and change rationale. Send updated itinerary and e-ticket confirmation to the traveler.</p>

<p><strong>Special Scenarios</strong></p><p><strong>Same-day Changes:</strong> Some airlines allow same-day flight changes with minimal fees.</p><p><strong>Action:</strong> Check airline policies. <strong>Schedule Changes:</strong> If the airline modified the schedule, verify rebooking options with minimal cost. <strong>Non-Refundable Tickets:</strong> Ensure traveler understands that non-refundable tickets may have high change fees or no change options. <strong>Group Bookings:</strong> Contact the airline's group desk for changes affecting multiple travelers.</p>

<p><strong>Escalation & Support</strong></p><p>If unable to modify the flight in the GDS, contact the airline directly. Escalate complex cases to the support team or supervisor. For policy-related questions, refer to the client's travel policy in Salesforce.</p>''',
                'clean_content': '''Purpose This article provides a step-by-step guide for modifying a flight booking, ensuring agents follow the correct process efficiently while assisting travelers. Flight Modification Eligibility Before proceeding, verify: The airline allows modifications (check fare rules in the GDS or airline website). The ticket is within the validity period for changes. Traveler agrees to fare differences, penalties, and airline policies. Client's travel policy permits the requested modification. Step-by-Step Process 1. Retrieve the Booking Access the traveler's PNR in the GDS (e.g., Sabre, Amadeus, Travelport). Action: Confirm ticket status (active, unused, partially used). 2. Check Modification Options Action: Review airline fare rules for change fees, fare differences, and restrictions. Determine if the new flight option is available. If a waiver is applicable, follow the airline's waiver process. 3. Inform the Traveler Clearly communicate change fees, fare differences, and new flight details. Obtain traveler's confirmation before proceeding. 4. Process the Change Modify the itinerary in the GDS per airline guidelines. Reprice the ticket and confirm the updated fare. Apply any applicable waiver codes. Issue a new ticket or revalidate, depending on airline policy. 5. Update Records & Notify Traveler Confirm successful modification in the GDS. Update the internal CRM (Salesforce) with new flight details and change rationale. Send updated itinerary and e-ticket confirmation to the traveler. Special Scenarios Same-day Changes: Some airlines allow same-day flight changes with minimal fees. Action: Check airline policies. Schedule Changes: If the airline modified the schedule, verify rebooking options with minimal cost. Non-Refundable Tickets: Ensure traveler understands that non-refundable tickets may have high change fees or no change options. Group Bookings: Contact the airline's group desk for changes affecting multiple travelers. Escalation & Support If unable to modify the flight in the GDS, contact the airline directly. Escalate complex cases to the support team or supervisor. For policy-related questions, refer to the client's travel policy in Salesforce.''',
                'relevance_score': 1.0,
                'source': 'Salesforce Knowledge Article',
                'type': 'salesforce_knowledge'
            },
            {
                'id': 'ka01234568',
                'title': 'Hotel Guest Services - Late Arrival Procedures',
                'content': '''<h2>Overview</h2><p>This document outlines procedures for handling guests who arrive late for their hotel reservation, including after-hours check-in processes and room availability management.</p>

<h3>Standard Late Arrival Process</h3><p><strong>Step 1:</strong> Verify reservation details in the property management system (PMS)</p><p><strong>Step 2:</strong> Check room availability and confirm the reserved room type</p><p><strong>Step 3:</strong> Process check-in following standard procedures</p><p><strong>Step 4:</strong> Provide room keys and hotel information</p>

<h3>After-Hours Procedures</h3><p>For arrivals after 11 PM:</p><ul><li>Security escort may be required for safety</li><li>Limited amenities available (restaurant, room service may be closed)</li><li>Night audit staff handles check-in process</li><li>Inform guest of morning check-out procedures</li></ul>

<h3>Special Considerations</h3><p><strong>Very Late Arrivals (After 2 AM):</strong> Contact guest services manager if issues arise. May require special authorization for room assignment.</p><p><strong>No-Show vs Late Arrival:</strong> Distinguish between confirmed late arrival and potential no-show. Follow no-show procedures if guest hasn't confirmed late arrival.</p>

<h3>Guest Communication</h3><p>Always inform late-arriving guests about:</p><ul><li>Available amenities during their stay</li><li>Breakfast service hours</li><li>Check-out procedures</li><li>Any room service or facility limitations</li></ul>''',
                'clean_content': '''Overview This document outlines procedures for handling guests who arrive late for their hotel reservation, including after-hours check-in processes and room availability management. Standard Late Arrival Process Step 1: Verify reservation details in the property management system (PMS) Step 2: Check room availability and confirm the reserved room type Step 3: Process check-in following standard procedures Step 4: Provide room keys and hotel information After-Hours Procedures For arrivals after 11 PM: Security escort may be required for safety Limited amenities available (restaurant, room service may be closed) Night audit staff handles check-in process Inform guest of morning check-out procedures Special Considerations Very Late Arrivals (After 2 AM): Contact guest services manager if issues arise. May require special authorization for room assignment. No-Show vs Late Arrival: Distinguish between confirmed late arrival and potential no-show. Follow no-show procedures if guest hasn't confirmed late arrival. Guest Communication Always inform late-arriving guests about: Available amenities during their stay Breakfast service hours Check-out procedures Any room service or facility limitations''',
                'relevance_score': 0.85,
                'source': 'Salesforce Knowledge Article',
                'type': 'salesforce_knowledge'
            }
        ]
        
        logger.info("✅ SalesforceResponseTester initialized")
    
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
    
    def generate_current_approach_answer(self, query: str, sf_results: List[Dict]) -> str:
        """
        Simulate current Salesforce approach: basic HTML cleaning and formatting.
        This replicates the logic from streamlit_rag_app.py _extract_sf_content and _format_salesforce_content.
        """
        logger.info(f"🔄 CURRENT APPROACH (HTML cleaning + formatting):")
        logger.info(f"   Query: '{query}'")
        
        try:
            if not sf_results:
                return "No Salesforce content available"
            
            best_result = sf_results[0]
            title = best_result.get('title', 'Knowledge Article')
            content = best_result.get('clean_content', 'No content available')
            
            # Simulate current _format_salesforce_content logic
            import re
            
            # Split into sentences for better processing
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            formatted_lines = []
            current_section = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Detect numbered lists (1., 2., etc.)
                if re.match(r'^\d+\.', sentence):
                    if current_section:
                        formatted_lines.append(' '.join(current_section))
                        current_section = []
                    formatted_lines.append(f"\n**{sentence}**")
                
                # Detect lettered lists (a., b., etc.)
                elif re.match(r'^[a-zA-Z]\.', sentence):
                    if current_section:
                        formatted_lines.append(' '.join(current_section))
                        current_section = []
                    formatted_lines.append(f"\n• {sentence}")
                
                # Detect action words that suggest new sections
                elif re.match(r'^(Contact|Call|Email|Visit|Check|Verify|Confirm|Review|Submit|Complete)', sentence, re.IGNORECASE):
                    if current_section:
                        formatted_lines.append(' '.join(current_section))
                        current_section = []
                    formatted_lines.append(f"\n**Action:** {sentence}")
                
                # Detect important keywords that should be emphasized
                elif re.search(r'\b(important|note|warning|attention|remember|caution)\b', sentence, re.IGNORECASE):
                    if current_section:
                        formatted_lines.append(' '.join(current_section))
                        current_section = []
                    formatted_lines.append(f"\n⚠️ **Important:** {sentence}")
                
                # Regular sentence - accumulate
                else:
                    current_section.append(sentence)
            
            # Add any remaining content
            if current_section:
                formatted_lines.append(' '.join(current_section))
            
            # Join and clean up extra whitespace
            result = '\n'.join(formatted_lines)
            result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)  # Normalize multiple line breaks
            formatted_content = result.strip()
            
            # Current approach wraps in title
            answer = f"**Based on '{title}':**\n\n{formatted_content}"
            
            logger.info(f"   Generated answer: {len(answer)} chars")
            logger.info(f"   Preview: {answer[:150]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ Current approach failed: {e}")
            return f"Current approach failed: {e}"
    
    def generate_llm_synthesis_answer(self, query: str, sf_results: List[Dict]) -> str:
        """
        Test new LLM-based synthesis approach.
        Similar to ColPali's VLM approach but for Salesforce text content.
        """
        logger.info(f"🎯 LLM SYNTHESIS APPROACH (proposed):")
        logger.info(f"   Query: '{query}'")
        
        start_time = time.time()
        
        try:
            if not sf_results:
                return "No Salesforce content available"
            
            # Use the best result (highest relevance score)
            best_result = sf_results[0]
            title = best_result.get('title', 'Knowledge Article')
            clean_content = best_result.get('clean_content', 'No content available')
            
            # Create query-specific system prompt (similar to ColPali approach)
            system_prompt = f"""You are an expert customer service analyst examining Salesforce knowledge article '{title}'.

CRITICAL INSTRUCTIONS for query: "{query}"
1. Read the article content carefully and extract specific information that directly answers the question
2. Focus ONLY on information relevant to this specific query  
3. Provide a clear, step-by-step answer if the query asks for procedures or steps
4. Convert administrative/technical language into natural, actionable guidance
5. DO NOT include generic purposes, overviews, or background unless directly relevant
6. Prioritize actionable steps and specific procedures over general information
7. If the query asks for "steps" or "how to", structure your response as a clear process

Your response should:
- Start with the specific information that answers "{query}"
- Be concise but complete with all necessary steps
- Use natural language, not technical jargon or administrative headers
- Focus on what the user needs to DO, not background information
- Structure steps clearly if it's a procedural query"""

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
                        "content": f"Based on this Salesforce knowledge article, please answer: {query}\n\nARTICLE CONTENT:\n{clean_content}\n\nPlease provide a clear, specific answer that directly addresses the question."
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
            logger.info(f"   Preview: {answer[:150]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"❌ LLM synthesis failed: {e}")
            return f"LLM synthesis failed: {e}"
    
    def compare_approaches(self, query: str, sf_results: List[Dict]):
        """Compare current approach vs LLM synthesis for a specific query."""
        logger.info(f"\n📊 COMPARING APPROACHES for: '{query}'")
        logger.info("="*80)
        
        # Generate answer with current approach
        current_answer = self.generate_current_approach_answer(query, sf_results)
        
        # Generate answer with LLM synthesis
        llm_answer = self.generate_llm_synthesis_answer(query, sf_results)
        
        # Analysis and comparison
        logger.info(f"\n📋 RESPONSE COMPARISON:")
        logger.info("-" * 60)
        
        logger.info(f"🔄 CURRENT (HTML cleaning + formatting):")
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
        key_quality_indicators = []
        
        if 'steps' in query_lower or 'how to' in query_lower:
            key_quality_indicators.extend(['step', 'first', 'then', 'next', 'finally', 'process'])
        elif 'modify' in query_lower or 'change' in query_lower:
            key_quality_indicators.extend(['modify', 'change', 'update', 'process', 'procedure'])
        elif 'cancel' in query_lower:
            key_quality_indicators.extend(['cancel', 'cancellation', 'refund', 'process'])
        elif 'late' in query_lower or 'arrival' in query_lower:
            key_quality_indicators.extend(['late', 'arrival', 'check-in', 'procedure', 'process'])
        
        # Check for administrative fluff (bad indicators)
        admin_fluff = ['purpose', 'overview', 'this article', 'this document', 'background']
        
        # Count good indicators in each response
        current_good = sum(1 for term in key_quality_indicators if term in current_answer.lower())
        llm_good = sum(1 for term in key_quality_indicators if term in llm_answer.lower())
        
        # Count bad indicators (administrative content)
        current_fluff = sum(1 for term in admin_fluff if term in current_answer.lower())
        llm_fluff = sum(1 for term in admin_fluff if term in llm_answer.lower())
        
        logger.info(f"   Key indicators expected: {key_quality_indicators}")
        logger.info(f"   Current response good indicators: {current_good}")
        logger.info(f"   LLM response good indicators: {llm_good}")
        logger.info(f"   Current response admin fluff: {current_fluff}")
        logger.info(f"   LLM response admin fluff: {llm_fluff}")
        
        # Check for actionable content
        current_actionable = self._check_actionable_content(current_answer, query)
        llm_actionable = self._check_actionable_content(llm_answer, query)
        
        logger.info(f"   Current answer actionability: {current_actionable}")
        logger.info(f"   LLM answer actionability: {llm_actionable}")
        
        # Overall assessment
        if llm_good > current_good and llm_fluff < current_fluff and llm_actionable > current_actionable:
            logger.info("   ✅ LLM synthesis shows clear improvement")
        elif llm_good > current_good or llm_actionable > current_actionable:
            logger.info("   ⚠️ LLM synthesis shows some improvement")
        else:
            logger.info("   ❌ LLM synthesis may not be better (needs investigation)")
    
    def _check_actionable_content(self, answer: str, query: str) -> int:
        """Check how actionable the answer is for the user."""
        actionability_score = 0
        answer_lower = answer.lower()
        query_lower = query.lower()
        
        # Query-specific actionability checks
        if 'steps' in query_lower or 'how to' in query_lower:
            # Look for step-by-step structure
            if any(indicator in answer_lower for indicator in ['step 1', 'first', '1.', 'begin by', 'start by']):
                actionability_score += 3
            if any(indicator in answer_lower for indicator in ['step 2', 'then', '2.', 'next', 'after']):
                actionability_score += 2
            if any(indicator in answer_lower for indicator in ['step 3', 'finally', '3.', 'complete', 'finish']):
                actionability_score += 2
        
        if 'modify' in query_lower or 'change' in query_lower:
            if any(indicator in answer_lower for indicator in ['access', 'retrieve', 'check', 'confirm', 'process']):
                actionability_score += 2
        
        # General actionability indicators
        if any(word in answer_lower for word in ['you need to', 'you should', 'follow these', 'to do this']):
            actionability_score += 1
        
        # Penalty for administrative content at the beginning
        if any(fluff in answer_lower[:200] for fluff in ['purpose', 'overview', 'this article provides', 'background']):
            actionability_score -= 2
        
        return max(0, actionability_score)
    
    def run_comprehensive_test(self):
        """Run comprehensive test comparing approaches across all queries."""
        logger.info("🚀 Starting Comprehensive Salesforce Response Test")
        logger.info("="*60)
        
        if not self.openai_client:
            logger.error("❌ Cannot run test - OpenAI client not available")
            return
        
        results = []
        total_improvement_count = 0
        
        for i, query in enumerate(self.test_queries, 1):
            logger.info(f"\n📋 Test {i}/{len(self.test_queries)}")
            
            # Find relevant mock result for this query
            relevant_result = self._find_relevant_mock_result(query)
            
            # Compare approaches for this query
            comparison = self.compare_approaches(query, [relevant_result] if relevant_result else [])
            results.append(comparison)
            
            # Simple improvement check based on actionability and reduced fluff
            current_len = len(comparison['current_answer'])
            llm_len = len(comparison['llm_answer'])
            
            # LLM is better if it's more concise AND actionable (or similar length but more focused)
            if (llm_len < current_len * 0.8 and llm_len > 200) or self._is_llm_response_better(comparison):
                total_improvement_count += 1
            
            logger.info("\n" + "="*80)
        
        # Final summary
        self._print_final_summary(results, total_improvement_count)
        
        return results
    
    def _find_relevant_mock_result(self, query: str) -> Dict:
        """Find the most relevant mock result for a given query."""
        query_lower = query.lower()
        
        # Simple keyword matching to find relevant mock data
        for result in self.mock_salesforce_results:
            title_lower = result['title'].lower()
            if ('modify' in query_lower or 'air' in query_lower) and 'air' in title_lower:
                return result
            elif ('late' in query_lower or 'hotel' in query_lower) and 'hotel' in title_lower:
                return result
        
        # Default to first result if no specific match
        return self.mock_salesforce_results[0]
    
    def _is_llm_response_better(self, comparison: Dict) -> bool:
        """Determine if LLM response is qualitatively better."""
        current = comparison['current_answer'].lower()
        llm = comparison['llm_answer'].lower()
        
        # Check for better structure (steps, actionable content)
        llm_has_steps = any(word in llm for word in ['step 1', 'first', '1.', 'then', 'next'])
        current_has_steps = any(word in current for word in ['step 1', 'first', '1.', 'then', 'next'])
        
        # Check for reduced administrative fluff
        admin_words = ['purpose', 'overview', 'this article', 'background', 'eligibility']
        current_fluff = sum(1 for word in admin_words if word in current[:300])  # Check first 300 chars
        llm_fluff = sum(1 for word in admin_words if word in llm[:300])
        
        # LLM is better if it has better structure AND less fluff
        return (llm_has_steps and not current_has_steps) or (llm_fluff < current_fluff and llm_fluff <= 1)
    
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
            logger.info("   ✅ LLM responses show significant improvement over HTML formatting")
            logger.info("   ✅ Quality should match ColPali's VLM and Text RAG's LLM responses")
        elif improvement_count >= len(results) * 0.6:  # 60% improvement
            logger.info("   ⚠️ PROCEED with caution - mixed results")
            logger.info("   ⚠️ Consider prompt engineering improvements")
        else:
            logger.info("   ❌ Current approach may be sufficient")
            logger.info("   ❌ LLM synthesis not showing clear benefits")
        
        logger.info("")
        logger.info("🎯 NEXT STEPS:")
        logger.info("   1. Review individual query results above")
        logger.info("   2. If LLM synthesis shows improvement, integrate into main Salesforce connector")
        logger.info("   3. Test with real Salesforce connection using user's 10 articles")
        logger.info("   4. Ensure integration doesn't break existing search functionality")

def main():
    """Main test function."""
    tester = SalesforceResponseTester()
    
    if not tester.openai_client:
        logger.error("❌ Cannot run test - OpenAI client not available")
        return
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    logger.info("\n🎯 TEST COMPLETE")
    logger.info("Review the results above to determine if LLM synthesis")
    logger.info("produces better Salesforce responses compared to HTML formatting.")

if __name__ == "__main__":
    main()