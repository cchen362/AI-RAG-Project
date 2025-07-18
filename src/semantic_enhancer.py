"""
🚀 TRANSFORMATIVE SEMANTIC SEARCH - Complete Implementation with Critical Fixes

FIXED ISSUES:
✅ OpenAI API v1.0+ compatibility 
✅ Enhanced intent extraction for complex scenarios
✅ Improved search strategy and relevance scoring
✅ Enhanced fallback mechanisms

"""

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

import json
import time
import logging
from typing import List, Dict, Any, Optional
import re
import html

class TransformativeSemanticSearch:
    """🧠 The ultimate semantic search combining multiple AI approaches."""
    
    def __init__(self, salesforce_connector, openai_api_key: str = None):
        self.sf_connector = salesforce_connector
        self.logger = logging.getLogger(__name__)
        
        # Setup OpenAI if available - FIXED: v1.0+ syntax
        self.openai_available = False
        self.openai_client = None
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                self.openai_available = True
                self.logger.info("✅ OpenAI integration enabled for LLM-enhanced search")
            except Exception as e:
                self.logger.warning(f"OpenAI setup failed: {e}")
        elif not OPENAI_AVAILABLE:
            self.logger.warning("OpenAI library not installed. Install with: pip install openai")
        elif not openai_api_key:
            self.logger.info("No OpenAI API key provided. Set OPENAI_API_KEY environment variable for full capabilities.")
        
        self._articles_cache = None
        self._cache_timestamp = None
        
    def transformative_search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """🚀 TRANSFORMATIVE SEARCH: The most intelligent search possible."""
        self.logger.info(f"🚀 Starting transformative search for: {query}")
        start_time = time.time()
        
        results = {
            'query': query,
            'methods_used': [],
            'articles': [],
            'confidence': 0.0,
            'explanation': '',
            'search_time': 0.0
        }
        
        # Phase 1: Intent-driven keyword search
        intent_results = []
        try:
            intent_results = self.sf_connector.search_knowledge_with_intent(query, limit)
            if intent_results:
                results['methods_used'].append('intent_keyword')
                avg_intent_score = sum(r['relevance_score'] for r in intent_results) / len(intent_results)
                
                if avg_intent_score > 0.9:
                    results['articles'] = intent_results
                    results['confidence'] = avg_intent_score
                    results['explanation'] = f"Found {len(intent_results)} highly relevant articles using intent-driven search"
                    results['search_time'] = time.time() - start_time
                    return results
        except Exception as e:
            self.logger.warning(f"Intent search failed: {e}")
        
        # Phase 2: LLM-Enhanced Semantic Search
        if self.openai_available:
            try:
                semantic_results = self._llm_semantic_search(query, limit * 2)
                if semantic_results:
                    results['methods_used'].append('llm_semantic')
                    combined_results = self._intelligent_result_combination(query, intent_results, semantic_results, limit)
                    
                    if combined_results:
                        results['articles'] = combined_results['articles']
                        results['confidence'] = combined_results['confidence']
                        results['explanation'] = combined_results['explanation']
                        results['methods_used'].extend(combined_results['methods_used'])
                    else:
                        if semantic_results:
                            results['articles'] = semantic_results[:limit]
                            results['confidence'] = sum(r.get('semantic_score', 0.5) for r in semantic_results) / len(semantic_results)
                            results['explanation'] = f"Found {len(semantic_results)} articles using LLM semantic analysis"
                        elif intent_results:
                            results['articles'] = intent_results
                            results['confidence'] = sum(r['relevance_score'] for r in intent_results) / len(intent_results)
                            results['explanation'] = f"Found {len(intent_results)} articles using intent-driven search"
            except Exception as e:
                self.logger.warning(f"LLM semantic search failed: {e}")
                if intent_results:
                    results['articles'] = intent_results
                    results['confidence'] = sum(r['relevance_score'] for r in intent_results) / len(intent_results)
                    results['explanation'] = f"Found {len(intent_results)} articles using intent-driven search (LLM fallback)"
        else:
            if intent_results:
                results['articles'] = intent_results
                results['confidence'] = sum(r['relevance_score'] for r in intent_results) / len(intent_results)
                results['explanation'] = f"Found {len(intent_results)} articles using intent-driven search"
        
        if not results['articles']:
            results['explanation'] = "No relevant articles found using any search method"
            results['confidence'] = 0.0
        
        results['search_time'] = time.time() - start_time
        return results
    
    def _llm_semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Use LLM to perform semantic search and analysis."""
        articles = self._get_all_articles()
        if not articles:
            return []
        
        query_analysis = self._analyze_query_with_llm(query)
        relevant_articles = self._analyze_articles_with_llm(query, query_analysis, articles, limit)
        return relevant_articles
    
    def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to deeply understand the user's query."""
        analysis_prompt = f'''
        Analyze this user query: "{query}"
        
        Extract: primary intent, domain, urgency, context clues, related concepts, expected answer type
        
        Respond with JSON only:
        {{
            "primary_intent": "string",
            "domain": "string", 
            "urgency": "low|medium|high",
            "context_clues": "string",
            "related_concepts": ["concept1", "concept2"],
            "expected_answer_type": "policy|procedure|steps|contact|general"
        }}
        '''
        
        try:
            # FIXED: Use new OpenAI v1.0+ syntax
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=400
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            self.logger.warning(f"LLM query analysis failed: {e}")
            return {
                "primary_intent": query,
                "domain": "unknown",
                "urgency": "medium",
                "context_clues": "",
                "related_concepts": [],
                "expected_answer_type": "general"
            }
    
    def _analyze_articles_with_llm(self, query: str, query_analysis: Dict, articles: List[Dict], limit: int) -> List[Dict]:
        """Use LLM to analyze which articles are most relevant."""
        batch_size = 3
        all_analyses = []
        
        for i in range(0, min(len(articles), 15), batch_size):
            batch = articles[i:i + batch_size]
            batch_analysis = self._analyze_article_batch_with_llm(query, query_analysis, batch)
            all_analyses.extend(batch_analysis)
        
        relevant_articles = [a for a in all_analyses if a.get('semantic_score', 0) > 0.3]
        relevant_articles.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
        return relevant_articles[:limit]
    
    def _analyze_article_batch_with_llm(self, query: str, query_analysis: Dict, articles: List[Dict]) -> List[Dict]:
        """Analyze a batch of articles with LLM."""
        article_data = []
        for i, article in enumerate(articles):
            content = article.get('content', '')
            clean_content = re.sub(r'<[^>]+>', ' ', content)
            clean_content = html.unescape(clean_content)
            clean_content = re.sub(r'\\s+', ' ', clean_content).strip()
            
            article_data.append({
                'index': i,
                'title': article.get('title', ''),
                'content_preview': clean_content[:800]
            })
        
        analysis_prompt = f'''
        User Query: "{query}"
        Analyze each article for relevance. Provide JSON array with relevance_score, reasoning, key_content for each.
        
        Articles: {json.dumps(article_data, indent=2)}
        '''
        
        try:
            # FIXED: Use new OpenAI v1.0+ syntax
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            analyses = json.loads(response.choices[0].message.content)
            
            result_articles = []
            for analysis in analyses:
                if isinstance(analysis, dict) and 'index' in analysis:
                    idx = analysis['index']
                    if 0 <= idx < len(articles):
                        article = articles[idx].copy()
                        article.update({
                            'semantic_score': analysis.get('relevance_score', 0.0),
                            'semantic_reasoning': analysis.get('reasoning', ''),
                            'key_content': analysis.get('key_content', ''),
                            'addresses_intent': analysis.get('addresses_intent', False),
                            'search_method': 'llm_semantic'
                        })
                        result_articles.append(article)
            
            return result_articles
            
        except Exception as e:
            self.logger.warning(f"LLM article analysis failed: {e}")
            return [{
                **article,
                'semantic_score': 0.3,
                'semantic_reasoning': 'LLM analysis failed',
                'search_method': 'llm_semantic_fallback'
            } for article in articles]
    
    def _intelligent_result_combination(self, query: str, intent_results: List[Dict], 
                                      semantic_results: List[Dict], limit: int) -> Optional[Dict]:
        """Intelligently combine results from different search methods."""
        if not intent_results and not semantic_results:
            return None
        
        combined = {}
        methods_used = []
        
        for result in semantic_results:
            if result.get('semantic_score', 0) > 0.5:
                article_id = result.get('id')
                combined[article_id] = result
                if 'llm_semantic' not in methods_used:
                    methods_used.append('llm_semantic')
        
        for result in intent_results:
            article_id = result.get('id')
            if article_id not in combined:
                combined[article_id] = result
                if 'intent_keyword' not in methods_used:
                    methods_used.append('intent_keyword')
            else:
                existing = combined[article_id]
                boost_score = (existing.get('semantic_score', existing.get('relevance_score', 0.5)) + 
                             result.get('relevance_score', 0.5)) / 2
                existing['combined_score'] = min(1.0, boost_score + 0.2)
                existing['found_by_both'] = True
        
        final_articles = []
        for article in combined.values():
            if 'combined_score' in article:
                article['final_score'] = article['combined_score']
            else:
                article['final_score'] = max(
                    article.get('semantic_score', 0),
                    article.get('relevance_score', 0)
                )
            final_articles.append(article)
        
        final_articles.sort(key=lambda x: x['final_score'], reverse=True)
        final_articles = final_articles[:limit]
        
        if final_articles:
            avg_confidence = sum(a['final_score'] for a in final_articles) / len(final_articles)
            both_methods = sum(1 for a in final_articles if a.get('found_by_both', False))
            
            explanation = f"Combined {len(semantic_results)} semantic + {len(intent_results)} intent results"
            if both_methods > 0:
                explanation += f", {both_methods} found by both methods"
                methods_used.append('hybrid_boost')
            
            return {
                'articles': final_articles,
                'confidence': avg_confidence,
                'explanation': explanation,
                'methods_used': methods_used
            }
        
        return None
    
    def _get_all_articles(self) -> List[Dict]:
        """Get all articles, using cache if available."""
        current_time = time.time()
        
        if (self._articles_cache is not None and 
            self._cache_timestamp is not None and
            current_time - self._cache_timestamp < 3600):
            return self._articles_cache
        
        try:
            if hasattr(self.sf_connector, 'sf') and self.sf_connector.sf:
                query = '''
                SELECT Id, Title, Article_Body__c
                FROM Knowledge__kav 
                WHERE PublishStatus = 'Online'
                AND Language = 'en_US'
                ORDER BY LastModifiedDate DESC
                LIMIT 30
                '''
                
                result = self.sf_connector.sf.query(query)
                articles = []
                
                for record in result['records']:
                    article = {
                        'id': record['Id'],
                        'title': record['Title'],
                        'content': record.get('Article_Body__c', ''),
                        'source': 'Salesforce Knowledge Article',
                        'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                        'type': 'salesforce_knowledge'
                    }
                    articles.append(article)
                
                self._articles_cache = articles
                self._cache_timestamp = current_time
                return articles
            else:
                self.logger.warning("Salesforce connection not available")
                return []
                
        except Exception as e:
            self.logger.error(f"Failed to fetch articles: {e}")
            return []
    
    # 🔧 ENHANCED FALLBACK MECHANISMS
    
    def expand_query_with_fallbacks(self, query: str, max_expansions: int = 3, fallback_mode: bool = False) -> List[str]:
        """Enhanced query expansion with sophisticated fallback mechanisms."""
        base_expansions = [
            query,
            self._simplify_query(query),
        ]
        
        if fallback_mode:
            base_expansions.extend([
                self._extract_key_terms(query),
                self._generate_topic_based_query(query),
                self._create_generic_fallback(query)
            ])
        
        final_expansions = []
        seen = set()
        for expansion in base_expansions:
            if expansion and expansion not in seen:
                final_expansions.append(expansion)
                seen.add(expansion)
                if len(final_expansions) >= max_expansions:
                    break
        
        mode_indicator = " (FALLBACK MODE)" if fallback_mode else ""
        self.logger.info(f"Query expansion{mode_indicator}: {final_expansions}")
        return final_expansions
    
    def _extract_key_terms(self, query: str) -> str:
        """Extract only the most essential terms from a query."""
        stop_patterns = [
            r'\\b(what|how|when|where|why|who|which)\\b',
            r'\\b(to|do|if|should|can|will|would|could)\\b',
            r'\\b(the|a|an|and|or|but|in|on|at|for)\\b'
        ]
        
        cleaned = query.lower()
        for pattern in stop_patterns:
            cleaned = re.sub(pattern, ' ', cleaned)
        
        terms = [t.strip() for t in cleaned.split() if len(t.strip()) > 2]
        return ' '.join(terms[:3])
    
    def _generate_topic_based_query(self, query: str) -> str:
        """Generate a topic-based query focusing on the main subject area."""
        query_lower = query.lower()
        
        if 'late' in query_lower or 'delayed' in query_lower:
            return "arrival procedures policies"
        elif 'no-show' in query_lower or 'missed' in query_lower:
            return "missed appointment policies"
        elif 'angry' in query_lower or 'complaint' in query_lower or 'escalate' in query_lower:
            return "customer service escalation"
        elif 'cancel' in query_lower:
            return "cancellation procedures"
        elif 'modify' in query_lower or 'change' in query_lower:
            return "modification procedures"
        elif 'refund' in query_lower:
            return "refund policies"
        else:
            if 'hotel' in query_lower:
                return "hotel procedures"
            elif 'flight' in query_lower or 'air' in query_lower:
                return "airline procedures"
            elif 'car' in query_lower or 'rental' in query_lower:
                return "rental procedures"
            else:
                return "customer service procedures"
    
    def _create_generic_fallback(self, query: str) -> str:
        """Create a very generic fallback query as last resort."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['hotel', 'room', 'guest']):
            return "hotel guest services"
        elif any(term in query_lower for term in ['flight', 'air', 'airline', 'passenger']):
            return "airline passenger services"
        elif any(term in query_lower for term in ['car', 'rental', 'vehicle']):
            return "car rental services"
        else:
            return "customer service guidelines"
    
    def _simplify_query(self, query: str) -> str:
        """Simplify query by removing complex phrasing and questions."""
        simplified = re.sub(r'^(what|how|when|where|why)\\s+(to|do|if|should|can|will)\\s+', '', query.lower())
        simplified = re.sub(r'\\?$', '', simplified)
        simplified = re.sub(r'\\s+', ' ', simplified).strip()
        
        return simplified if simplified else query
    
    def enhanced_search_with_fallbacks(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Enhanced search that uses multiple fallback strategies when primary search fails."""
        self.logger.info(f"🔍 Starting enhanced search with fallbacks for: '{query}'")
        
        # Strategy 1: Primary transformative search
        try:
            primary_result = self.transformative_search(query, limit)
            if primary_result['articles'] and primary_result['confidence'] > 0.4:
                self.logger.info(f"✅ Primary search successful (confidence: {primary_result['confidence']:.2f})")
                primary_result['fallback_used'] = False
                primary_result['search_strategy'] = 'primary_transformative'
                return primary_result
            else:
                self.logger.info(f"⚠️ Primary search insufficient (confidence: {primary_result['confidence']:.2f})")
        except Exception as e:
            self.logger.warning(f"Primary search failed: {e}")
            primary_result = {'articles': [], 'confidence': 0.0}
        
        # Strategy 2: Fallback with expanded queries
        self.logger.info("🔄 Attempting fallback with expanded queries...")
        fallback_queries = self.expand_query_with_fallbacks(query, max_expansions=5, fallback_mode=True)
        
        best_fallback_result = {'articles': [], 'confidence': 0.0}
        
        for i, fallback_query in enumerate(fallback_queries[1:], 1):
            try:
                self.logger.info(f"📝 Fallback {i}: Trying '{fallback_query}'")
                fallback_result = self.transformative_search(fallback_query, limit)
                
                if fallback_result['articles'] and fallback_result['confidence'] > best_fallback_result['confidence']:
                    best_fallback_result = fallback_result
                    best_fallback_result['fallback_query'] = fallback_query
                    self.logger.info(f"💡 Better fallback found (confidence: {fallback_result['confidence']:.2f})")
                    
                    if fallback_result['confidence'] > 0.3:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Fallback query {i} failed: {e}")
                continue
        
        # Strategy 3: Intent-based search as final fallback
        if not best_fallback_result['articles'] or best_fallback_result['confidence'] < 0.25:
            self.logger.info("🎯 Attempting intent-based search as final fallback...")
            try:
                intent_results = self.sf_connector.search_knowledge_with_intent(query, limit)
                if intent_results:
                    best_fallback_result = {
                        'articles': intent_results,
                        'confidence': sum(r.get('relevance_score', 0) for r in intent_results) / len(intent_results),
                        'explanation': "Found results using intent-based keyword search",
                        'methods_used': ['intent_keyword_fallback'],
                        'search_time': 0.5
                    }
                    self.logger.info(f"🆘 Intent fallback found {len(intent_results)} results")
            except Exception as e:
                self.logger.warning(f"Intent fallback failed: {e}")
        
        # Prepare final result
        if best_fallback_result['articles']:
            final_result = best_fallback_result
            final_result['fallback_used'] = True
            final_result['search_strategy'] = 'enhanced_fallback'
            final_result['original_query'] = query
            self.logger.info(f"✅ Fallback search successful (confidence: {final_result['confidence']:.2f})")
        else:
            # Honest failure
            final_result = {
                'articles': [],
                'confidence': 0.0,
                'explanation': f"No relevant articles found for '{query}' using any search strategy",
                'methods_used': ['all_strategies_exhausted'],
                'search_time': 0.0,
                'fallback_used': True,
                'search_strategy': 'honest_failure',
                'original_query': query
            }
            self.logger.info(f"❌ All search strategies exhausted - honest failure")
        
        return final_result
    
    def analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Public wrapper for LLM query analysis."""
        return self._analyze_query_with_llm(query)
