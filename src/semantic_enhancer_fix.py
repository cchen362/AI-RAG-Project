    def _simplify_query(self, query: str) -> str:
        """
        Simplify query by removing complex phrasing and questions.
        """
        import re
        
        # Remove question phrases
        simplified = re.sub(r'^(what|how|when|where|why)\s+(to|do|if|should|can|will)\s+', '', query.lower())
        simplified = re.sub(r'\?$', '', simplified)
        simplified = re.sub(r'\s+', ' ', simplified).strip()
        
        return simplified if simplified else query
    
    def enhanced_search_with_fallbacks(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Enhanced search that uses multiple fallback strategies when primary search fails.
        """
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
        
        for i, fallback_query in enumerate(fallback_queries[1:], 1):  # Skip original query
            try:
                self.logger.info(f"📝 Fallback {i}: Trying '{fallback_query}'")
                fallback_result = self.transformative_search(fallback_query, limit)
                
                if fallback_result['articles'] and fallback_result['confidence'] > best_fallback_result['confidence']:
                    best_fallback_result = fallback_result
                    best_fallback_result['fallback_query'] = fallback_query
                    self.logger.info(f"💡 Better fallback found (confidence: {fallback_result['confidence']:.2f})")
                    
                    # If we found good results, use them
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
                    # Convert to expected format
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
