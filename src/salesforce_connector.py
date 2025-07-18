import os
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceAuthenticationFailed
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
from datetime import datetime
import time

class SalesforceConnector:
    def __init__(self):
        load_dotenv()
        self.username = os.getenv('SALESFORCE_USERNAME')
        self.password = os.getenv('SALESFORCE_PASSWORD')
        self.security_token = os.getenv('SALESFORCE_SECURITY_TOKEN')
        self.client_id = os.getenv('SALESFORCE_CLIENT_ID')
        self.client_secret = os.getenv('SALESFORCE_CLIENT_SECRET')

        self.sf = None
        self.logger = logging.getLogger(__name__)

    def authenticate(self):
        """
        Real-world analogy: Like showing your ID badge at the office entrance
        """
        try:
            # Check if we have required credentials
            if not self.username or not self.password:
                self.logger.error("Missing required Salesforce credentials (username/password)")
                return False
            
            self.logger.info(f"Attempting authentication for user: {self.username}")
            
            # Import required modules for manual authentication
            from simple_salesforce.api import SalesforceLogin
            import requests
            
            # Create session
            session = requests.Session()
            
            # Try manual authentication approach that worked in testing
            try:
                self.logger.info("Trying manual SalesforceLogin approach...")
                session_id, instance = SalesforceLogin(
                    username=self.username,
                    password=self.password,
                    security_token=self.security_token,
                    sf_version='58.0',
                    session=session
                )
                
                # Create Salesforce instance using the session
                self.sf = Salesforce(
                    session_id=session_id,
                    instance=instance,
                    session=session
                )
                
                # Test the connection
                org_info = self.sf.query("SELECT Name FROM Organization LIMIT 1")
                org_name = org_info['records'][0]['Name'] if org_info['records'] else 'Unknown'
                
                self.logger.info(f"Successfully authenticated with Salesforce")
                self.logger.info(f"Connected to organization: {org_name}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Manual authentication failed: {e}")
                
                # Fallback to standard methods if manual approach fails
                auth_methods = []
                
                # Method 1: Username + Password + Security Token (Production)
                if self.security_token:
                    password_with_token = self.password + self.security_token
                    auth_methods.append({
                        'name': 'Production with Security Token',
                        'params': {
                            'username': self.username,
                            'password': password_with_token
                        }
                    })
                
                # Method 2: Username + Password for Sandbox
                auth_methods.append({
                    'name': 'Sandbox',
                    'params': {
                        'username': self.username,
                        'password': self.password,
                        'domain': 'test'
                    }
                })
                
                # Try fallback methods
                for method in auth_methods:
                    try:
                        self.logger.info(f"Trying fallback {method['name']} authentication...")
                        self.sf = Salesforce(**method['params'])
                        
                        # Test the connection
                        org_info = self.sf.query("SELECT Name FROM Organization LIMIT 1")
                        org_name = org_info['records'][0]['Name'] if org_info['records'] else 'Unknown'
                        
                        self.logger.info(f"Successfully authenticated with Salesforce using {method['name']}")
                        self.logger.info(f"Connected to organization: {org_name}")
                        return True
                        
                    except Exception as fallback_e:
                        self.logger.warning(f"{method['name']} authentication failed: {fallback_e}")
                        continue
            
            self.logger.error("All authentication methods failed")
            return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error during authentication: {e}")
            return False
        
    def test_connection(self):
        """
        Real-world analogy: Like checking if your phone has signal
        """
        if not self.sf:
            if not self.authenticate():
                return False
            
        try:
            # Try to get basic org information
            org_info = self.sf.query("SELECT Name, Id from Organization LIMIT 1")
            self.logger.info(f"Connected to Salesforce org: {org_info['records'][0]['Name']}")
            return True
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
        
    def get_knowledge_articles(self, limit=100):
        """
        Real-time search for knowledge articles - no syncing needed
        """
        if not self.sf:
            self.authenticate()

        try:
            # Query with Product Documentation record type filter
            query = f"""
            SELECT Id, Title, Article_Body__c,
                   CreatedDate, LastModifiedDate, PublishStatus, Language
            FROM Knowledge__kav 
            WHERE PublishStatus = 'Online'
            AND Language = 'en_US'
            AND RecordType.Name = 'Product Documentation'
            ORDER BY LastModifiedDate DESC
            LIMIT {limit}
            """

            self.logger.info("Querying knowledge articles with Article_Body__c field...")
            result = self.sf.query(query)
            articles = []

            for record in result['records']:
                article_body = record.get('Article_Body__c', '')
                
                # Log for debugging
                self.logger.info(f"Article '{record['Title']}': Body length = {len(article_body)}")
                
                article = {
                    'id': record['Id'],
                    'title': record['Title'],
                    'content': article_body or f"No content available for {record['Title']}",
                    'created_date': record['CreatedDate'],
                    'modified_date': record['LastModifiedDate'],
                    'publish_status': record['PublishStatus'],
                    'language': record['Language'],
                    'source': 'Salesforce Knowledge Article',
                    'source_url': f"https://yourinstance.salesforce.com/{record['Id']}"
                }
                articles.append(article)

            self.logger.info(f"Retrieved {len(articles)} knowledge articles with content")
            return articles
        
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge articles: {e}")
            return []
        
    def get_case_solutions(self, limit=200):
        """
        Real-world analogy: Like asking experienced doctors about their most successful treatments
        """
        if not self.sf:
            self.authenticate()

        try:
            # Query closed cases with solutions
            # Start with basic fields that should exist in all orgs
            query = f"""
            SELECT Id, CaseNumber, Subject, Description, 
               Status, Priority, CreatedDate, ClosedDate
            FROM Case 
            WHERE Status = 'Closed' 
            AND Description != null 
            AND Description != ''
            ORDER BY ClosedDate DESC
            LIMIT {limit}
            """

            result = self.sf.query(query)
            cases = []

            for record in result['records']:
                case = {
                    'id': record['Id'],
                    'case_number': record['CaseNumber'],
                    'title': record['Subject'],
                    'problem': record.get('Description', ''),
                    'solution': record.get('Description', ''),  # Use description as solution for now
                    'status': record['Status'],
                    'priority': record.get('Priority', 'Medium'),
                    'created_date': record['CreatedDate'],
                    'closed_date': record.get('ClosedDate', ''),
                    'product': record.get('Product', ''),
                    'category': record.get('Type', ''),  # Use Type instead of Category__c
                    'source': 'Salesforce Case',
                    'source_url': f"https://yourinstance.salesforce.com/{record['Id']}"
                }
                cases.append(case)

            self.logger.info(f"Retrieved {len(cases)} case solutions")
            return cases
        
        except Exception as e:
            self.logger.error(f"Error retrieving cases: {e}")
            # Log more specific error information
            self.logger.info("This might be due to missing custom fields or insufficient permissions")
            return []

    def process_for_rag(self, articles: List[Dict], cases: List[Dict]) -> List[Dict]:
        """
        Process Salesforce data for RAG system
        Real-world analogy: Like organizing research papers into a 
        consistent format for easy reference
        """
        processed_documents = []
        
        # Process Knowledge Articles
        for article in articles:
            # Combine title, summary, and content in a natural way
            full_content = f"{article['title']}\n\n{article['summary']}\n\n{article['content']}"
            
            doc = {
                'content': full_content.strip(),
                'metadata': {
                    'source': article['source'],
                    'source_id': article['id'],
                    'title': article['title'],
                    'type': 'knowledge_article',
                    'created_date': article['created_date'],
                    'modified_date': article['modified_date'],
                    'article_type': article['article_type'],
                    'source_url': article['source_url']
                }
            }
            processed_documents.append(doc)
        
        # Process Case Solutions
        for case in cases:
            # Format as natural Q&A without structured headers
            problem_text = case['problem'] if case['problem'] else "No problem description available"
            solution_text = case['solution'] if case['solution'] else "No solution available"
            
            full_content = f"{case['title']}\n\nProblem: {problem_text}\n\nSolution: {solution_text}"
            
            # Add context if available
            context_parts = []
            if case['product']:
                context_parts.append(f"Product: {case['product']}")
            if case['category']:
                context_parts.append(f"Category: {case['category']}")
            if case['priority']:
                context_parts.append(f"Priority: {case['priority']}")
            
            if context_parts:
                full_content += "\n\n" + ", ".join(context_parts)
            
            doc = {
                'content': full_content.strip(),
                'metadata': {
                    'source': case['source'],
                    'source_id': case['id'],
                    'title': case['title'],
                    'type': 'case_solution',
                    'created_date': case['created_date'],
                    'closed_date': case['closed_date'],
                    'priority': case['priority'],
                    'product': case['product'],
                    'category': case['category'],
                    'source_url': case['source_url']
                }
            }
            processed_documents.append(doc)
        
        return processed_documents
        
    def search_knowledge_realtime(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Real-time search of Salesforce knowledge articles
        This is called during user queries for up-to-date results
        """
        if not self.sf:
            if not self.authenticate():
                return []
        
        # Since Article_Body__c can't be filtered, we'll search only on Title
        # and then retrieve the full content for matching articles
        try:
            self.logger.info(f"Searching Salesforce knowledge for: {query}")
            
            # Split query into words for better matching
            query_words = [word.lower() for word in query.split() if len(word) > 2]
            
            # Build SOQL query with LIKE conditions ONLY on Title
            like_conditions = []
            for word in query_words:
                # Escape single quotes and special characters for SOQL
                escaped_word = word.replace("'", "\\'")
                escaped_word = escaped_word.replace("?", "")
                escaped_word = escaped_word.replace("!", "")
                like_conditions.append(f"Title LIKE '%{escaped_word}%'")
            
            if not like_conditions:
                # Fallback for short queries or single words
                escaped_query = query.replace("'", "\\'")
                escaped_query = escaped_query.replace("?", "")
                escaped_query = escaped_query.replace("!", "")
                like_conditions.append(f"Title LIKE '%{escaped_query}%'")
            
            where_clause = " OR ".join(like_conditions)
            
            soql_query = f"""
            SELECT Id, Title, Article_Body__c
            FROM Knowledge__kav 
            WHERE PublishStatus = 'Online'
            AND Language = 'en_US'
            AND RecordType.Name = 'Product Documentation'
            AND ({where_clause})
            ORDER BY LastModifiedDate DESC
            LIMIT {limit}
            """
            
            self.logger.info(f"Executing SOQL query: {soql_query}")
            result = self.sf.query(soql_query)
            articles = []
            
            import re
            
            for record in result['records']:
                article_body = record.get('Article_Body__c', '')
                
                # Clean HTML content for better text matching
                clean_title = re.sub(r'<[^>]+>', '', record['Title'])
                clean_content = re.sub(r'<[^>]+>', ' ', article_body)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                
                # Since we can only search on title, we'll check content after retrieval
                title_lower = clean_title.lower()
                content_lower = clean_content.lower()
                query_lower = query.lower()
                
                # Count matches in title and content
                title_matches = sum(1 for word in query_words if word in title_lower)
                content_matches = sum(1 for word in query_words if word in content_lower)
                
                # Also check for exact phrase matches
                if query_lower in title_lower:
                    title_matches += 2
                if query_lower in content_lower:
                    content_matches += 1
                
                # Calculate relevance score
                total_words = max(1, len(query_words))
                relevance_score = min(1.0, (title_matches * 0.6 + content_matches * 0.4) / total_words)
                
                # Include the article (we already filtered by title, so include all)
                article = {
                    'id': record['Id'],
                    'title': record['Title'],
                    'content': article_body,
                    'clean_content': clean_content,
                    'source': 'Salesforce Knowledge Article',
                    'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                    'relevance_score': max(0.3, relevance_score),  # Minimum score since title matched
                    'type': 'salesforce_knowledge',
                    'title_matches': title_matches,
                    'content_matches': content_matches
                }
                articles.append(article)
                self.logger.info(f"Found match: {record['Title']} (score: {relevance_score:.2f}, title_matches: {title_matches}, content_matches: {content_matches})")
            
            # Sort by relevance score
            articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            self.logger.info(f"SOQL search found {len(articles)} articles")
            
            # If no results from title search, get all articles and search in memory
            if not articles:
                self.logger.info("No title matches, retrieving all articles for content search...")
                
                all_query = f"""
                SELECT Id, Title, Article_Body__c
                FROM Knowledge__kav 
                WHERE PublishStatus = 'Online'
                AND Language = 'en_US'
                AND RecordType.Name = 'Product Documentation'
                ORDER BY LastModifiedDate DESC
                LIMIT {limit * 3}
                """
                
                all_result = self.sf.query(all_query)
                for record in all_result['records']:
                    article_body = record.get('Article_Body__c', '')
                    clean_content = re.sub(r'<[^>]+>', ' ', article_body)
                    clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                    
                    # Check if any query words appear in the content
                    content_lower = clean_content.lower()
                    query_lower = query.lower()
                    
                    content_matches = sum(1 for word in query_words if word in content_lower)
                    if query_lower in content_lower:
                        content_matches += 2
                    
                    # Only include if there are content matches
                    if content_matches > 0:
                        relevance_score = min(1.0, content_matches * 0.4 / len(query_words))
                        
                        article = {
                            'id': record['Id'],
                            'title': record['Title'],
                            'content': article_body,
                            'clean_content': clean_content,
                            'source': 'Salesforce Knowledge Article',
                            'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                            'relevance_score': relevance_score,
                            'type': 'salesforce_knowledge',
                            'search_method': 'content_search'
                        }
                        articles.append(article)
                        self.logger.info(f"Content match: {record['Title']} (score: {relevance_score:.2f})")
                
                # Limit and sort results
                articles = sorted(articles, key=lambda x: x['relevance_score'], reverse=True)[:limit]
                self.logger.info(f"Content search found {len(articles)} articles")
            
            return articles
            
        except Exception as e:
            self.logger.error(f"SOQL search failed: {e}")
            
            # Final fallback: just return all articles
            try:
                self.logger.info("All searches failed, returning all articles...")
                
                fallback_query = f"""
                SELECT Id, Title, Article_Body__c
                FROM Knowledge__kav 
                WHERE PublishStatus = 'Online'
                AND Language = 'en_US'
                AND RecordType.Name = 'Product Documentation'
                ORDER BY LastModifiedDate DESC
                LIMIT {limit}
                """
                
                fallback_result = self.sf.query(fallback_query)
                articles = []
                
                for record in fallback_result['records']:
                    article_body = record.get('Article_Body__c', '')
                    
                    article = {
                        'id': record['Id'],
                        'title': record['Title'],
                        'content': article_body,
                        'source': 'Salesforce Knowledge Article',
                        'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                        'relevance_score': 0.2,
                        'type': 'salesforce_knowledge',
                        'search_method': 'fallback_all'
                    }
                    articles.append(article)
                
                self.logger.info(f"Fallback returned {len(articles)} articles")
                return articles
                
            except Exception as final_e:
                self.logger.error(f"Final fallback also failed: {final_e}")
                return []
        
    def get_updated_content(self, last_sync_date: str) -> tuple:
        """
        Get content updated since last sync
        Real-world analogy: Like checking only for new emails instead of
        re-reading your entire inbox
        """
        if not self.sf:
            self.authenticate()
            
        # Query for articles updated since last sync
        articles_query = f"""
        SELECT Id, Title, Summary, ArticleBody, LastModifiedDate
        FROM Knowledge__kav 
        WHERE PublishStatus = 'Online' 
        AND LastModifiedDate > {last_sync_date}
        ORDER BY LastModifiedDate DESC
        """
        
        # Query for cases updated since last sync  
        cases_query = f"""
        SELECT Id, CaseNumber, Subject, Description, ClosedDate
        FROM Case 
        WHERE Status = 'Closed' 
        AND Description != null 
        AND ClosedDate > {last_sync_date}
        ORDER BY ClosedDate DESC
        """

        try:
            updated_articles = self.sf.query(articles_query)
            updated_cases = self.sf.query(cases_query)
            return updated_articles, updated_cases
        except Exception as e:
            self.logger.error(f"Error getting updated content: {e}")
            return [], []
        
    def get_all_knowledge_for_rag(self) -> List[Dict]:
        """
        Get all Salesforce Knowledge formmatted for your RAG system
        This is the main method to call for integration
        """
        articles = self.get_knowledge_articles()
        cases = self.get_case_solutions()
        return self.process_for_rag(articles, cases)
    
    # 🎯 NEW INTENT-DRIVEN SEARCH ARCHITECTURE
    # Clean, testable, and honest about what it can/cannot find
    
    def extract_user_intent(self, query: str) -> Dict[str, any]:
        """
        Extract user intent from query: action, service type, and context.
        Enhanced to handle complex scenarios like late arrivals, escalations, etc.
        """
        query_lower = query.lower()
        
        # Enhanced action patterns with scenario-specific mappings
        action_patterns = {
            'handle': [
                # Late arrival scenarios
                'arrives late', 'arrive late', 'very late', 'comes late', 'late arrival',
                'late for', 'delayed arrival', 'after hours',
                # General handling
                'handle', 'handling', 'manage', 'process', 'deal with', 'address',
                'what if', 'what to do', 'how to handle', 'what happens',
                # Problem scenarios  
                'no-show', 'no show', 'noshow', 'missed', "didn't show",
                'complaint', 'complain', 'unhappy', 'angry', 'upset',
                'problem', 'issue', 'trouble', 'escalate', 'escalation'
            ],
            'cancel': ['cancel', 'cancellation', 'cancelled', 'cancelling', 'void', 'refund'],
            'modify': ['modify', 'modification', 'change', 'update', 'amend', 'alter', 'edit'],
            'book': ['book', 'booking', 'reserve', 'reservation', 'create', 'new', 'make']
        }
        
        # Define service type keywords (unchanged)
        service_patterns = {
            'air': ['air', 'flight', 'airline', 'aviation', 'plane'],
            'hotel': ['hotel', 'accommodation', 'room', 'stay', 'lodging', 'guest'],
            'car': ['car', 'rental', 'vehicle', 'auto', 'automobile']
        }
        
        # Define context keywords (enhanced)
        context_patterns = {
            'booking': ['booking', 'reservation', 'ticket', 'itinerary'],
            'no-show': ['no-show', 'no show', 'noshow', 'missed', "didn't show"],
            'late-arrival': ['late', 'delayed', 'after hours', 'very late'],
            'escalation': ['escalate', 'escalation', 'supervisor', 'manager'],
            'waiver': ['waiver', 'waive', 'fee waiver', 'refund', 'exception']
        }
        
        # Enhanced action detection with scenario priority
        detected_action = None
        
        # Priority 1: Check for specific scenarios first (more specific patterns)
        scenario_actions = {
            'handle': ['arrives late', 'arrive late', 'very late', 'what if', 'no-show', 
                      'complaint', 'escalate', 'problem', 'issue', 'angry', 'unhappy']
        }
        
        for action, scenarios in scenario_actions.items():
            if any(scenario in query_lower for scenario in scenarios):
                detected_action = action
                break
        
        # Priority 2: Check for direct action words if no scenario detected
        if not detected_action:
            for action, keywords in action_patterns.items():
                if any(keyword in query_lower for keyword in keywords):
                    detected_action = action
                    break
        
        # Extract service type (unchanged logic)
        detected_service = None
        for service, keywords in service_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_service = service
                break
        
        # Extract context (enhanced)
        detected_context = []
        for context, keywords in context_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_context.append(context)
        
        intent = {
            'action': detected_action,
            'service': detected_service,
            'context': detected_context,
            'is_valid': detected_action is not None and detected_service is not None,
            'original_query': query,
            'confidence': self._calculate_intent_confidence(detected_action, detected_service, query_lower),
            'scenario_detected': True if detected_action == 'handle' and any(s in query_lower for s in scenario_actions['handle']) else False
        }
        
        self.logger.info(f"Enhanced intent extraction: {intent}")
        return intent
        
    def _calculate_intent_confidence(self, action: str, service: str, query_lower: str) -> float:
        """Calculate confidence in intent extraction."""
        if not action or not service:
            return 0.0
            
        confidence = 0.5  # Base confidence
        
        # Boost confidence if query contains specific action words
        action_words = {
            'cancel': ['cancel', 'cancellation', 'cancelled'],
            'modify': ['modify', 'change', 'update'],
            'book': ['book', 'booking', 'new'],
            'handle': ['handle', 'manage', 'process']
        }.get(action, [])
        
        if any(word in query_lower for word in action_words):
            confidence += 0.3
            
        # Boost confidence if query contains specific service words
        service_words = {
            'air': ['flight', 'airline', 'air'],
            'hotel': ['hotel', 'room', 'accommodation'],
            'car': ['car', 'rental', 'vehicle']
        }.get(service, [])
        
        if any(word in query_lower for word in service_words):
            confidence += 0.2
            
        return min(1.0, confidence)
        
    def validate_article_relevance(self, article: Dict, intent: Dict, min_score: float = 0.3) -> bool:
        """Phase 3: Validate that an article is actually relevant to the user's intent."""
        relevance = article.get('intent_match', {})
        
        # Check minimum relevance score
        if relevance.get('score', 0.0) < min_score:
            self.logger.info(f"Article '{article['title']}' rejected: score {relevance.get('score', 0.0):.2f} < {min_score}")
            return False
            
        # Ensure it has both action and service relevance
        if not relevance.get('has_both', False):
            self.logger.info(f"Article '{article['title']}' rejected: missing both action and service relevance")
            return False
            
        # Check for topic mismatch (e.g., query about air bookings but article about air cancellations)
        action_mismatch = self._check_action_mismatch(article, intent)
        if action_mismatch:
            self.logger.info(f"Article '{article['title']}' rejected: action mismatch")
            return False
            
        return True
        
    def _check_action_mismatch(self, article: Dict, intent: Dict) -> bool:
        """Check if article is about a completely different action."""
        title_lower = article['title'].lower()
        user_action = intent['action']
        
        # Define conflicting actions
        action_conflicts = {
            'book': ['cancel', 'cancellation'],
            'cancel': ['book', 'booking', 'new'],
            'modify': [],  # Modify is generally compatible
            'handle': []   # Handle is generally compatible
        }
        
        conflicts = action_conflicts.get(user_action, [])
        return any(conflict in title_lower for conflict in conflicts)
        
    def compare_search_methods(self, query: str, limit: int = 5) -> Dict[str, any]:
        """🔍 TESTING METHOD: Compare old vs new search approaches for analysis."""
        self.logger.info(f"Comparing search methods for: {query}")
        
        # Test new intent-driven search
        start_time = time.time()
        intent_results = self.search_knowledge_with_intent(query, limit)
        intent_time = time.time() - start_time
        
        # Test old broad search (if needed for comparison)
        start_time = time.time()
        old_results = self.search_knowledge_realtime(query, limit)
        old_time = time.time() - start_time
        
        # Analyze results
        comparison = {
            'query': query,
            'intent_driven': {
                'results_count': len(intent_results),
                'search_time': intent_time,
                'results': intent_results,
                'avg_relevance': sum(r['relevance_score'] for r in intent_results) / max(1, len(intent_results))
            },
            'old_method': {
                'results_count': len(old_results),
                'search_time': old_time,
                'results': old_results,
                'avg_relevance': sum(r['relevance_score'] for r in old_results) / max(1, len(old_results))
            },
            'analysis': self._analyze_search_comparison(intent_results, old_results)
        }
        
        self.logger.info(f"Search comparison complete: Intent({len(intent_results)}) vs Old({len(old_results)})")
        return comparison
        
    def _analyze_search_comparison(self, intent_results: List[Dict], old_results: List[Dict]) -> Dict[str, any]:
        """Analyze the differences between search methods."""
        
        # Calculate metrics
        intent_avg = sum(r['relevance_score'] for r in intent_results) / max(1, len(intent_results))
        old_avg = sum(r['relevance_score'] for r in old_results) / max(1, len(old_results))
        
        # Check for quality differences
        intent_high_quality = sum(1 for r in intent_results if r['relevance_score'] > 0.7)
        old_high_quality = sum(1 for r in old_results if r['relevance_score'] > 0.7)
        
        intent_low_quality = sum(1 for r in intent_results if r['relevance_score'] < 0.3)
        old_low_quality = sum(1 for r in old_results if r['relevance_score'] < 0.3)
        
        return {
            'intent_better': intent_avg > old_avg,
            'relevance_improvement': intent_avg - old_avg,
            'intent_high_quality_count': intent_high_quality,
            'old_high_quality_count': old_high_quality,
            'intent_low_quality_count': intent_low_quality,
            'old_low_quality_count': old_low_quality,
            'recommendation': 'Use intent-driven search' if intent_avg > old_avg or intent_high_quality > old_high_quality else 'Old method performed better'
        }
        
    def search_knowledge_with_intent(self, query: str, limit: int = 5) -> List[Dict]:
        """
        NEW: Intent-driven search that prioritizes both action and service type.
        This is the main method that should replace search_knowledge_realtime.
        """
        if not self.sf:
            if not self.authenticate():
                return []
        
        self.logger.info(f"🎯 Intent-driven search for: {query}")
        
        # Phase 1: Extract user intent
        intent = self.extract_user_intent(query)
        
        if not intent['is_valid']:
            self.logger.warning(f"Could not extract valid intent from query: {query}")
            self.logger.warning(f"Detected - action: {intent['action']}, service: {intent['service']}")
            # Return empty results with clear reasoning
            return []
            
        # Log intent for debugging
        self.logger.info(f"Valid intent detected: {intent['action']} + {intent['service']} (confidence: {intent['confidence']:.2f})")
        
        # Phase 2: Search for articles matching the intent
        try:
            action = intent['action']
            service = intent['service']
            
            # Build search terms with enhanced handling for scenarios
            action_terms = {
                'cancel': ['cancel', 'cancellation'],
                'modify': ['modification', 'modify', 'change'],
                'book': ['booking', 'book', 'new'],
                'handle': ['handling', 'handle', 'manage', 'procedures', 'process', 'guidelines']
            }.get(action, [action])
            
            service_terms = {
                'air': ['air', 'flight'],
                'hotel': ['hotel', 'guest'],
                'car': ['car']
            }.get(service, [service])
            
            # Enhanced search for handle actions - include scenario-specific terms
            if action == 'handle':
                # Add scenario-specific search terms based on context
                scenario_terms = []
                if 'late' in query.lower() or 'arrival' in query.lower():
                    scenario_terms.extend(['late', 'arrival', 'check-in', 'after hours'])
                if 'no-show' in query.lower() or 'missed' in query.lower():
                    scenario_terms.extend(['no-show', 'missed', 'noshow'])
                if 'escalate' in query.lower():
                    scenario_terms.extend(['escalation', 'supervisor', 'manager'])
                if 'complaint' in query.lower() or 'angry' in query.lower():
                    scenario_terms.extend(['complaint', 'customer service', 'resolution'])
                
                action_terms.extend(scenario_terms)
            
            # Strategy 1: Look for articles with BOTH action and service in title
            self.logger.info(f"Searching for articles with both '{action}' and '{service}' relevance")
            
            primary_conditions = []
            for action_term in action_terms:
                for service_term in service_terms:
                    primary_conditions.append(f"(Title LIKE '%{action_term}%' AND Title LIKE '%{service_term}%')")
            
            if primary_conditions:
                primary_query = f"""
                SELECT Id, Title, Article_Body__c
                FROM Knowledge__kav 
                WHERE PublishStatus = 'Online'
                AND Language = 'en_US'
                AND RecordType.Name = 'Product Documentation'
                AND ({' OR '.join(primary_conditions)})
                ORDER BY LastModifiedDate DESC
                LIMIT {limit}
                """
                
                self.logger.info(f"Executing precise search: {primary_query}")
                result = self.sf.query(primary_query)
                
                if result['records']:
                    articles = self._process_intent_results(result['records'], intent)
                    if articles:
                        self.logger.info(f"✅ Found {len(articles)} articles with both action and service")
                        return articles
            
            # Strategy 2: Look for articles with action in title, filter by service
            self.logger.info(f"Fallback: Searching for articles with '{action}' in title")
            
            action_conditions = [f"Title LIKE '%{term}%'" for term in action_terms]
            
            fallback_query = f"""
            SELECT Id, Title, Article_Body__c
            FROM Knowledge__kav 
            WHERE PublishStatus = 'Online'
            AND Language = 'en_US'
            AND RecordType.Name = 'Product Documentation'
            AND ({' OR '.join(action_conditions)})
            ORDER BY LastModifiedDate DESC
            LIMIT {limit * 2}
            """
            
            self.logger.info(f"Executing fallback search: {fallback_query}")
            result = self.sf.query(fallback_query)
            
            if result['records']:
                # Filter for service relevance
                relevant_articles = []
                for record in result['records']:
                    title_lower = record['Title'].lower()
                    content_lower = (record.get('Article_Body__c', '') or '').lower()
                    
                    if any(term in title_lower or term in content_lower for term in service_terms):
                        relevant_articles.append(record)
                
                if relevant_articles:
                    articles = self._process_intent_results(relevant_articles[:limit], intent)
                    self.logger.info(f"⚠️ Found {len(articles)} articles with action + some service relevance")
                    return articles
            
            # Phase 3: Honest failure - no relevant articles found
            self.logger.info(f"❌ No articles found matching intent: {action} + {service}")
            self.logger.info(f"Intent-driven search was honest: no relevant content available")
            return []
            
        except Exception as e:
            self.logger.error(f"Intent-driven search failed: {e}")
            # Don't fallback to old search - be honest about failure
            return []
            
    def _process_intent_results(self, records: List[Dict], intent: Dict[str, any]) -> List[Dict]:
        """
        Process search results for intent-driven search.
        """
        import re
        import html
        
        articles = []
        
        for record in records:
            article_body = record.get('Article_Body__c', '')
            
            # Clean content
            clean_content = re.sub(r'<[^>]+>', ' ', article_body)
            clean_content = html.unescape(clean_content)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            # Calculate intent relevance
            relevance = self._calculate_intent_relevance(record, intent)
            
            article = {
                'id': record['Id'],
                'title': record['Title'],
                'content': article_body,
                'clean_content': clean_content,
                'source': 'Salesforce Knowledge Article',
                'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                'relevance_score': relevance['score'],
                'type': 'salesforce_knowledge',
                'intent_match': relevance,
                'search_method': 'intent_driven'
            }
            articles.append(article)
            
            self.logger.info(f"Article: {record['Title']} - Relevance: {relevance['score']:.2f} - Intent match: {relevance}")
            
        # Phase 3: Validate relevance before returning (enhanced)
        validated_articles = []
        for article in articles:
            if self.validate_article_relevance(article, intent):
                validated_articles.append(article)
                self.logger.info(f"✅ Validated: {article['title']} (score: {article['relevance_score']:.2f})")
            else:
                self.logger.info(f"❌ Filtered out: {article['title']} (score: {article['relevance_score']:.2f}) - Low relevance")
        
        # Enhanced sorting with scenario consideration
        validated_articles.sort(key=lambda x: (
            x['relevance_score'],
            len(x.get('intent_match', {}).get('context_matches', [])),
            x.get('intent_match', {}).get('scenario_detected', False)
        ), reverse=True)
        
        if not validated_articles:
            self.logger.info("⚠️ All articles filtered out due to low relevance")
        else:
            self.logger.info(f"✅ Returning {len(validated_articles)} validated articles")
            
        return validated_articles
    
    def _calculate_intent_relevance(self, record: Dict, intent: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate how well an article matches the user's intent.
        Enhanced for complex scenarios like late arrivals, escalations, etc.
        """
        title_lower = record['Title'].lower()
        content_lower = (record.get('Article_Body__c', '') or '').lower()
        
        action = intent['action']
        service = intent['service']
        context = intent.get('context', [])
        scenario_detected = intent.get('scenario_detected', False)
        
        # Enhanced action terms with scenario-specific mappings
        action_terms = {
            'cancel': ['cancel', 'cancellation', 'refund', 'void'],
            'modify': ['modification', 'modify', 'change', 'update', 'amend'],
            'book': ['booking', 'book', 'new', 'reservation', 'create'],
            'handle': ['handling', 'handle', 'manage', 'process', 'deal with']
        }.get(action, [action])
        
        # Add scenario-specific terms for handle actions
        if action == 'handle' and scenario_detected:
            query_lower = intent.get('original_query', '').lower()
            if any(term in query_lower for term in ['late', 'delayed']):
                action_terms.extend(['late arrival', 'delayed', 'after hours', 'grace period'])
            elif any(term in query_lower for term in ['no-show', 'missed']):
                action_terms.extend(['no-show', 'missed', 'absent', 'failure to appear'])
            elif any(term in query_lower for term in ['escalate', 'complaint']):
                action_terms.extend(['escalation', 'complaint', 'supervisor', 'manager'])
        
        service_terms = {
            'air': ['air', 'flight', 'airline', 'aviation'],
            'hotel': ['hotel', 'accommodation', 'room', 'lodging'],
            'car': ['car', 'rental', 'vehicle', 'auto']
        }.get(service, [service])
        
        # Check matches with enhanced logic
        action_in_title = any(term in title_lower for term in action_terms)
        action_in_content = any(term in content_lower for term in action_terms)
        service_in_title = any(term in title_lower for term in service_terms)
        service_in_content = any(term in content_lower for term in service_terms)
        
        # Check context matches
        context_matches = []
        for ctx in context:
            ctx_terms = {
                'no-show': ['no-show', 'missed', 'absent'],
                'late-arrival': ['late', 'delayed', 'tardy'],
                'escalation': ['escalate', 'supervisor', 'manager'],
                'waiver': ['waiver', 'waive', 'exception']
            }.get(ctx, [])
            
            if any(term in title_lower or term in content_lower for term in ctx_terms):
                context_matches.append(ctx)
        
        # Enhanced scoring with context consideration
        score = 0.0
        
        # Base scoring
        if action_in_title and service_in_title:
            score = 1.0  # Perfect match
        elif action_in_title and service_in_content:
            score = 0.8  # Good match
        elif action_in_content and service_in_title:
            score = 0.7  # Good match
        elif action_in_content and service_in_content:
            score = 0.5  # Moderate match
        elif action_in_title:
            score = 0.4  # Action match only
        elif service_in_title:
            score = 0.3  # Service match only
        else:
            score = 0.1  # Minimal relevance
        
        # Boost score for context matches (scenario-specific)
        if context_matches:
            context_boost = len(context_matches) * 0.1
            score = min(1.0, score + context_boost)
        
        # Extra boost for scenario detection
        if scenario_detected and score > 0.3:
            score = min(1.0, score + 0.1)
        
        return {
            'score': score,
            'action_in_title': action_in_title,
            'action_in_content': action_in_content,
            'service_in_title': service_in_title,
            'service_in_content': service_in_content,
            'context_matches': context_matches,
            'scenario_detected': scenario_detected,
            'has_both': (action_in_title or action_in_content) and (service_in_title or service_in_content)
        }
    
    def validate_article_relevance(self, article: Dict, intent: Dict[str, any]) -> bool:
        """
        Validate that an article is truly relevant to the user's intent.
        Enhanced to filter out irrelevant results more effectively.
        """
        relevance_score = article.get('relevance_score', 0.0)
        intent_match = article.get('intent_match', {})
        
        # Minimum relevance threshold
        min_score = 0.3
        
        # For complex scenarios, be more selective
        if intent.get('scenario_detected', False):
            min_score = 0.4
        
        # Must have minimum score
        if relevance_score < min_score:
            return False
        
        # Must have both action and service relevance for basic validation
        has_both = intent_match.get('has_both', False)
        if not has_both and relevance_score < 0.5:
            return False
        
        # Additional validation for specific scenarios
        action = intent.get('action')
        context = intent.get('context', [])
        
        # For handle actions with specific contexts, ensure context alignment
        if action == 'handle' and context:
            context_matches = intent_match.get('context_matches', [])
            if not context_matches and relevance_score < 0.6:
                return False
        
        return True
        
    # 🧪 TESTING AND DEMONSTRATION METHODS
    
    def test_intent_extraction(self, test_queries: List[str] = None) -> Dict[str, any]:
        """Test intent extraction on various queries for demonstration."""
        if test_queries is None:
            test_queries = [
                "How to cancel a hotel booking?",
                "Modify air reservation", 
                "Book new car rental",
                "Handle no-show flight",
                "Customer arrives late for hotel check-in",  # Enhanced scenario
                "What to do when flight passenger no-shows?",  # Enhanced scenario
                "How to escalate angry car rental customer?",  # Enhanced scenario
                "What is the weather today?",  # Should fail
                "Air cancellation policy",
                "Hotel modification process"
            ]
        
        results = {
            'test_queries': [],
            'successful_extractions': 0,
            'failed_extractions': 0
        }
        
        for query in test_queries:
            intent = self.extract_user_intent(query)
            result = {
                'query': query,
                'intent': intent,
                'success': intent['is_valid'],
                'confidence': intent.get('confidence', 0.0)
            }
            results['test_queries'].append(result)
            
            if intent['is_valid']:
                results['successful_extractions'] += 1
            else:
                results['failed_extractions'] += 1
        
        self.logger.info(f"Intent extraction test: {results['successful_extractions']}/{len(test_queries)} successful")
        return results
    
    def demonstrate_improvements(self, demo_queries: List[str] = None) -> Dict[str, any]:
        """Demonstrate the improvements of the new search architecture."""
        if demo_queries is None:
            demo_queries = [
                "How to cancel a hotel booking?",
                "Modify air reservation",
                "Handle car rental no-show"
            ]
        
        demonstrations = []
        
        for query in demo_queries:
            demo = {
                'query': query,
                'intent_analysis': self.extract_user_intent(query),
                'search_comparison': self.compare_search_methods(query, 3)
            }
            demonstrations.append(demo)
        
        return {
            'demonstrations': demonstrations,
            'summary': 'Intent-driven search provides more relevant, targeted results'
        }
