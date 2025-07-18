import os
from simple_salesforce import Salesforce
from simple_salesforce.exceptions import SalesforceAuthenticationFailed
from dotenv import load_dotenv
import logging
from typing import List, Dict, Optional
from datetime import datetime

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
                        self.logger.info(f"Intent extraction: {intent}")
        return intent
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
            
            # IMPROVED: Better query term processing to handle compound terms
            import re
            
            # Define domain-specific compound terms that should be kept together
            compound_terms = {
                'no-show': ['no-show', 'no show', 'noshow'],
                'check-in': ['check-in', 'check in', 'checkin'],
                'check-out': ['check-out', 'check out', 'checkout'],
                'follow-up': ['follow-up', 'follow up', 'followup'],
                'walk-in': ['walk-in', 'walk in', 'walkin']
            }
            
            # Extract compound terms first
            preserved_phrases = []
            cleaned_query = query.lower()
            
            for canonical_term, variations in compound_terms.items():
                for variation in variations:
                    if variation in cleaned_query:
                        preserved_phrases.append(canonical_term)
                        # Remove the variation from query to avoid double processing
                        cleaned_query = cleaned_query.replace(variation, '')
                        break
            
            # Define stop words to exclude from SOQL queries
            stop_words = {'how', 'to', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'for', 'of', 'with', 'by', 'my', 'your', 'their', 'what', 'when', 'where', 'why', 'should', 'would', 'could', 'can', 'will', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'get', 'got', 'make', 'made'}
            
            # Then split remaining words and filter out stop words and very short words
            remaining_words = [word.lower() for word in cleaned_query.split() 
                              if len(word) > 2 and word.lower() not in stop_words and word.isalpha()]
            
            # Combine preserved phrases with remaining words
            query_words = preserved_phrases + remaining_words
            
            # Enhanced logging
            self.logger.info(f"Query processing: original='{query}', preserved_phrases={preserved_phrases}, remaining_words={remaining_words}")
            
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
                
                # IMPROVED: More realistic scoring logic with better weighting
                # Define domain-specific terms that should get bonus points
                domain_terms = ['hotel', 'booking', 'reservation', 'cancellation', 'modification', 
                               'no-show', 'check-in', 'check-out', 'flight', 'car', 'rental']
                
                # Count matches in title and content
                title_matches = sum(1 for word in query_words if word in title_lower)
                content_matches = sum(1 for word in query_words if word in content_lower)
                
                # Bonus for domain-specific terms
                domain_bonus = 0
                for term in domain_terms:
                    if term in query_lower:
                        if term in title_lower:
                            domain_bonus += 1  # Reduced bonus
                        elif term in content_lower:
                            domain_bonus += 0.5  # Reduced bonus
                
                # Also check for exact phrase matches
                phrase_bonus = 0
                if query_lower in title_lower:
                    phrase_bonus += 3  # Reduced bonus
                elif query_lower in content_lower:
                    phrase_bonus += 1  # Reduced bonus
                
                # Calculate relevance score with more conservative weighting
                total_words = max(1, len(query_words))
                if total_words == 0:
                    relevance_score = 0.0
                else:
                    # Title matches weighted higher but not too high
                    base_score = (title_matches * 2.0 + content_matches * 1.0) / (total_words * 2.0)
                    
                    # Add bonuses but cap the total
                    bonus_score = (domain_bonus * 0.1) + (phrase_bonus * 0.1)
                    final_score = base_score + bonus_score
                    
                    # Normalize to 0-1 range with more realistic distribution
                    relevance_score = min(0.95, max(0.05, final_score))
                
                # Include the article with proper scoring
                article = {
                    'id': record['Id'],
                    'title': record['Title'],
                    'content': article_body,
                    'clean_content': clean_content,
                    'source': 'Salesforce Knowledge Article',
                    'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                    'relevance_score': relevance_score,  # Use actual calculated score
                    'type': 'salesforce_knowledge',
                    'title_matches': title_matches,
                    'content_matches': content_matches,
                    'domain_bonus': domain_bonus,
                    'phrase_bonus': phrase_bonus
                }
                articles.append(article)
                self.logger.info(f"Found match: {record['Title']} (score: {relevance_score:.2f}, title_matches: {title_matches}, content_matches: {content_matches}, domain_bonus: {domain_bonus}, phrase_bonus: {phrase_bonus})")
            
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
        
    def extract_user_intent(self, query: str) -> Dict[str, any]:
        """
        Extract user intent from query: action, service type, and context.
        Returns structured intent information.
        """
        query_lower = query.lower()
        
        # Define action keywords with variations
        action_patterns = {
            'cancel': ['cancel', 'cancellation', 'cancelled', 'cancelling', 'void', 'refund'],
            'modify': ['modify', 'modification', 'change', 'update', 'amend', 'alter', 'edit'],
            'book': ['book', 'booking', 'reserve', 'reservation', 'create', 'new', 'make'],
            'handle': ['handle', 'handling', 'manage', 'process', 'deal with', 'address']
        }
        
        # Define service type keywords
        service_patterns = {
            'air': ['air', 'flight', 'airline', 'aviation', 'plane'],
            'hotel': ['hotel', 'accommodation', 'room', 'stay', 'lodging'],
            'car': ['car', 'rental', 'vehicle', 'auto', 'automobile']
        }
        
        # Define context keywords
        context_patterns = {
            'booking': ['booking', 'reservation', 'ticket', 'itinerary'],
            'no-show': ['no-show', 'no show', 'noshow', 'missed', 'didn\'t show']
        }
        
        # Extract action
        detected_action = None
        for action, keywords in action_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_action = action
                break
        
        # Extract service type
        detected_service = None
        for service, keywords in service_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_service = service
                break
        
        # Extract context
        detected_context = []
        for context, keywords in context_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_context.append(context)
        
        intent = {
            'action': detected_action,
            'service': detected_service,
            'context': detected_context,
            'is_valid': detected_action is not None and detected_service is not None,
            'original_query': query
        }
        
        self.logger.info(f"Intent extraction: {intent}")
        return intent
        
    def search_by_intent(self, intent: Dict[str, any], limit: int = 5) -> List[Dict]:
        """
        Search for articles that match the user's intent (action + service).
        Uses precise SOQL queries instead of broad OR conditions.
        """
        if not intent['is_valid']:
            self.logger.warning("Invalid intent provided to search_by_intent")
            return []
        
        action = intent['action']
        service = intent['service']
        
        try:
            # Build precise search terms for action
            action_terms = {
                'cancel': ['cancel', 'cancellation'],
                'modify': ['modification', 'modify', 'change'],
                'book': ['booking', 'book', 'new'],
                'handle': ['handling', 'handle', 'manage']
            }.get(action, [action])
            
            # Build precise search terms for service
            service_terms = {
                'air': ['air', 'flight'],
                'hotel': ['hotel'],
                'car': ['car']
            }.get(service, [service])
            
            # Strategy 1: Look for articles with BOTH action and service in title
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
                
                self.logger.info(f"Precise search (both action+service): {primary_query}")
                result = self.sf.query(primary_query)
                
                if result['records']:
                    articles = self._process_search_results(result['records'], intent)
                    if articles:
                        self.logger.info(f"Found {len(articles)} articles with both action and service")
                        return articles
            
            # Strategy 2: Look for articles with action in title, service anywhere
            fallback_conditions = []
            for action_term in action_terms:
                fallback_conditions.append(f"Title LIKE '%{action_term}%'")
            
            fallback_query = f"""
            SELECT Id, Title, Article_Body__c
            FROM Knowledge__kav 
            WHERE PublishStatus = 'Online'
            AND Language = 'en_US'
            AND RecordType.Name = 'Product Documentation'
            AND ({' OR '.join(fallback_conditions)})
            ORDER BY LastModifiedDate DESC
            LIMIT {limit * 2}
            """
            
            self.logger.info(f"Fallback search (action in title): {fallback_query}")
            result = self.sf.query(fallback_query)
            
            if result['records']:
                # Filter results to include only those with service relevance
                filtered_articles = []
                for record in result['records']:
                    title_lower = record['Title'].lower()
                    content_lower = (record.get('Article_Body__c', '') or '').lower()
                    
                    # Check if service appears in title or content
                    if any(service_term in title_lower or service_term in content_lower 
                          for service_term in service_terms):
                        filtered_articles.append(record)
                
                if filtered_articles:
                    articles = self._process_search_results(filtered_articles[:limit], intent)
                    self.logger.info(f"Found {len(articles)} articles with action + service relevance")
                    return articles
            
            self.logger.info("No articles found matching intent")
            return []
            
    def _process_search_results(self, records: List[Dict], intent: Dict[str, any]) -> List[Dict]:
        """
        Process raw Salesforce records into structured article results.
        """
        import re
        articles = []
        
        for record in records:
            article_body = record.get('Article_Body__c', '')
            
            # Clean content
            clean_content = re.sub(r'<[^>]+>', ' ', article_body)
            clean_content = re.sub(r'\s+', ' ', clean_content).strip()
            
            article = {
                'id': record['Id'],
                'title': record['Title'],
                'content': article_body,
                'clean_content': clean_content,
                'source': 'Salesforce Knowledge Article',
                'source_url': f"https://yourinstance.salesforce.com/{record['Id']}",
                'type': 'salesforce_knowledge',
                'intent_match': self._calculate_intent_relevance(record, intent)
            }
            articles.append(article)
            
        return articles
    
    def _calculate_intent_relevance(self, record: Dict, intent: Dict[str, any]) -> Dict[str, any]:
        """
        Calculate how well an article matches the user's intent.
        """
        title_lower = record['Title'].lower()
        content_lower = (record.get('Article_Body__c', '') or '').lower()
        
        action = intent['action']
        service = intent['service']
        
        # Check action relevance
        action_terms = {
            'cancel': ['cancel', 'cancellation'],
            'modify': ['modification', 'modify', 'change'],
            'book': ['booking', 'book', 'new'],
            'handle': ['handling', 'handle', 'manage']
        }.get(action, [action])
        
        # Check service relevance
        service_terms = {
            'air': ['air', 'flight'],
            'hotel': ['hotel'],
            'car': ['car']
        }.get(service, [service])
        
        action_in_title = any(term in title_lower for term in action_terms)
        action_in_content = any(term in content_lower for term in action_terms)
        service_in_title = any(term in title_lower for term in service_terms)
        service_in_content = any(term in content_lower for term in service_terms)
        
        return {
            'action_in_title': action_in_title,
            'action_in_content': action_in_content,
            'service_in_title': service_in_title,
            'service_in_content': service_in_content,
            'title_relevance': action_in_title and service_in_title,
            'overall_relevance': (action_in_title or action_in_content) and (service_in_title or service_in_content)
        }

