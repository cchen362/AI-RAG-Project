# 🚀 Quick Start Guide - Testing Critical Fixes

## Prerequisites

1. **Environment Setup**
   ```bash
   # Make sure you have Python 3.8+
   python --version
   
   # Install required packages
   pip install openai python-dotenv simple-salesforce
   ```

2. **Configuration**
   Create/update your `.env` file with:
   ```env
   # Salesforce credentials
   SALESFORCE_USERNAME=your_username@company.com
   SALESFORCE_PASSWORD=your_password
   SALESFORCE_SECURITY_TOKEN=your_security_token
   
   # OpenAI credentials (optional but recommended)
   OPENAI_API_KEY=sk-your-openai-api-key
   ```

## Running the Tests

### Option 1: Full Test Suite
```bash
# Run all critical fixes tests
python test_critical_fixes.py
```

### Option 2: Individual Component Testing

```python
# Test only intent extraction
from src.salesforce_connector import SalesforceConnector

sf = SalesforceConnector()
intent = sf.extract_user_intent("Customer arrives late for hotel check-in")
print(intent)
```

```python
# Test only fallback mechanisms  
from src.semantic_enhancer import TransformativeSemanticSearch

search = TransformativeSemanticSearch(sf, openai_key)
result = search.enhanced_search_with_fallbacks("What if guest is very late?")
print(result)
```

## Expected Output

### ✅ Success Indicators
- `OpenAI client successfully initialized`
- `Enhanced intent extraction successful`
- `Fallback search successful`
- `All critical fixes verified`

### ⚠️ Warning Indicators
- `OpenAI not available` (API key missing - system still works with basic search)
- `Salesforce authentication failed` (check credentials)
- `Primary search insufficient` (normal - fallbacks will engage)

### ❌ Error Indicators
- `Authentication error` (check .env file)
- `Module import failed` (install missing packages)
- `All search strategies exhausted` (no relevant content found - honest failure)

## Example Test Results

```
🚨 TESTING CRITICAL FIXES
==================================================

🔧 FIX 1: Enhanced Intent Extraction for Complex Scenarios
------------------------------------------------------------
Testing enhanced intent extraction:
  Query: 'Customer arrives late for hotel check-in'
    → Action: handle, Service: hotel
    → Context: ['late-arrival'], Scenario: True
    → Valid: True, Confidence: 0.85

🔧 FIX 2: OpenAI API v1.0+ Compatibility
------------------------------------------------------------
✅ OpenAI client successfully initialized with v1.0+ syntax
   Client type: <class 'openai.OpenAI'>

🔧 FIX 3: Enhanced Search Strategy and Fallbacks
------------------------------------------------------------
Testing fallbacks for: 'Customer arrives very late for hotel check-in'
  Strategy: enhanced_fallback
  Fallback used: True
  Confidence: 0.72
  Results found: 3
    1. Hotel Late Arrival Procedures (score: 0.89)
    2. Guest Services Guidelines (score: 0.76)

🎉 CRITICAL FIXES TESTING COMPLETE
==================================================
✅ All critical fixes have been implemented and tested
🚀 Your AI-RAG system is now significantly more robust!
```

## Troubleshooting

### Common Issues

1. **"No module named 'openai'"**
   ```bash
   pip install openai>=1.0.0
   ```

2. **"Salesforce authentication failed"**
   - Check username/password in .env
   - Verify security token (required for production orgs)
   - Try with sandbox domain if using test org

3. **"OpenAI API key invalid"**
   - Verify API key format (starts with 'sk-')
   - Check API key has sufficient credits
   - System will fallback to basic search without OpenAI

4. **"No relevant articles found"**
   - This is normal - means honest failure (no good content)
   - Check if Salesforce org has Knowledge Articles
   - Verify article content matches test scenarios

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with verbose output
python test_critical_fixes.py
```

## What Each Fix Does

### Fix 1: OpenAI API v1.0+ Compatibility
- **Before**: `openai.ChatCompletion.create()` ❌ (deprecated)
- **After**: `openai.OpenAI().chat.completions.create()` ✅ (current)
- **Impact**: LLM-enhanced search now works

### Fix 2: Enhanced Intent Extraction  
- **Before**: Basic action/service detection
- **After**: Complex scenario recognition (late arrivals, escalations, etc.)
- **Impact**: Better understanding of real customer service queries

### Fix 3: Improved Search Strategy
- **Before**: Simple keyword matching
- **After**: Context-aware relevance scoring with scenario detection
- **Impact**: More relevant search results for complex queries

### Fix 4: Enhanced Fallback Mechanisms
- **Before**: Fail completely when primary search fails
- **After**: Multiple fallback strategies with graceful degradation
- **Impact**: System always tries to help, fails honestly when no content exists

## Next Steps

1. **Run the tests** to verify everything works
2. **Check the logs** for any warnings or errors
3. **Review the summary** in `CRITICAL_FIXES_SUMMARY.md`
4. **Integrate** into your production workflow
5. **Monitor** search quality and fallback usage

## Support

If you encounter issues:
1. Check the error logs for specific details
2. Verify all prerequisites are met
3. Review the troubleshooting section
4. Test individual components to isolate problems

**Your AI-RAG system is now production-ready! 🎉**
