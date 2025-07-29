"""
Quick verification that the re-ranker fixes work for the problematic scenario.
"""

def calculate_fixed_score(original_score, rerank_score, source_type, bias_colpali, is_explicit_chart=False):
    """Calculate score with the new fixed formula."""
    
    # New minimum thresholds
    min_thresholds = {
        'text': 0.15,
        'colpali': 0.20,  # Reduced from 0.25
        'salesforce': 0.2
    }
    
    min_threshold = min_thresholds.get(source_type, 0.2)
    
    # New score combination logic
    if original_score < min_threshold:
        combined = (original_score * 0.4 + rerank_score * 0.6) * 0.3
    else:
        if source_type == 'colpali':
            # NEW: Weight original score more heavily for ColPali
            combined = original_score * 0.6 + rerank_score * 0.4
        else:
            # Standard combination for text
            combined = original_score * 0.3 + rerank_score * 0.7
    
    # Apply bias
    if bias_colpali and source_type == 'colpali':
        if is_explicit_chart:
            combined *= 2.5  # 150% boost
        else:
            combined *= 2.0  # 100% boost (increased from 1.5x)
    elif source_type == 'text' and original_score > 0.4:
        if not is_explicit_chart:
            combined *= 1.05
    
    return max(0.0, min(1.0, combined))

def main():
    print("VERIFYING RE-RANKER FIXES")
    print("="*50)
    
    # Test problematic scenario from logs
    query = "what's the retrieval time in a ColPali RAG pipeline?"
    text_original = 0.508
    colpali_original = 0.748
    
    # Assume poor rerank score for ColPali (the problem case)
    text_rerank = 0.35    # Decent rerank score for text
    colpali_rerank = 0.15  # Poor rerank score for ColPali
    
    # Check if performance query triggers bias
    query_lower = query.lower()
    bias_colpali = any(term in query_lower for term in ['time', 'performance', 'speed', 'latency', 'rate'])
    
    print(f"Query: '{query}'")
    print(f"Performance query bias: {bias_colpali}")
    print()
    
    # Calculate with OLD logic (from debug app)
    print("OLD LOGIC (problematic):")
    old_text = text_original * 0.3 + text_rerank * 0.7
    if text_original > 0.4:
        old_text *= 1.05
    
    old_colpali = (colpali_original * 0.3 + colpali_rerank * 0.7) * 1.5  # Old 1.5x bias
    
    print(f"  Text: {text_original:.3f} * 0.3 + {text_rerank:.3f} * 0.7 * 1.05 = {old_text:.3f}")
    print(f"  ColPali: ({colpali_original:.3f} * 0.3 + {colpali_rerank:.3f} * 0.7) * 1.5 = {old_colpali:.3f}")
    print(f"  Winner: {'Text' if old_text > old_colpali else 'ColPali'} (margin: {abs(old_text - old_colpali):.3f})")
    print()
    
    # Calculate with NEW logic (fixed)
    print("NEW LOGIC (fixed):")
    new_text = calculate_fixed_score(text_original, text_rerank, 'text', bias_colpali)
    new_colpali = calculate_fixed_score(colpali_original, colpali_rerank, 'colpali', bias_colpali)
    
    print(f"  Text: {text_original:.3f} * 0.3 + {text_rerank:.3f} * 0.7 * 1.05 = {new_text:.3f}")
    print(f"  ColPali: ({colpali_original:.3f} * 0.6 + {colpali_rerank:.3f} * 0.4) * 2.0 = {new_colpali:.3f}")
    print(f"  Winner: {'Text' if new_text > new_colpali else 'ColPali'} (margin: {abs(new_text - new_colpali):.3f})")
    print()
    
    if new_colpali > new_text:
        print("SUCCESS: ColPali now wins with the fixes!")
        print(f"   Improvement: {new_colpali - old_colpali:.3f} point increase for ColPali")
    else:
        print("Still needs more adjustment")
    
    print()
    print("APPLIED FIXES:")
    print("1. Reduced ColPali minimum threshold: 0.25 → 0.20")
    print("2. Changed ColPali scoring weights: 30% original + 70% rerank → 60% original + 40% rerank")
    print("3. Increased bias multipliers: 1.5x → 2.0x for performance queries")

if __name__ == "__main__":
    main()