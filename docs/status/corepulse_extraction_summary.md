# CorePulse-LLM Extraction Summary

## Key Insights from Actual Implementation

### 1. Core Technique: Attention Manipulation (Not "DataVoid")
The actual CorePulse-LLM implementation doesn't use the term "DataVoid" - instead they use:
- **Amplification**: Increase attention to specific concepts (up to 5x)
- **Suppression**: Reduce attention to distracting concepts (down to 0.1x)
- **Balanced Attention**: Combination of amplification and suppression

### 2. Implementation Architecture

```python
# Their approach:
injector = LLMAttentionInjector(model, tokenizer)
injector.amplify_phrases(["golden gate"], amplification_factor=5.0)
injector.suppress_phrases(["Bay Bridge"], suppression_factor=0.1)
```

### 3. Key Parameters from Their Code
- `amplification_factor`: 3.0-5.0 (boost attention)
- `suppression_factor`: 0.1-0.2 (reduce attention)
- `interaction_type`: "amplify", "suppress", "redirect"

### 4. Zero-Entropy Principle Implementation
While not explicitly stated as "DataVoid", their technique follows the same principle:
1. Identify target phrases/concepts
2. Find token indices for those phrases
3. Modify attention weights during forward pass
4. Apply changes at specific transformer layers

### 5. Technical Approach
- Hooks into transformer attention mechanism
- Modifies attention scores before softmax
- Works at token level, not spatial
- Layer-specific targeting possible

## Mapping to Our MLX Implementation

### Current State (What We Have)
```python
# In corpus_mlx/attention.py
class DataVoidController:
    config = {
        "amplification_factor": 2.5,
        "void_threshold": 0.25,
        "product_weight": 0.85,
        "redistribution_rate": 0.7,
        "cross_attention_scale": 2.0
    }
```

### Their Actual Approach
```python
# CorePulse-LLM approach
class LLMAttentionInjector:
    # Amplify specific phrases (products)
    amplify_phrases(phrases, amplification_factor=5.0)
    # Suppress distractors (voids)
    suppress_phrases(phrases, suppression_factor=0.1)
```

## Key Differences

1. **Terminology**: They use "amplify/suppress" not "void/product"
2. **Factors**: They use more extreme values (5.0x amplification, 0.1x suppression)
3. **Focus**: Token-level manipulation, not spatial attention
4. **Architecture**: Hooks into transformer layers, not UNet blocks

## Next Steps for MLX Port

1. Rename concepts to match their actual implementation
2. Adjust amplification factors to their proven values
3. Implement token-level manipulation (not just position-based)
4. Add phrase-based targeting (not just position indices)

## Core Insight Confirmed

The zero-entropy principle holds true in their implementation:
- **"Attention is zero-sum"** - They redistribute attention from suppressed to amplified
- **"Take from hallucination, give to truth"** - Suppress distractors, amplify targets
- The technique works by manipulating attention weights directly