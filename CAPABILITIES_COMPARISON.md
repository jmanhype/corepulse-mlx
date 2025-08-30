# corpus-mlx vs CorePulse: Capabilities Comparison

## Summary
After analyzing both implementations, here are the key differences and capabilities:

## Architecture Differences

### CorePulse (PyTorch/Diffusers)
- **Conditioning Replacement**: Replaces text embeddings BEFORE attention computation
- **Method**: Encodes injection prompts and swaps conditioning tensors at UNet blocks
- **Result**: Can achieve true semantic object replacement (apple→banana)

### corpus-mlx (MLX/Apple Silicon)
- **Attention Manipulation**: Modifies attention values (V) AFTER conditioning is processed
- **Method**: Manipulates pre-attention KV values during cross-attention
- **Result**: Can modify style/attributes but NOT replace objects

## Capabilities Comparison

| Feature | CorePulse | corpus-mlx | Notes |
|---------|-----------|------------|-------|
| **Semantic Object Replacement** | ✅ Yes | ❌ No | Requires conditioning replacement |
| **Style Transfer** | ✅ Yes | ✅ Yes | Both can inject style prompts |
| **Regional Control** | ✅ Yes | ✅ Yes | Spatial masks work in both |
| **Time-Windowed Injection** | ✅ Yes | ✅ Yes | Start/end step control |
| **Token-Level Masking** | ✅ Yes | ✅ Yes | Focus on specific words |
| **Attention Amplification** | ✅ Yes | ✅ Yes | Strengthen/weaken attention |
| **Multi-Prompt Blending** | ✅ Yes | ✅ Yes | Combine multiple prompts |

## What corpus-mlx CAN Do
✅ **Style Injection**: Add artistic styles to images
✅ **Attribute Modification**: Change colors, lighting, mood
✅ **Composition Changes**: Alter layout and structure
✅ **Detail Enhancement**: Add specific details or textures
✅ **Regional Effects**: Apply changes to specific image regions
✅ **Progressive Injection**: Control injection timing during generation

## What corpus-mlx CANNOT Do
❌ **Semantic Object Replacement**: Cannot change apple to banana
❌ **Entity Substitution**: Cannot replace cat with dog
❌ **Concept Swapping**: Cannot fundamentally change object types

## Technical Explanation

### Why Semantic Replacement Doesn't Work in corpus-mlx

1. **Timing of Intervention**:
   - CorePulse: Replaces text embeddings BEFORE attention
   - corpus-mlx: Modifies attention values AFTER embeddings are processed

2. **What Gets Modified**:
   - CorePulse: The actual text guidance (conditioning embeddings)
   - corpus-mlx: How strongly the model attends to text features

3. **Effect**:
   - CorePulse: Model "sees" different text → generates different objects
   - corpus-mlx: Model still "sees" original text → maintains object identity

### Code Comparison

**CorePulse (Conditioning Replacement)**:
```python
# Replaces the text embeddings
encoded_prompt = pipeline.encode_prompt(injection_prompt)
encoder_hidden_states = encoded_prompt * weight  # Replace conditioning
```

**corpus-mlx (Attention Manipulation)**:
```python
# Modifies attention values
v_modified = v * (1 - strength) + inject_vals * strength  # Blend values
```

## Recommendations

### For corpus-mlx Users
- Use for **style transfer** and **attribute modification**
- Don't expect **object replacement** to work
- Increase strength for stronger style effects (0.3-0.8 range)
- Use regional masks for localized changes

### To Achieve Semantic Replacement
Would require architectural changes to corpus-mlx:
1. Intercept text encoding before attention
2. Replace conditioning tensors entirely
3. Modify UNet to accept alternate conditioning

## Test Results
- ✅ Style injection: Works well
- ✅ Color/mood changes: Works well  
- ✅ Regional control: Works well
- ❌ Apple→Banana: Does not work
- ❌ Cat→Dog: Does not work
- ❌ Car→Bicycle: Does not work

## Conclusion
corpus-mlx is excellent for style transfer and attribute modification on Apple Silicon, but cannot achieve true semantic object replacement due to architectural differences from CorePulse. This is a fundamental limitation of manipulating attention values versus replacing conditioning embeddings.