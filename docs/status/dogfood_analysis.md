# Dogfood Analysis - CorePulse MLX Implementation

## What's Working ✅

### 1. Image Generation
- SDXL-Turbo successfully generates images (512x512)
- Seeds maintain consistency between comparisons
- Memory usage reasonable (~11GB peak)

### 2. Basic Techniques Implemented
- **Attention Manipulation**: Amplifying/suppressing specific terms works
- **Token Masking**: Can mask and replace tokens (cat→dog transformation)
- **Multi-Scale Control**: Layer-specific injection functional
- **Regional Control**: Basic spatial injection working

### 3. Visual Output
- Comparison grids successfully created
- Side-by-side before/after format matches CorePulse
- Master showcase combines all techniques

## Issues to Fix 🔧

### 1. Attention Boost Effect
**Problem**: The "Original" vs "Attention Boosted" astronaut images look identical
- No visible difference between 5x photorealistic boost
- Might not be applying attention weights correctly during generation

**Solution Needed**:
- Verify attention weights are actually being used in the UNet forward pass
- Check if we need to modify the SD pipeline itself

### 2. Token Masking Inconsistency
**Problem**: Cat→Dog transformation changes the entire scene
- Background completely different (lighting, angle, grass)
- Should preserve scene context, only swap subject

**Solution Needed**:
- Use same latent initialization
- Apply masking at correct denoising steps
- Preserve more context tokens

### 3. Multi-Scale Control Not Visible
**Problem**: Can't see clear structure vs detail separation
- Cathedral comparison shows different images, not multi-scale effect
- Building variations don't show layer-specific control

**Solution Needed**:
- Implement proper UNet block intervention
- Hook into specific resolution layers
- Apply prompts at correct pyramid levels

### 4. Regional Control Missing
**Problem**: Fire/Ice, Day/Night splits not creating actual regional effects
- Images show mixed concepts, not spatial separation
- Need true left/right or top/bottom control

**Solution Needed**:
- Implement spatial attention masks
- Create position-aware injection
- Use cross-attention masking for regions

## Technical Issues 🐛

### 1. API Mismatch
```python
# Current (wrong):
controller.injector.inject_prompt_at_blocks()  # Doesn't exist

# Should be:
# Need to hook into UNet layers directly
```

### 2. Attention Weights Not Applied
```python
# Setting weights but not using them:
self.attention_weights[phrase] = weight
# Missing: Actually applying during generation
```

### 3. Layer Injection Not Working
```python
# Current:
controller.injector.layer_injections = {0: prompt1, 8: prompt2}
# But SD doesn't use this during forward pass
```

## Next Steps 📋

### Priority 1: Fix Attention Manipulation
1. Hook into SD's cross-attention layers
2. Modify attention scores during forward pass
3. Verify with heatmap visualization

### Priority 2: Fix Token Masking
1. Preserve initial noise/seed between comparisons
2. Apply masking only to cross-attention, not self-attention
3. Mask at specific timesteps (early for structure, late for details)

### Priority 3: Implement Real Multi-Scale
1. Access UNet's downsample/upsample blocks
2. Inject prompts at specific resolutions:
   - Blocks 0-3: Structure (32x32, 16x16)
   - Blocks 4-7: Mid-level (64x64)
   - Blocks 8-11: Details (256x256, 512x512)

### Priority 4: Add Regional Control
1. Create spatial masks (left/right, quadrants, etc.)
2. Apply masks to cross-attention maps
3. Weight different regions independently

## Code Fixes Needed

### 1. Proper UNet Hooking
```python
def hook_unet_attention(sd_model):
    """Hook into UNet's attention layers for manipulation."""
    for name, module in sd_model.unet.named_modules():
        if "attention" in name.lower():
            module.register_forward_hook(attention_hook)
```

### 2. Real Attention Application
```python
def attention_hook(module, input, output):
    """Modify attention during forward pass."""
    if hasattr(module, 'attention_weights'):
        # Apply our custom weights
        output = output * module.attention_weights
    return output
```

### 3. Spatial Masking
```python
def create_spatial_mask(height, width, region="left"):
    """Create masks for regional control."""
    mask = mx.zeros((height, width))
    if region == "left":
        mask[:, :width//2] = 1.0
    return mask
```

## Test Cases for Validation

1. **Attention Test**: Generate same prompt with 0.1x vs 10x on "photorealistic"
   - Should see clear quality difference
   
2. **Token Test**: Mask "red" in "red car on street"
   - Should get car in different color, same street
   
3. **Multi-Scale Test**: "Castle" at low-res, "ornate details" at high-res
   - Should see simple shape with complex textures
   
4. **Regional Test**: "Fire" left half, "ice" right half
   - Should see clear spatial separation

## Success Metrics

- [ ] Attention boost creates visible enhancement (>30% quality difference)
- [ ] Token masking preserves >80% of scene context
- [ ] Multi-scale shows clear layer separation
- [ ] Regional control creates distinct spatial zones
- [ ] All techniques work without changing seeds
- [ ] Memory usage stays under 16GB
- [ ] Generation time under 5 seconds per image

## Implementation Priority

1. **Fix attention manipulation first** - Core technique
2. **Fix token masking** - Most visually obvious
3. **Implement proper multi-scale** - Architectural requirement
4. **Add regional control** - Advanced feature