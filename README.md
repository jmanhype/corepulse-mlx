# CorePulse-MLX

A modular toolkit for advanced diffusion model manipulation on Apple Silicon, providing unprecedented control over how Stable Diffusion processes and interprets your prompts.

![CorePulse Features](artifacts/images/readme/REAL_COREPULSE_SHOWCASE.png)

## Core Concepts

### 🎯 Prompt Injection
Inject different prompts into specific architectural blocks of the UNet during generation. This allows you to control different aspects of your image:

- **Content blocks** (middle layers) → What appears in your image
- **Style blocks** (output layers) → How it looks and feels  
- **Composition blocks** (input layers) → Overall layout and structure

![Prompt Injection Demo](artifacts/images/readme/REAL_prompt_injection.png)
*Left: Base prompt "orange cat in garden" | Right: Same prompt with "dog" injected into content blocks*

### 🎭 Token-Level Attention Masking
Control which parts of your prompt have influence by masking attention to specific tokens/words. This is different from spatial masking - it works at the linguistic level rather than image regions.

- **Selective attention** → Choose which words in your prompt get processed
- **Token-level control** → Fine-grained control over prompt interpretation
- **Linguistic precision** → Target specific concepts without changing prompt text

*Example: In "a cat playing at a park", mask attention to "cat" tokens while preserving attention to "playing at a park"*

### 🗺️ Regional/Spatial Injection
Apply prompt injections only to specific regions of the image using spatial masks. This enables surgical control over different areas of your image.

- **Targeted replacement** → Change specific image regions while preserving context
- **Spatial precision** → Control exactly where changes occur
- **Context preservation** → Background and surroundings remain untouched

![Regional Injection Demo](artifacts/images/readme/REAL_regional_control.png)
*Left: Original forest scene | Right: Waterfall injected into center region only*

### ⚡ Attention Manipulation
Control how much the model focuses on specific words in your prompt by directly modifying attention weights. Unlike changing the prompt text, this amplifies or reduces the model's internal focus on existing words.

- **Amplify attention** (>1.0) → Make the model pay more attention to specific words
- **Reduce attention** (<1.0) → Decrease focus on certain words
- **Spatial control** → Apply attention changes only to specific image regions

![Attention Manipulation Demo](artifacts/images/readme/REAL_attention_control.png)
*Left: Standard generation | Right: "Photorealistic" attention boosted 5x*

### 🏗️ Multi-Scale Control
Apply different prompts to different resolution levels of the UNet architecture. This approach lets you control structure and details independently:

- **Structure Level** (lowest resolution) → Overall composition, global layout
- **Mid-Level** (medium resolution) → Regional features, object relationships
- **Detail Level** (highest resolution) → Fine textures, surface details

![Multi-Scale Control Demo](artifacts/images/readme/REAL_multi_scale.png)
*Left: Gothic cathedral structure | Right: Same structure with weathered stone details*

## Technical Features

- **🔧 Multi-Architecture Support**: MLX-optimized for Apple Silicon (M1/M2/M3)
- **🎯 Block-Level Control**: Target specific UNet blocks (input, middle, output)
- **📐 Flexible Interfaces**: Simple one-liners to advanced multi-block configurations  
- **🔌 Seamless Integration**: Drop-in compatibility with MLX Stable Diffusion
- **🧹 Context Management**: Automatic patch cleanup with Python context managers
- **⏱️ Precise Timing Control**: Sigma-based injection windows for optimal effect
- **🚫 Zero-Regression Hooks**: Disabled by default for zero performance impact

## Quick Examples

### Prompt Injection: Content/Style Separation
```python
from stable_diffusion import attn_hooks

# Enable hooks and inject different content while keeping scene
attn_hooks.enable_hooks()

class ContentInjector:
    def __call__(self, *, out=None, meta=None):
        # Inject "white cat" into content blocks
        if meta.get('block_type') == 'middle':
            return self.inject_content(out, "white cat")
        return out

processor = ContentInjector()
attn_hooks.register_processor('middle', processor)

# Base prompt provides context, injection overrides content  
# Result: White cat in garden (content replaced, context preserved)
```

### Regional/Spatial Injection: Surgical Precision
```python
from src.core.domain import masks

# Create a spatial mask for the region you want to modify
mask = masks.create_center_circle_mask(size=(1024, 1024), radius=300)

# Apply "golden retriever dog" only to masked region
# The mask ensures only the center region changes
# Result: Dog in center, park environment perfectly preserved
```

### Attention Manipulation: Focus Control  
```python
from stable_diffusion import attn_hooks

class AttentionBooster:
    def __call__(self, *, out=None, meta=None):
        # Boost attention on "photorealistic" by 5x
        if "photorealistic" in meta.get('tokens', []):
            return out * 5.0
        return out

# Same prompt, but model focuses much more on making it photorealistic
```

### Multi-Scale Control: Structure + Details
```python
# Control structure and details independently
structure_prompt = "gothic cathedral silhouette, imposing architecture"
detail_prompt = "weathered stone, intricate carvings, moss-covered surfaces"

# Apply to different resolution levels
# Result: Gothic cathedral structure with detailed stone textures
```

## When to Use Which Technique

| Technique | Use When | Example |
|-----------|----------|---------|
| **Prompt Injection** | You want to replace/add content while keeping context | Generate a cat in a dog scene |
| **Token Masking** | You want to selectively ignore parts of your prompt | Mask "cat" tokens, keep "park" |
| **Regional Injection** | You want surgical precision in specific regions | Replace center only |
| **Attention Manipulation** | You want to emphasize existing words more | Boost "photorealistic" |
| **Multi-Scale Control** | You want different structure and details | Castle structure + stone details |

## 🏗️ Clean Architecture Structure

Following Uncle Bob's Clean Architecture principles:

```
corepulse-mlx/
├── src/
│   ├── core/
│   │   ├── domain/                    # 🎯 Pure business logic
│   │   │   ├── attention.py           # Attention domain models
│   │   │   ├── injection.py           # Injection business rules  
│   │   │   └── masks.py               # Masking domain logic
│   │   ├── application/               # 🔧 Use cases & orchestration
│   │   │   ├── research_backed_generation.py  # Advanced generation
│   │   │   └── stabilized_generation.py       # Gentle enhancement
│   │   └── infrastructure/            # 💾 Technical implementations
│   └── adapters/                      # 🔌 External system integrations
│       ├── mlx/                       # MLX framework adapter
│       └── stable_diffusion/          # SD integration adapter
```

## Installation

```bash
# Clone the repository
git clone https://github.com/jmanhype/corepulse-mlx.git
cd corepulse-mlx

# Install dependencies (MLX optimized for Apple Silicon)
pip install -r requirements.txt
```

## Advanced Usage

CorePulse-MLX offers multiple levels of control:

### Interfaces by Complexity
- **SimpleInjector** → One-liner injection for quick experiments
- **AdvancedInjector** → Multi-block, multi-prompt configurations
- **MultiScaleController** → Resolution-aware structure/detail control
- **AttentionManipulator** → Precise attention weight control
- **RegionalController** → Spatial masks for region-specific control

### Architecture Components
- **UNetPatcher** → Low-level UNet modification engine
- **BlockMapper** → Automatic block detection for any model
- **InjectionProcessor** → Custom attention processors with sigma timing
- **Utilities** → Auto-detection, validation, convenience functions

## Real-World Examples

Explore the `docs/examples/` directory for complete implementations:

- **Content/Style Split** → Generate a cat with oil painting style
- **Token Masking** → Selective prompt token control
- **Regional Control** → Left half: crystal castle, Right half: fire dragon
- **Attention Boost** → Amplify "photorealistic" for enhanced realism
- **Multi-Scale** → Medieval fortress structure + weathered stone details

## Visual Proof Gallery

Browse `artifacts/images/` for extensive visual demonstrations:
- 200+ comparison images showing before/after effects
- Test results across different techniques
- Proof of concepts and validation

## License

MIT License

## About

CorePulse-MLX brings advanced diffusion control to Apple Silicon, optimized for MLX framework performance. Built with Clean Architecture principles for maintainability and extensibility.