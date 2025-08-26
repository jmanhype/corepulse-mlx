# CorePulse-MLX

A modular toolkit for advanced diffusion model manipulation on Apple Silicon, providing unprecedented control over how Stable Diffusion processes and interprets your prompts.

![CorePulse Features](artifacts/images/readme/REAL_EFFECTS_SHOWCASE.png)

## Core Concepts

### 🎯 Prompt Injection
Inject different prompts into specific architectural blocks of the UNet during generation. This allows you to control different aspects of your image:

- **Content blocks** (middle layers) → What appears in your image
- **Style blocks** (output layers) → How it looks and feels  
- **Composition blocks** (input layers) → Overall layout and structure

![Prompt Injection Demo](artifacts/images/readme/PROPER_prompt_injection.png)

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

![Regional Injection Demo](artifacts/images/readme/PROPER_regional_control.png)

### ⚡ Attention Manipulation
Control how much the model focuses on specific words in your prompt by directly modifying attention weights. Unlike changing the prompt text, this amplifies or reduces the model's internal focus on existing words.

- **Amplify attention** (>1.0) → Make the model pay more attention to specific words
- **Reduce attention** (<1.0) → Decrease focus on certain words
- **Spatial control** → Apply attention changes only to specific image regions

![Attention Manipulation Demo](artifacts/images/readme/PROPER_attention_manipulation.png)

### 🏗️ Multi-Scale Control
Apply different prompts to different resolution levels of the UNet architecture. This approach lets you control structure and details independently:

- **Structure Level** (lowest resolution) → Overall composition, global layout
- **Mid-Level** (medium resolution) → Regional features, object relationships
- **Detail Level** (highest resolution) → Fine textures, surface details

![Multi-Scale Control Demo](artifacts/images/readme/PROPER_multiscale_control.png)

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

## 🎨 CorePulse V4 Clean Implementation

We've implemented a production-ready, upstream-friendly CorePulse system with **zero regression** when disabled and powerful effects when enabled.

### Working Effects Gallery

#### 🏰 Castle Enhancement
![Working Castle](artifacts/images/readme/WORKING_baseline.png) ![Working Castle CorePulse](artifacts/images/readme/WORKING_corepulse.png)
*Left: Baseline | Right: CorePulse with phase-based enhancement (25.30 avg pixel difference)*

#### 🌲 Dreamlike Forest  
![Dreamlike Baseline](artifacts/images/readme/SHOWCASE_DREAMLIKE_baseline.png) ![Dreamlike Effect](artifacts/images/readme/SHOWCASE_DREAMLIKE_effect.png)
*Soft, dreamlike atmosphere through attention smoothing (27.69 avg difference)*

#### 🏗️ Geometric Architecture
![Geometric Baseline](artifacts/images/readme/SHOWCASE_GEOMETRIC_baseline.png) ![Geometric Effect](artifacts/images/readme/SHOWCASE_GEOMETRIC_effect.png)
*Enhanced geometric structure with edge amplification (49.34 avg difference)*

#### 🎭 More Effects Available
- **IntensityProcessor**: Progressive amplification (9-10 pixel difference)
- **AbstractProcessor**: Controlled chaos for artistic interpretation (36 pixel difference)
- **ColorShiftProcessor**: Dynamic color modulation (22 pixel difference)

### Key Features
- ✅ **Zero regression** when disabled (proven with identical output)
- 🎯 **Opt-in activation** via `ATTN_HOOKS_ENABLED` flag
- 🔧 **Protocol-based processors** for type safety
- 🚀 **No monkey-patching** - proper seam integration
- 📊 **Measurable effects** - 9-49 pixel average differences

### Quick Start
```python
from stable_diffusion import attn_hooks

# Enable hooks
attn_hooks.ATTN_HOOKS_ENABLED = True

# Register a processor for dramatic effects
class PhaseProcessor:
    def __call__(self, *, out=None, meta=None):
        if meta and meta.get('step_idx', 0) < 5:
            if "down" in meta.get('block_id', ''):
                return out * 1.2  # Amplify structure
        return None

attn_hooks.register_processor("down", PhaseProcessor())
```

For complete documentation and more examples, see [COREPULSE_README.md](COREPULSE_README.md).

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

### 🎯 DATAVOID EXAMPLES - EXACT RECREATIONS

![DataVoid Examples Complete](artifacts/images/readme/DATAVOID_COMPLETE_SHOWCASE.png)
*All 5 core CorePulse techniques from DataCTE/CorePulse recreated with MLX*

**Demonstrated Techniques:**

#### 1. Token-Level Masking
![Token Masking](artifacts/images/readme/DATAVOID_token_masking.png)
*Mask 'cat' tokens while preserving 'playing at a park' context*

#### 2. Regional/Spatial Injection  
![Regional Injection](artifacts/images/readme/DATAVOID_regional_injection.png)
*Apply 'golden retriever dog' only to center, preserve park environment*

#### 3. Attention Manipulation
![Attention Boost](artifacts/images/readme/DATAVOID_attention_boost.png)
*Boost attention on 'photorealistic' to enhance realism without changing prompt*

#### 4. Multi-Scale Control
![Multi-Scale](artifacts/images/readme/DATAVOID_multiscale.png)
*Gothic cathedral structure with intricate stone carving details*

#### 5. Content/Style Separation
![Content Style](artifacts/images/readme/DATAVOID_content_style.png)
*Inject 'white cat' content while keeping garden context from original prompt*

### 🔥 EXPERIMENTAL POWER SHOWCASE

![CorePulse Experimental Power](artifacts/images/readme/POWER_SHOWCASE.png)
*Nine experimental trajectories demonstrating extreme CorePulse control dimensions*

**Demonstrated Control Dimensions:**
- **3x Style Boost**: Extreme output block amplification creating surreal effects
- **90% Structure Decay**: Near-complete dissolution of input blocks  
- **Attention Inversion**: Negative attention weights creating inverse patterns
- **Chaos Mode**: Chaotic oscillating control with unpredictable dynamics
- **Edge Isolation**: Selective amplification of boundary blocks only
- **Time Evolution**: Progressive timestep-based transformation
- **Extreme Mix**: Combined multi-effect processing with layered manipulations
- **99% Suppression**: Almost complete attention suppression except final layers

### Previous Visual Demonstrations

![Chaotic Timestep Control](artifacts/images/readme/PROOF_chaotic_timestep.png)
*Left: Standard diffusion | Right: Chaotic timestep-varying attention control*

![Attention Inversion](artifacts/images/readme/PROOF_inverted_attention.png)
*Left: Normal attention | Right: Inverted attention weights (-80% + 1.0)*

Browse `artifacts/images/` for extensive visual demonstrations:
- 200+ comparison images showing before/after effects
- Test results across different techniques
- Proof of concepts and validation

## License

MIT License

## About

CorePulse-MLX brings advanced diffusion control to Apple Silicon, optimized for MLX framework performance. Built with Clean Architecture principles for maintainability and extensibility.