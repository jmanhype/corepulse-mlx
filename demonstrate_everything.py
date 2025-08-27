#\!/usr/bin/env python3
import sys
import gc
from pathlib import Path
import mlx.core as mx
import PIL.Image
import numpy as np

sys.path.append('src/adapters/mlx/mlx-examples/stable_diffusion')

from stable_diffusion import attn_scores
attn_scores.enable_kv_hooks(True)

from stable_diffusion import StableDiffusionXL

def create_demonstration_hook(technique="blend", inject_prompt="", strength=0.15):
    def hook(q, k, v, meta=None):
        if k.shape[2] < 100 and k.shape[2] == v.shape[2]:
            batch, heads, seq_len, dim = v.shape
            block_id = meta.get('block_id', 'unknown')
            
            if technique == "suppress":
                v_new = v * 0.7
                print(f"    ✅ Suppression at {block_id}: 70% strength")
                return q, k, v_new
                
            elif technique == "amplify":
                v_new = v * 1.3
                v_new = mx.clip(v_new, -10.0, 10.0)
                print(f"    ✅ Amplification at {block_id}: 130% strength")
                return q, k, v_new
                
        return q, k, v
    
    return hook

def main():
    print("🚀 CorePulse V4 - Fixed Demonstration")
    print("=" * 60)
    
    base_prompt = "a majestic lion in savanna"
    num_steps = 4
    cfg_weight = 7.5
    seed = 42
    
    output_dir = Path("artifacts/images/complete_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("
📦 Loading SDXL-Turbo...")
    model = StableDiffusionXL("stabilityai/sdxl-turbo", float16=True)
    
    demos = [
        ("baseline", None, {}),
        ("suppress", "suppress", {"strength": 0.3}),
        ("amplify", "amplify", {"strength": 0.3})
    ]
    
    for demo_name, technique, params in demos:
        print(f"
🎨 Demo: {demo_name}")
        
        attn_scores.KV_REGISTRY.clear()
        
        if technique:
            hook = create_demonstration_hook(technique, **params)
            for block in ['mid', 'up_0']:
                attn_scores.KV_REGISTRY.set(block, hook)
        
        print(f"  Generating with prompt: '{base_prompt}'")
        latents = model.generate_latents(
            base_prompt,
            num_steps=num_steps,
            cfg_weight=cfg_weight,
            seed=seed
        )
        
        for i, x in enumerate(latents):
            if i == num_steps - 1:
                img = model.decode(x)
        
        img_array = (img[0] * 255).astype(mx.uint8)
        pil_img = PIL.Image.fromarray(np.array(img_array))
        
        filename = f"{demo_name}.png"
        pil_img.save(output_dir / filename)
        print(f"  ✅ Saved: {filename}")
    
    del model
    mx.metal.clear_cache()
    gc.collect()
    
    print("
" + "=" * 60)
    print("✅ Demonstration Complete\!")

if __name__ == "__main__":
    main()
