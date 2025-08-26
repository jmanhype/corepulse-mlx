#!/usr/bin/env python3
import argparse
import mlx.core as mx
import numpy as np
from PIL import Image
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper_simple import CorePulseStableDiffusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="out.png")
    args = ap.parse_args()

    print(f"Loading Stable Diffusion model...")
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    print(f"Generating: '{args.prompt}'")
    latents = cpsd.generate_latents(
        base_prompt=args.prompt,
        negative_text=args.negative,
        num_steps=args.steps, cfg_weight=args.cfg,
        n_images=1, height=args.h, width=args.w, seed=args.seed
    )
    for x_t in latents: mx.eval(x_t)
    
    print("Decoding image...")
    img = sd.decode(x_t); mx.eval(img)
    
    # Save image using PIL
    img_np = np.array(img)
    img_np = (img_np * 255).astype(np.uint8)
    im = Image.fromarray(img_np[0])  # Take first image
    im.save(args.output)
    print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()