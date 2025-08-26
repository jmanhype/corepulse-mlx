#!/usr/bin/env python3
import argparse
import mlx.core as mx
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=36)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    latents = cpsd.generate_latents(
        base_prompt=args.prompt,
        negative_text=args.negative,
        num_steps=args.steps, cfg_weight=args.cfg,
        n_images=1, height=args.h, width=args.w, seed=args.seed
    )
    for x_t in latents: mx.eval(x_t)
    img = sd.decode(x_t); mx.eval(img)
    try:
        from stable_diffusion import to_image
        to_image(img, "out.png")
    except Exception:
        pass

if __name__ == "__main__":
    main()
