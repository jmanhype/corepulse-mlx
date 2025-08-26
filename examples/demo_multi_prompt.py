# Demo: multi-prompt regional injection
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion
from plugins.regional_prompt import add_regional_prompt
import mlx.core as mx

def main():
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    # Background prompt only (global in base prompt)
    base = "a serene valley with a lake at sunrise, volumetric light, detailed foliage"

    # Regional prompts
    add_regional_prompt(cpsd, "a bright sun", ("circle_frac", 0.8, 0.15, 0.1, 0.08), start=0.6, end=1.0, weight=0.8, token_mask="sun")
    add_regional_prompt(cpsd, "a wooden cabin", ("rect_frac", 0.3, 0.55, 0.18, 0.18, 0.08), start=0.35, end=0.9, weight=0.65, token_mask="cabin")

    latents = cpsd.generate_latents(
        base_prompt=base,
        negative_text="watermarks, extra text, logo",
        num_steps=40, cfg_weight=7.0, n_images=1, height=768, width=768, seed=101,
    )
    for x_t in latents:
        mx.eval(x_t)
    img = sd.decode(x_t)
    mx.eval(img)
    try:
        from stable_diffusion import to_image
        to_image(img, "out_multi.png")
    except Exception:
        pass

if __name__ == "__main__":
    main()
