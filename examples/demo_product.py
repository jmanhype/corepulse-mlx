# Minimal demo: HQ product placement (inpaint mode)
from stable_diffusion import StableDiffusion
from corpus_mlx.sd_wrapper import CorePulseStableDiffusion
from plugins.product_placement import build_product_placement
import mlx.core as mx

def main():
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=True)
    cpsd = CorePulseStableDiffusion(sd)

    build_product_placement(
        cpsd,
        region=("rect_frac", 0.55, 0.15, 0.25, 0.25, 0.06),
        mode="inpaint",
        reference_rgba_path="product.png",
        phase=(0.55, 1.0),
        alpha=1.0,
        cfg_cap=6.0,
        ramp_steps=8,
    )

    latents = cpsd.generate_latents(
        base_prompt="a cozy wooden desk near a window, soft morning light, film grain",
        negative_text="extra logos, melted text, wrong labels, distortions, watermark",
        num_steps=36, cfg_weight=6.0, n_images=1, height=768, width=768, seed=42,
    )
    for x_t in latents:
        mx.eval(x_t)
    img = sd.decode(x_t)
    mx.eval(img)
    # Save image (assumes helper in your MLX example; otherwise write your own Numpy saver)
    try:
        from stable_diffusion import to_image
        to_image(img, "out_product.png")
    except Exception:
        pass

if __name__ == "__main__":
    main()
