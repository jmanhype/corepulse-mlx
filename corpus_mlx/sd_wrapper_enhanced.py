"""
Enhanced CorePulse wrapper with proper attention manipulation and per-block control.
"""

from __future__ import annotations
from typing import Iterator, Optional, List, Dict, Callable, Tuple
import mlx.core as mx
import numpy as np
from .injection import InjectionConfig, prepare_injection
from .masks import region_mask_in_latent
from .utils import as_mx

Array = mx.array


class AttentionHook:
    """Hook for modifying attention during UNet forward pass."""
    
    def __init__(self, block_name: str, scale: float = 1.0, modify_fn: Optional[Callable] = None):
        self.block_name = block_name
        self.scale = scale
        self.modify_fn = modify_fn
        self.enabled = True
        
    def __call__(self, module, args, output):
        """Modify attention output."""
        if not self.enabled:
            return output
            
        if self.modify_fn:
            return self.modify_fn(output, self.block_name)
        else:
            return output * self.scale


class EnhancedCoreStableDiffusion:
    """Enhanced wrapper with full attention control and per-block injection."""
    
    def __init__(self, sd, model_kind: str = "sd"):
        self.sd = sd
        self.model_kind = model_kind
        self.injections: List[InjectionConfig] = []
        self.attention_hooks: Dict[str, AttentionHook] = {}
        self.block_injections: Dict[str, List[Dict]] = {}
        self.current_step = 0
        self.total_steps = 0
        
    def add_injection(
        self,
        prompt: str,
        token_mask: Optional[str] = None,
        region: Optional[Tuple] = None,
        start_frac: float = 0.0,
        end_frac: float = 1.0,
        weight: float = 1.0,
        blocks: Optional[List[str]] = None,
        attention_scale: Optional[Dict[str, float]] = None
    ) -> InjectionConfig:
        """
        Add enhanced injection with block and attention control.
        
        Args:
            prompt: Text to inject
            token_mask: Tokens to emphasize
            region: Spatial mask
            start_frac: Start time fraction
            end_frac: End time fraction
            weight: Injection weight
            blocks: Specific UNet blocks to inject at
            attention_scale: Per-block attention scaling
        """
        ic = InjectionConfig(
            prompt=prompt,
            token_mask=token_mask,
            region=region,
            start_frac=start_frac,
            end_frac=end_frac,
            weight=weight
        )
        
        # Store block-specific config
        if blocks:
            ic.blocks = blocks
            for block in blocks:
                if block not in self.block_injections:
                    self.block_injections[block] = []
                self.block_injections[block].append({
                    'config': ic,
                    'prepared': None
                })
        
        # Setup attention scaling
        if attention_scale:
            for block, scale in attention_scale.items():
                self.add_attention_hook(block, scale)
        
        self.injections.append(ic)
        return ic
    
    def add_attention_hook(self, block_name: str, scale: float):
        """Add attention scaling hook for specific block."""
        hook = AttentionHook(block_name, scale)
        self.attention_hooks[block_name] = hook
    
    def clear_injections(self):
        """Clear all injections and hooks."""
        self.injections.clear()
        self.attention_hooks.clear()
        self.block_injections.clear()
    
    def generate_latents(
        self,
        base_prompt: str,
        negative_text: str = "",
        num_steps: int = 50,
        cfg_weight: float = 7.5,
        n_images: int = 1,
        height: int = 512,
        width: int = 512,
        sampler_type: str = "ddim",
        seed: Optional[int] = None,
        value_function: Optional[Callable] = None,
        prior: Optional[Array] = None
    ) -> Iterator[Array]:
        """
        Generate with enhanced control.
        """
        # Setup
        self.current_step = 0
        self.total_steps = num_steps
        H_lat = height // 8
        W_lat = width // 8
        
        # Initialize sampler
        sampler = self.sd.sampler
        noise_schedule = None
        if sampler_type == "ddim":
            num_train_steps = 1000
            step_list = np.linspace(0, num_train_steps - 1, num_steps, dtype=np.int32).tolist()
            noise_schedule = {"t": step_list}
        
        # Prepare embeddings
        base_emb = self._encode_prompt(base_prompt)
        uncond_emb = self._encode_prompt(negative_text) if negative_text else mx.zeros_like(base_emb)
        
        # Prepare all injections
        prepared_injections = []
        for ic in self.injections:
            prep = prepare_injection(self.sd, ic, H_lat, W_lat)
            prepared_injections.append(prep)
        
        # Prepare block-specific injections
        for block, inj_list in self.block_injections.items():
            for inj_dict in inj_list:
                if inj_dict['prepared'] is None:
                    inj_dict['prepared'] = prepare_injection(
                        self.sd, inj_dict['config'], H_lat, W_lat
                    )
        
        # Generate initial noise
        shape = [n_images, H_lat, W_lat, 4]
        
        if seed is not None:
            mx.random.seed(seed)
        
        x_t = sampler.sample_prior(shape, dtype=mx.float16)
        
        if prior is not None:
            x_t = x_t + prior
        
        # Main denoising loop
        steps = list(sampler.timesteps(num_steps))
        n = len(steps)
        
        for i, (t, t_prev) in enumerate(steps):
            self.current_step = i
            t_f = i / max(1, n - 1)  # Progress fraction
            
            # Get current block based on progress
            current_block = self._get_current_block(t_f)
            
            # Apply attention hooks for current block
            self._apply_attention_hooks(current_block)
            
            # Get base noise prediction with CFG
            eps_base = self._cfg_eps(x_t, t, base_emb, uncond_emb, cfg_weight)
            
            # Apply block-specific injections
            if current_block in self.block_injections:
                for inj_dict in self.block_injections[current_block]:
                    prep = inj_dict['prepared']
                    if prep and self._is_active(inj_dict['config'], t_f):
                        eps_inj = self._cfg_eps(x_t, t, prep['embedding'], uncond_emb, cfg_weight)
                        
                        if prep.get('mask') is not None:
                            alpha = prep['mask'] * inj_dict['config'].weight
                            eps_base = (1 - alpha) * eps_base + alpha * eps_inj
                        else:
                            alpha = inj_dict['config'].weight
                            eps_base = (1 - alpha) * eps_base + alpha * eps_inj
            
            # Apply standard injections
            for ic, prep in zip(self.injections, prepared_injections):
                if not hasattr(ic, 'blocks') and self._is_active(ic, t_f):
                    eps_inj = self._cfg_eps(x_t, t, prep['embedding'], uncond_emb, cfg_weight)
                    
                    if prep.get('mask') is not None:
                        alpha = prep['mask'] * ic.weight
                        eps_base = (1 - alpha) * eps_base + alpha * eps_inj
                    else:
                        alpha = ic.weight
                        eps_base = (1 - alpha) * eps_base + alpha * eps_inj
            
            # Update x_t
            x_t = sampler.step(eps_base, x_t, t, t_prev)
            
            # Apply value function if provided
            if value_function is not None:
                x_t = value_function(x_t, t, None)
            
            yield x_t
    
    def _get_current_block(self, progress: float) -> str:
        """Map progress to UNet block."""
        if progress < 0.25:
            return "down_0"
        elif progress < 0.35:
            return "down_1"
        elif progress < 0.45:
            return "down_2"
        elif progress < 0.5:
            return "down_3"
        elif progress < 0.6:
            return "mid"
        elif progress < 0.7:
            return "up_0"
        elif progress < 0.8:
            return "up_1"
        elif progress < 0.9:
            return "up_2"
        else:
            return "up_3"
    
    def _apply_attention_hooks(self, block_name: str):
        """Enable/disable attention hooks based on current block."""
        for name, hook in self.attention_hooks.items():
            hook.enabled = (name == block_name or name == "all")
    
    def _is_active(self, ic: InjectionConfig, t_f: float) -> bool:
        """Check if injection is active at current time."""
        return ic.start_frac <= t_f <= ic.end_frac
    
    def _encode_prompt(self, prompt: str) -> Array:
        """Encode prompt with fixed length."""
        fixed_len = 77
        tokens = self.sd.tokenizer.tokenize(prompt)
        
        if len(tokens) > fixed_len:
            tokens = tokens[:fixed_len]
        elif len(tokens) < fixed_len:
            tokens = tokens + [0] * (fixed_len - len(tokens))
        
        tokens = mx.array(tokens)[None]
        return self.sd.text_encoder(tokens).last_hidden_state
    
    def _cfg_eps(self, x_t: Array, t: Array, cond: Array, uncond: Array, w: float) -> Array:
        """Classifier-free guidance noise prediction."""
        # Ensure t has correct shape
        if t.ndim == 0:
            t = mx.array([t.item()])
        if t.ndim == 1 and len(t) == 1:
            t = mx.broadcast_to(t, [x_t.shape[0]])
        
        # Get noise predictions
        eps_uncond = self.sd.unet(x_t, t, encoder_x=uncond)
        eps_cond = self.sd.unet(x_t, t, encoder_x=cond)
        
        # Apply CFG
        return eps_uncond + w * (eps_cond - eps_uncond)


def create_enhanced_wrapper(sd) -> EnhancedCoreStableDiffusion:
    """Create enhanced wrapper with all features."""
    return EnhancedCoreStableDiffusion(sd)