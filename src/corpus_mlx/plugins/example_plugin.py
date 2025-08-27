"""
Example plugin demonstrating the CorePulse plugin system.
"""

from typing import Dict, Any
import mlx.core as mx
from . import CorePulsePlugin


class ExamplePlugin(CorePulsePlugin):
    """Example plugin that demonstrates plugin functionality."""
    
    def __init__(self):
        super().__init__(name="example", priority=10)
        self.step_count = 0
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Modified data dictionary
        """
        # Example: Add metadata
        data["plugin_processed"] = True
        data["plugin_name"] = self.name
        
        # Example: Modify tensors if present
        if "embeddings" in data:
            # Apply some transformation
            embeddings = data["embeddings"]
            data["embeddings"] = self._transform_embeddings(embeddings)
            
        return data
        
    def pre_generation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Called before generation starts.
        
        Args:
            config: Generation configuration
            
        Returns:
            Modified configuration
        """
        # Example: Log configuration
        print(f"[{self.name}] Starting generation with prompt: {config.get('prompt', '')[:50]}...")
        
        # Example: Modify configuration
        if "guidance_scale" in config:
            # Slightly adjust guidance scale
            config["guidance_scale"] = config["guidance_scale"] * 1.1
            
        self.step_count = 0
        return config
        
    def post_generation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Called after generation completes.
        
        Args:
            result: Generation result
            
        Returns:
            Modified result
        """
        # Example: Add metadata to result
        result["plugin_metadata"] = {
            "plugin": self.name,
            "total_steps": self.step_count
        }
        
        print(f"[{self.name}] Generation complete after {self.step_count} steps")
        return result
        
    def on_step(self, step: int, latents: Any) -> Any:
        """Called on each denoising step.
        
        Args:
            step: Current step number
            latents: Current latents
            
        Returns:
            Modified latents
        """
        self.step_count += 1
        
        # Example: Apply subtle modification every 5 steps
        if step % 5 == 0:
            # Add tiny noise for variation
            noise = mx.random.normal(latents.shape) * 0.001
            latents = latents + noise
            
        return latents
        
    def _transform_embeddings(self, embeddings):
        """Example embedding transformation.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Transformed embeddings
        """
        # Example: Normalize embeddings
        norm = mx.linalg.norm(embeddings, axis=-1, keepdims=True)
        norm = mx.maximum(norm, 1e-12)
        return embeddings / norm


class LoggingPlugin(CorePulsePlugin):
    """Plugin that logs all operations for debugging."""
    
    def __init__(self, log_file: str = "corepulse.log"):
        super().__init__(name="logger", priority=100)  # High priority to run first
        self.log_file = log_file
        self.logs = []
        
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Log data passing through.
        
        Args:
            data: Input data
            
        Returns:
            Unchanged data
        """
        self.logs.append({
            "type": "process",
            "keys": list(data.keys()),
            "shapes": {k: str(v.shape) if hasattr(v, "shape") else type(v).__name__ 
                      for k, v in data.items()}
        })
        return data
        
    def pre_generation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Log generation start.
        
        Args:
            config: Generation configuration
            
        Returns:
            Unchanged configuration
        """
        self.logs.append({
            "type": "pre_generation",
            "prompt": config.get("prompt", "")[:100],
            "steps": config.get("num_inference_steps", 0),
            "size": f"{config.get('width', 0)}x{config.get('height', 0)}"
        })
        return config
        
    def save_logs(self):
        """Save logs to file."""
        import json
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)