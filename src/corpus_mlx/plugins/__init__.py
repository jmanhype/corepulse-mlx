"""
CorePulse plugin system for extensibility.
"""

from typing import Dict, Any, Callable, Optional
from abc import ABC, abstractmethod


class CorePulsePlugin(ABC):
    """Base class for CorePulse plugins."""
    
    def __init__(self, name: str, priority: int = 0):
        """Initialize plugin.
        
        Args:
            name: Plugin name
            priority: Execution priority (higher runs first)
        """
        self.name = name
        self.priority = priority
        self.enabled = True
        
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through the plugin.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Modified data dictionary
        """
        pass
        
    def pre_generation(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before generation starts.
        
        Args:
            config: Generation configuration
            
        Returns:
            Modified configuration
        """
        return config
        
    def post_generation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called after generation completes.
        
        Args:
            result: Generation result
            
        Returns:
            Modified result
        """
        return result
        
    def on_step(self, step: int, latents: Any) -> Any:
        """Hook called on each denoising step.
        
        Args:
            step: Current step number
            latents: Current latents
            
        Returns:
            Modified latents
        """
        return latents


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        self.plugins: Dict[str, CorePulsePlugin] = {}
        
    def register(self, plugin: CorePulsePlugin):
        """Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self.plugins[plugin.name] = plugin
        
    def unregister(self, name: str):
        """Unregister a plugin by name.
        
        Args:
            name: Plugin name
        """
        if name in self.plugins:
            del self.plugins[name]
            
    def get(self, name: str) -> Optional[CorePulsePlugin]:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self.plugins.get(name)
        
    def get_enabled(self) -> list:
        """Get all enabled plugins sorted by priority.
        
        Returns:
            List of enabled plugins
        """
        enabled = [p for p in self.plugins.values() if p.enabled]
        return sorted(enabled, key=lambda x: x.priority, reverse=True)
        
    def process_all(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data through all enabled plugins.
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        for plugin in self.get_enabled():
            data = plugin.process(data)
        return data
        
    def clear(self):
        """Clear all registered plugins."""
        self.plugins.clear()


# Global plugin registry
PLUGIN_REGISTRY = PluginRegistry()