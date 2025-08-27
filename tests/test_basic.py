#!/usr/bin/env python3
"""
Basic tests for CorePulse functionality.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from corpus_mlx import CorePulse, InjectionConfig
from corpus_mlx.utils import KVRegistry


class TestCorePulseBasic(unittest.TestCase):
    """Test basic CorePulse functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.corepulse = CorePulse()
        self.registry = KVRegistry()
        
    def test_initialization(self):
        """Test CorePulse initialization."""
        self.assertIsNotNone(self.corepulse)
        self.assertIsNotNone(self.corepulse.kv_registry)
        self.assertIsNotNone(self.corepulse.regional_control)
        
    def test_add_injection(self):
        """Test adding injections."""
        # Note: Without a real model, we can only test the interface
        self.corepulse.add_injection(
            prompt="test prompt",
            strength=0.3,
            blocks=["mid", "up_0"]
        )
        # Should not raise an error
        
    def test_injection_config(self):
        """Test InjectionConfig creation."""
        config = InjectionConfig(
            inject_prompt="test",
            strength=0.5,
            blocks=["mid"],
            start_step=0,
            end_step=10
        )
        
        self.assertEqual(config.inject_prompt, "test")
        self.assertEqual(config.strength, 0.5)  # Within safe range
        self.assertEqual(config.blocks, ["mid"])
        
    def test_strength_clamping(self):
        """Test that strength values are clamped to safe range."""
        # Test over-limit
        config = InjectionConfig(
            inject_prompt="test",
            strength=2.0  # Should be clamped to 0.5
        )
        self.assertEqual(config.strength, 0.5)
        
        # Test under-limit
        config = InjectionConfig(
            inject_prompt="test",
            strength=-1.0  # Should be clamped to 0.1
        )
        self.assertEqual(config.strength, 0.1)
        
    def test_kv_registry(self):
        """Test KV registry functionality."""
        # Test setting and getting hooks
        def dummy_hook(q, k, v, meta=None):
            return q, k, v
            
        self.registry.set("test_block", dummy_hook)
        hook = self.registry.get("test_block")
        self.assertEqual(hook, dummy_hook)
        
        # Test clearing
        self.registry.clear()
        hook = self.registry.get("test_block")
        self.assertIsNone(hook)
        
    def test_registry_enable_disable(self):
        """Test registry enable/disable functionality."""
        def dummy_hook(q, k, v, meta=None):
            return q, k, v
            
        self.registry.set("test_block", dummy_hook)
        
        # Should work when enabled
        self.assertTrue(self.registry.active)
        hook = self.registry.get("test_block")
        self.assertIsNotNone(hook)
        
        # Should return None when disabled
        self.registry.disable()
        self.assertFalse(self.registry.active)
        hook = self.registry.get("test_block")
        self.assertIsNone(hook)
        
        # Should work again when re-enabled
        self.registry.enable()
        self.assertTrue(self.registry.active)
        hook = self.registry.get("test_block")
        self.assertIsNotNone(hook)


class TestInjectionFlow(unittest.TestCase):
    """Test injection flow and hook system."""
    
    def test_injection_without_model(self):
        """Test that injection setup works even without a model."""
        corepulse = CorePulse()
        
        # Should handle no model gracefully
        with self.assertRaises(ValueError):
            corepulse.generate("test prompt")
            
    def test_clear_functionality(self):
        """Test clearing injections and settings."""
        corepulse = CorePulse()
        
        # Add some injections
        corepulse.add_injection("test1", strength=0.3)
        corepulse.add_regional_prompt("test2", (0, 0, 100, 100))
        
        # Clear everything
        corepulse.clear()
        
        # Verify cleared
        self.assertEqual(len(corepulse.regional_control.regions), 0)
        self.assertEqual(len(corepulse.kv_registry.hooks), 0)


if __name__ == "__main__":
    unittest.main()