import pytest

def test_injection_config_import():
    from corpus_mlx.injection import InjectionConfig
    ic = InjectionConfig(prompt="test", weight=0.5)
    assert ic.prompt == "test"
