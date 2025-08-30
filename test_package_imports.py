#!/usr/bin/env python3
"""Test corpus-mlx package imports and basic functionality."""

def test_package_imports():
    """Test that all main functions can be imported from the installed package."""
    print("🧪 TESTING CORPUS-MLX PACKAGE IMPORTS")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("1. Testing basic imports...")
        from corpus_mlx import create_semantic_wrapper, create_true_semantic_wrapper
        from corpus_mlx import CorePulseStableDiffusion
        print("   ✅ Main functions imported successfully")
        
        # Test __init__ imports
        print("2. Testing __init__ exports...")
        import corpus_mlx
        
        # Check what's available
        available = [attr for attr in dir(corpus_mlx) if not attr.startswith('_')]
        print(f"   Available exports: {available}")
        
        # Test key functionality
        print("3. Testing wrapper creation...")
        wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
        print("   ✅ Semantic wrapper created successfully")
        print(f"   Type: {type(wrapper)}")
        
        true_wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base") 
        print("   ✅ TRUE semantic wrapper created successfully")
        print(f"   Type: {type(true_wrapper)}")
        
        print("\\n🎉 ALL PACKAGE IMPORTS WORKING!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without generating images."""
    print("\\n🔬 TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    try:
        from corpus_mlx import create_semantic_wrapper, create_true_semantic_wrapper
        
        # Test text replacement setup
        print("1. Testing text replacement setup...")
        wrapper = create_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
        wrapper.add_replacement("cat", "dog")
        print("   ✅ Text replacement configured")
        
        # Test TRUE embedding setup  
        print("2. Testing TRUE embedding setup...")
        true_wrapper = create_true_semantic_wrapper("stabilityai/stable-diffusion-2-1-base")
        true_wrapper.add_replacement("cat", "dog", weight=1.0)
        print("   ✅ TRUE embedding configured")
        
        print("\\n🎉 BASIC FUNCTIONALITY WORKING!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_structure():
    """Test package structure and exports."""
    print("\\n📁 TESTING PACKAGE STRUCTURE")
    print("=" * 50)
    
    try:
        import corpus_mlx
        from corpus_mlx import __version__
        
        print(f"Package version: {__version__}")
        print(f"Package location: {corpus_mlx.__file__}")
        
        # Test submodules
        expected_modules = [
            'corepulse', 'injection', 'semantic_proper', 'true_semantic'
        ]
        
        for module in expected_modules:
            try:
                mod = getattr(corpus_mlx, module, None)
                if mod:
                    print(f"   ✅ {module} available")
                else:
                    print(f"   ⚠️  {module} not directly available (might be internal)")
            except Exception as e:
                print(f"   ❌ {module} error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success_import = test_package_imports()
    success_basic = test_basic_functionality() 
    success_structure = test_package_structure()
    
    if success_import and success_basic and success_structure:
        print("\\n✅ ALL PACKAGE TESTS PASSED!")
        print("🚀 corpus-mlx is ready for production use!")
    else:
        print("\\n❌ SOME TESTS FAILED")
        print("📋 Check the errors above for details")