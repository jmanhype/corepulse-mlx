#!/usr/bin/env python3
"""
Runner for all advanced artistic manipulation tests.
Tests sophisticated artistic capabilities beyond basic manipulation.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_test(test_file, test_name):
    """Run a single test and report results"""
    print(f"\n{'='*60}")
    print(f"🎨 Running: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} - PASSED ({elapsed:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {test_name} - FAILED ({elapsed:.2f}s)")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ {test_name} - TIMEOUT (exceeded 120s)")
        return False
    except Exception as e:
        print(f"💥 {test_name} - ERROR: {e}")
        return False

def main():
    print("🎨 Advanced Artistic Manipulation Tests")
    print("=" * 60)
    print("Testing sophisticated artistic capabilities...")
    print("These tests prove we can create complex artistic effects!")
    
    # Define all tests
    tests = [
        ("test_style_mixing.py", "Style Mixing - Multiple Artistic Styles"),
        ("test_concept_fusion.py", "Concept Fusion - Hybrid Concepts"),
        ("test_temporal_styles.py", "Temporal Styles - Time Period Generation"),
        ("test_cultural_blending.py", "Cultural Blending - Multicultural Elements"),
        ("test_abstract_conceptualization.py", "Abstract Conceptualization - Pure Abstractions")
    ]
    
    # Track results
    results = []
    passed = 0
    failed = 0
    
    # Run each test
    for test_file, test_name in tests:
        if Path(test_file).exists():
            success = run_test(test_file, test_name)
            results.append((test_name, success))
            if success:
                passed += 1
            else:
                failed += 1
        else:
            print(f"\n⚠️ Test file not found: {test_file}")
            results.append((test_name, False))
            failed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ADVANCED ARTISTIC TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n📈 Results: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("\n🎉 ALL ADVANCED ARTISTIC TESTS PASSED!")
        print("💡 We have successfully demonstrated:")
        print("   • Multi-style artistic fusion")
        print("   • Hybrid concept creation")
        print("   • Temporal style progression")
        print("   • Multicultural visual blending")
        print("   • Pure mathematical abstractions")
        print("\n🚀 These capabilities go beyond basic manipulation!")
        print("🎨 We can create entirely new forms of artistic expression!")
    else:
        print(f"\n⚠️ {failed} test(s) failed - review output above")
    
    # Check for generated images
    print("\n📷 Checking for generated images...")
    image_dir = Path("artifacts/images/advanced_artistic")
    if image_dir.exists():
        images = list(image_dir.glob("*.png"))
        if images:
            print(f"✅ Found {len(images)} generated images in {image_dir}")
            for img in sorted(images):
                print(f"   • {img.name}")
        else:
            print("⚠️ No images found in output directory")
    else:
        print("⚠️ Output directory not found")
    
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())