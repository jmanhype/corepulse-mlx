#!/usr/bin/env python3
"""
Master Runner: Execute all embedding injection tests
Demonstrates the SECOND technique - complete embedding injection pipeline.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_test(test_file, test_name):
    """Run a single test and report results"""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {test_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per test
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {test_name} completed in {elapsed:.1f}s")
            # Show last few lines of output
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-3:]:
                if line:
                    print(f"   {line}")
            return True
        else:
            print(f"❌ FAILED: {test_name}")
            print(f"Error output:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {test_name} took too long")
        return False
    except Exception as e:
        print(f"❌ ERROR running {test_name}: {e}")
        return False

def main():
    print("🎯 CorePulse V4: Embedding Injection Test Suite")
    print("=" * 60)
    print("Running all 7 embedding injection capability tests...")
    print("This demonstrates TRUE prompt injection, not just attention masking!")
    
    # Define all tests
    tests = [
        ("test_multi_prompt.py", "Multi-Prompt Injection"),
        ("test_prompt_replacement.py", "Complete Prompt Replacement"),
        ("test_regional_semantic.py", "Regional Semantic Control"),
        ("test_dynamic_swapping.py", "Dynamic Prompt Swapping"),
        ("test_embedding_blend.py", "Embedding Blending"),
        ("test_word_level.py", "Word-Level Manipulation"),
        ("test_progressive_embedding.py", "Progressive Embedding Injection"),
    ]
    
    # Check that all test files exist
    print("\n📁 Checking test files...")
    all_exist = True
    for test_file, test_name in tests:
        if not Path(test_file).exists():
            print(f"   ❌ Missing: {test_file}")
            all_exist = False
        else:
            print(f"   ✅ Found: {test_file}")
    
    if not all_exist:
        print("\n❌ Some test files are missing!")
        return
    
    # Create output directory
    output_dir = Path("artifacts/images/embedding_injection")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")
    
    # Run all tests
    print("\n🔬 Starting test execution...")
    results = []
    
    for test_file, test_name in tests:
        success = run_test(test_file, test_name)
        results.append((test_name, success))
        
        # Small delay between tests
        if success:
            time.sleep(2)
    
    # Summary
    print("\n" + "="*60)
    print("📊 EMBEDDING INJECTION TEST RESULTS")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        if success:
            print(f"✅ PASS: {test_name}")
            passed += 1
        else:
            print(f"❌ FAIL: {test_name}")
            failed += 1
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL EMBEDDING INJECTION TESTS PASSED!")
        print("\n💡 Proven Capabilities:")
        print("   ✅ Multi-prompt injection at different blocks")
        print("   ✅ Complete prompt replacement")
        print("   ✅ Regional semantic control") 
        print("   ✅ Dynamic prompt swapping during generation")
        print("   ✅ Mathematical embedding blending")
        print("   ✅ Word-level token manipulation")
        print("   ✅ Progressive conceptual morphing")
        print("\n🏆 CorePulse V4 has FULL semantic control over SDXL!")
    else:
        print(f"⚠️ {failed} tests failed. Check the output above for details.")
    
    # List generated images
    print(f"\n📸 Generated images in {output_dir}:")
    for img_file in sorted(output_dir.glob("*.png")):
        print(f"   🖼️ {img_file.name}")

if __name__ == "__main__":
    main()