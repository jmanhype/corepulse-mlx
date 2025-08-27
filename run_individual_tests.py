#!/usr/bin/env python3
"""
Master Script: Run All Individual Tests
Executes each CorePulse V4 capability test individually for detailed analysis.
"""

import subprocess
import sys
from pathlib import Path
import time

def run_test(script_name, description):
    """Run an individual test script and report results"""
    print(f"\n{'='*80}")
    print(f"🚀 RUNNING: {description}")
    print(f"📄 Script: {script_name}")
    print('='*80)
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=False)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {description} - FAILED ({duration:.1f}s)")
            return False
            
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False

def main():
    print("🎯 COREPULSE V4 INDIVIDUAL TEST SUITE")
    print("=" * 80)
    print("Running each capability test individually for detailed analysis")
    print("Each test generates its own image in artifacts/images/individual_tests/")
    
    # List of all tests to run
    tests = [
        ("test_baseline.py", "Baseline Generation (No Manipulation)"),
        ("test_token_removal.py", "Token Removal (Concept Extraction)"),
        ("test_amplification.py", "Extreme Amplification (5x Boost)"),
        ("test_suppression.py", "Extreme Suppression (95% Reduced)"),
        ("test_chaos.py", "Maximum Chaos (Noise Injection)"),
        ("test_inversion.py", "Attention Inversion (Anti-Prompt)"),
        ("test_progressive.py", "Progressive Manipulation (Gradient Control)")
    ]
    
    # Track results
    passed = 0
    failed = 0
    total_start = time.time()
    
    # Run each test
    for script, description in tests:
        if run_test(script, description):
            passed += 1
        else:
            failed += 1
        
        print(f"\n⏳ Waiting 2 seconds before next test...")
        time.sleep(2)
    
    total_time = time.time() - total_start
    
    # Final summary
    print(f"\n{'='*80}")
    print("🏁 INDIVIDUAL TEST SUITE COMPLETE")
    print('='*80)
    print(f"📊 Results: {passed} passed, {failed} failed")
    print(f"⏱️ Total time: {total_time:.1f} seconds")
    print(f"📁 All images saved to: artifacts/images/individual_tests/")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - CorePulse V4 is fully operational!")
        return 0
    else:
        print(f"⚠️ {failed} test(s) failed - check individual outputs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())