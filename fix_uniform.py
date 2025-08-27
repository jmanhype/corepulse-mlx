import re

tests = [
    "test_style_mixing.py",
    "test_concept_fusion.py", 
    "test_temporal_styles.py",
    "test_cultural_blending.py",
    "test_abstract_conceptualization.py"
]

for test_file in tests:
    with open(test_file, 'r') as f:
        content = f.read()
    
    # Fix pattern: mx.random.uniform((shape), low, high)
    pattern = r'mx\.random\.uniform\(\((batch, heads, seq_len, dim)\),\s*([\d.]+),\s*([\d.]+)\)'
    replacement = r'mx.random.uniform(shape=(batch, heads, seq_len, dim), low=\2, high=\3)'
    
    content = re.sub(pattern, replacement, content)
    
    # Also fix standalone shape tuples
    pattern2 = r'mx\.random\.uniform\((batch, heads, seq_len, dim)\)'
    replacement2 = r'mx.random.uniform(shape=(batch, heads, seq_len, dim))'
    
    content = re.sub(pattern2, replacement2, content)
    
    with open(test_file, 'w') as f:
        f.write(content)
    
    print(f"Fixed {test_file}")
