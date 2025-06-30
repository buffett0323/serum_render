#!/usr/bin/env python
"""
Test script for multiprocessing ADSR rendering
"""

import os
import sys
import subprocess

def test_multiprocessing():
    """Test the multiprocessing functionality with a small dataset"""
    
    # Test with minimal parameters
    cmd = [
        sys.executable, "render_t_adsr_c.py",
        "--midi_amount", "2",  # Only 2 MIDI files
        "--num_processes", "2",  # Use 2 processes
        "--output_dir", "/tmp/test_adsr_output"
    ]
    
    print("Testing multiprocessing ADSR rendering...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✅ Multiprocessing test passed!")
            print("Output:")
            print(result.stdout)
        else:
            print("❌ Multiprocessing test failed!")
            print("Error output:")
            print(result.stderr)
            print("Standard output:")
            print(result.stdout)
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    test_multiprocessing() 