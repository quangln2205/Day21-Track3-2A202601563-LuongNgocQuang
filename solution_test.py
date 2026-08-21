#!/usr/bin/env python3
"""
Test script to verify the environment works correctly
"""

import sys
print("Python version:", sys.version)

try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except ImportError as e:
    print("PyTorch import error:", e)

try:
    import transformers
    print("Transformers version:", transformers.__version__)
except ImportError as e:
    print("Transformers import error:", e)

try:
    import peft
    print("PEFT version:", peft.__version__)
except ImportError as e:
    print("PEFT import error:", e)

try:
    import trl
    print("TRL version:", trl.__version__)
except ImportError as e:
    print("TRL import error:", e)

print("Environment test completed successfully!")