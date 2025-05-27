#!/usr/bin/env python3
"""
Simple script to test the monitoring functionality
"""
import numpy as np
import torch
import time

print("Starting test process...")

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

# Create a large tensor
x = torch.randn(2000, 2000).to(device)
print(f"Created tensor of shape {x.shape} on {device}")

# Run matrix operations in a loop
iteration = 0
try:
    while True:
        # Matrix multiplication
        y = x @ x.T
        
        # Normalize
        x = y / y.sum()
        
        iteration += 1
        if iteration % 10 == 0:
            print(f"Completed {iteration} iterations")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Test process stopped by user")
