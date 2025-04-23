# filepath: minimal_test.py
from accelerate import Accelerator
print("Initializing Accelerator...")
accelerator = Accelerator()
print(f"Accelerator initialized. Device: {accelerator.device}, Process Index: {accelerator.process_index}, Num Processes: {accelerator.num_processes}")
print("Minimal test finished.")