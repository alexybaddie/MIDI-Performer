import torch
print(torch.cuda.is_available())  # Should print: True
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
