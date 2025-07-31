import torch
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA version:", torch.version.cuda)
print("Torch compiled with:", torch.__config__.show())
