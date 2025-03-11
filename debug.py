import torch
print(torch.version.cuda)  # Should be 12.1
print(torch.cuda.is_available())  # Should return True
