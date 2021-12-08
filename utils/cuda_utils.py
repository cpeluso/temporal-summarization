import gc
import torch

def clean_torch_memory():

  torch.cuda.empty_cache()
  gc.collect()

  t = torch.cuda.get_device_properties(0).total_memory
  r = torch.cuda.memory_reserved(0)
  a = torch.cuda.memory_allocated(0)
  print(r-a)