from joblib import Parallel, delayed
import torch
import pdb

def parallel(fn, n_jobs=-1, *fn_args):
  outs = Parallel(n_jobs=n_jobs, prefer='threads')(delayed(fn)(*args) for args in zip(*fn_args))
  #outs = Parallel(n_jobs=n_jobs)(delayed(fn)(*args) for args in zip(*fn_args))
  return outs

def get_parallel_list(x, length, force=False):
  if force:
    return [x for _ in range(length)]
  else:
    return [x for _ in range(length)] if not isinstance(x, list) else x
  
get_tensor_items = lambda x: [X.item() if isinstance(X, torch.Tensor) else X for X in x]  
