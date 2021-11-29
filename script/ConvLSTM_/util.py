import numpy as np
import torch
def stack_uneven(arrays, fill_value=-9999):
        sizes = [a.shape for a in arrays]
        max_sizes = np.max(list(zip(*sizes)), -1)
        # The resultant array has stacked on the first dimension
        result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
        for i, a in enumerate(arrays):
          # The shape of this array `a`, turned into slices
          slices = tuple(slice(0,s) for s in sizes[i])
          # Overwrite a block slice of `result` with this array `a`
          result[i][slices] = a
        return result
    
def data_iter(data, batch_size=10, device="cpu"):
    num_examples=len(data)
    idx=np.arange(num_examples)
    np.random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        d=[data[i][:, 0:-1] for i in idx[i:min(i+10, num_examples)]]
        l=[data[i][:, -1] for i in idx[i:min(i+10, num_examples)]]
        length=np.array([e.shape[1] for e in d])
        sorted_index=np.argsort(-length)
        sorted_length=length[sorted_index]
        yield torch.tensor(stack_uneven(d)[sorted_index]).float().to(device), torch.tensor(np.stack(l))[:,0].float().to(device), torch.tensor(sorted_length).to(device)
        