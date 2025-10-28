# dataloader file
from torch.utils.data import IterableDataset, DataLoader

class Dataset(IterableDataset):
    def __init__(self):
        # initialize dataset
        pass
        
    def __iter__(self,):
        # yield a sample
        pass


"""
design:
there will be shards of data files in a directory
individual shards will be loaded randomly or sequentially
individual data points will be loaded randomly or sequentially
next shard will be loaded, first one flushed out of memory
"""