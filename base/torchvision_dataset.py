from .base_dataset import BaseADDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self):
        super().__init__()
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader

'''

Custom Dataset Example

'''
class CustomDataset(TorchvisionDataset):
    def __init__(self,train_dataset: Dataset
                    ,test_dataset:Dataset):
        super().__init__()
        self.train_set = train_dataset
        self.test_set = test_dataset
