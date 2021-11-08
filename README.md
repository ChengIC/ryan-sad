
### Custom Dataset and network 

class custom_dataset(Dataset):
    def __init__(self):
        super().__init__()
        pass
    def __gettime__(self,index):
        pass
    def __len__(self):
        pass


Custom = CustomDataset(
    train_dataset(transform=transform),
    test_dataset(transform=transform)
)


### Custom NetWork

Example is in `./network/custom.py`

### Config Parameters

Example is in config.py

