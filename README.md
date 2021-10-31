
### Custom Dataset

你可以使用如下的形式创建数据集:
和`pytorch`自定义数据集的方式一样
```py
class custom_dataset(Dataset):
    def __init__(self):
        super().__init__()
        pass
    def __gettime__(self,index):
        pass
    def __len__(self):
        pass
```

接着划分你的数据集(任何方式),用`CustomDataset`包装一下即可:
```py
Custom = CustomDataset(
    train_dataset(transform=transform),
    test_dataset(transform=transform)
)
```

### Custom NetWork

对于使用到nn的模块你应该在`config.py`中配置你的`network_name`,并确保你的网络的正确性,并且你应该在`./network/main.py`文件中的元组添加你的自定义网络的名,example在`./network/custom.py`中


### Config Parameters


1. `train_param` 你得模型训练的参数
2. `pretrain_param`如果你选择了pretrain那么应该像Custom Network 一样自定义Autoencode.
3. `method_param` 针对不同的算法的参数,和`train_param`不同,`train_param`调节的是训练时的参数比如`lr`之类的
