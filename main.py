from DeepSAD import *
from baselines.isoforest import*
from baselines.kde import*
from baselines.ocsvm import*
from baselines.ssad import*
from baselines.SemiDGM import *
from config import *

Implemented_Methods  = ['sad','kde','isoforest','ocsvm','semi-dgm','ssad']
Implemented_Pretrain = ['semi-dgm','sad']
NotNetwork           = ['kde','ocsvm','ssad','isoforest']
if task not in Implemented_Methods:
    raise RuntimeError(
        'Not Implemented Method !!!'
    )
if Pretrain and task not in Implemented_Pretrain:
    raise RuntimeError(
        'Not Implemented Pretrain Method !!!'
    )

Methods = {
    'sad':DeepSAD,
    'kde':KDE,
    'isoforest':IsoForest,
    'ocsvm':OCSVM,
    'semi-dgm':SemiDeepGenerativeModel,
    'ssad':SSAD,
}

Task = Methods[task](**method_params)
def pretrain_sad():
    Task.pretrain(**pretrain_params)
    Task.init_network_weights_from_pretraining()
def pretrain_dgm():
    Task.set_vae(ae_net_name)


pretrain_map = {
    'sad':pretrain_sad,
    'dgm':pretrain_dgm
}

save_ae = False
if Pretrain:
    save_ae = True
    pretrain_map[task]()


if not(task in NotNetwork):
    Task.set_network(net_name)
    
Task.train(**train_params)

Task.save_model(save_path,save_ae)