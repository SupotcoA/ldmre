import sys
sys.path.append('/kaggle/working/ldmre/ldm')
import os
import torch
import shutil
from utils import Logger
from train import train
from build_model import build_model
from data import build_cached_dataset


global_config = dict(
    ver="rldm init",
    description="init",
    outcome_root="/kaggle/working",
)
global_config["outcome_dir_root"] = os.path.join(global_config["outcome_root"],
                                                 global_config["ver"])

net_config = dict(
    in_channels=4,
    emb_dim=96,
    channels_mult=(1, 2, 4),
    num_res_blocks=2,
    attn_resolutions=(8, 16),
    cond_dim=192,
    n_class=5
)

diffusion_config = dict(

)

train_config = dict(
    train_steps=100000,
    log_every_n_steps=1000,
    eval_every_n_steps=10000,
    save=True,
    pretrained=None,
    batch_size=32,
    base_learning_rate=1e-4,
    min_learning_rate=1e-5,
    use_lr_scheduler=True,
)

dataset_paths={'afhq':'/kaggle/input/afhq-512',
               'ffhq':'/kaggle/input/flickrfaceshq-dataset-nvidia-resized-256px',
               'celebahq':'/kaggle/input/celebahq256-images-only',
               'fa':'/kaggle/input/face-attributes-grouped',
               'animestyle':'/kaggle/input/gananime-lite',
               'animefaces':'/kaggle/input/another-anime-face-dataset',
              }

data_config = dict(
    image_size=256,
    batch_size=train_config['batch_size'],
    ae_batch_size=48,
    split=1.00,
    data_paths=dataset_paths,
    enc_path=os.path.join(global_config["outcome_dir_root"], "enc"),
    enc_inp_path='/kaggle/input/sd-vae-ft-ema-f8-256-faces6-enc',
    dataset_names=['afhq', 'ffhq', 'celebahq', 'fa', 'animestyle', 'animefaces'],
    ignored_dataset=['fa'],
)
data_config['n_class'] = len(data_config['dataset_names']) - len(data_config['ignored_dataset'])
assert data_config['n_class'] == net_config['n_class']


######################################################################


logger = Logger(log_every_n_steps=train_config['log_every_n_steps'],
                log_root=global_config["outcome_dir_root"])

logger.log_text(str(global_config), "config")
logger.log_text(str(net_config), "config", newline=True)
logger.log_text(str(diffusion_config), "config", newline=True)
logger.log_text(str(train_config), "config", newline=True)
logger.log_text(str(data_config), "config", newline=True)

torch.manual_seed(42)

train_dataset = build_cached_dataset(data_config)

model, optim, lr_scheduler = build_model(logger,
                                         net_config,
                                         diffusion_config,
                                         train_config)

try:
    train(model, optim, lr_scheduler, train_config, train_dataset, logger)
except Exception as e:
    info = f"Exception: {str(e)} \n"+\
            f"Step: {logger.step}"
    print(info)
    logger.log_text(info, "error")
finally:
    shutil.make_archive(global_config["outcome_dir_root"],
                        'zip',
                        global_config["outcome_dir_root"])





