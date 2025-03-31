import torch
from diffusion_model_edm_unet import DiffusionModel
from utils import calculate_num_params


def build_model(logger,
                net_config,
                diffusion_config,
                train_config):
    model = DiffusionModel(net_config,
                           diffusion_config,)
    ae_params = calculate_num_params(model.ae)
    net_params = calculate_num_params(model.net)
    info = f"AE params: {ae_params:,}, net params: {net_params:,}"
    print(info)
    logger.log_text(info, "config", newline=True)

    if train_config['pretrained']:
        sd = torch.load(train_config['pretrained'], map_location=torch.device('cpu'))
        model.load_state_dict(sd, strict=True)
    if torch.cuda.is_available():
        model.cuda()
        print("running on cuda")
    else:
        print("running on cpu!")
    optim = torch.optim.Adam(model.net.parameters(),
                             lr=train_config['base_learning_rate'])
    if train_config['use_lr_scheduler']:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim,
                                                                  T_max=train_config['train_steps'],
                                                                  eta_min=train_config['min_learning_rate'])
    else:
        lr_scheduler = None
    return model.eval(), optim, lr_scheduler
