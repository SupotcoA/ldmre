import torch
import numpy as np
from utils import Logger, check_ae


def train(model,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          logger: Logger):
    if train_config['train_steps']<=0:
        model.eval()
        final_eval_generation(model, logger, verbose=train_config['train_steps']==0)
        return

    if train_config['need_check']:
        for [x0, cls] in train_dataset:
            x0 = x0.to(model.device)
            imgs=check_ae(model, x0)
            logger.log_images(imgs, 3, 3, "check_ae")
            imgs = model.eval_solver(x0, 9)
            logger.log_images(imgs, 3, 3, "eval_solver")
            break
    
    logger.train_start()
    for [x0, cls] in train_dataset:
        model.train()
        x0, cls = x0.to(model.device), cls.to(model.device)
        x0 = model.ae.preprocess(x0)
        optim.zero_grad()
        loss = model.train_step(x0, cls)
        # if loss.detach().cpu().item()>0.725:
        #     logger.train_step(0.725)
        # else:
        loss.backward()
        optim.step()
        model.ema_update(train_config['ema_decay'])
        logger.train_step(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            model.eval()
            eval_generation(model, logger)
        if train_config['use_ema'] \
            and logger.step == train_config['train_steps']-train_config['ema_steps']:
            model.init_ema()
        if logger.step == train_config['train_steps']:
            model.eval()
            logger.train_end()
            test(model, logger, train_dataset, num_test_steps=1000)
            if train_config['use_ema']:
                model.apply_ema()
                test(model, logger, train_dataset, num_test_steps=1000)
            final_eval_generation(model, logger)
            if train_config['save']:
                logger.log_net(model.net.cpu(),f"edm_{logger.step}_{logger.model_name}")
            break

@torch.no_grad()
def eval_generation(model, logger):
    logger.generation_start()
    for cls in range(5):
        imgs = model.conditional_generation(cls,
                                            guidance_scale=1,
                                            batch_size=9,
                                            use_2nd_order=False,
                                            n_steps=512,
                                            )
        logger.log_images(imgs, 3, 3, f"step_{logger.step}_cls_{cls}_cfg_1")
    logger.generation_end()
    logger.train_resume()

@torch.no_grad()
def final_eval_generation(model, logger, verbose=False):
    logger.generation_start()
    if verbose:
        for cfg in [1,2,3,4]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    cfg_zero_star=True
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_czs_cls_{cls}_cfg_{cfg}")
        # for cfg in [1,2,3]:
        #     for cls in range(5):
        #         imgs = model.conditional_generation(cls,
        #                                             cfg,
        #                                             16,
        #                                             use_2nd_order=True,
        #                                             n_steps=512,
        #                                             cfg_zero_star=True
        #                                             )
        #         logger.log_images(imgs, 4, 4, f"step_{logger.step}_2nd_order_cls_{cls}_cfg_{cfg}")
        for cfg in [1,2,3]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    cfg_zero_star=(True,False),
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_zero_init_only_cls_{cls}_cfg_{cfg}")
        for cfg in [1,2,3]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=1024,
                                                    cfg_zero_star=False
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_long_chain_cls_{cls}_cfg_{cfg}")
        # for cfg in [1,2,3]:
        #     for cls in range(5):
        #         imgs, imgs0 = model.conditional_generation_with_middle_steps(cls,
        #                                                                     cfg,
        #                                                                     use_2nd_order=False,
        #                                                                     batch_size=4,
        #                                                                     n_steps=512,
        #                                                                     n_middle_steps=8)
        #         logger.log_images(imgs, 4, 8, f"step_{logger.step}_mid_cls_{cls}_cfg_{cfg}")
        #         logger.log_images(imgs0, 4, 8, f"step_{logger.step}_mid_pred_cls_{cls}_cfg_{cfg}")
    else:
        for cfg in [1,2,3]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    12,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    cfg_zero_star=True
                                                    )
                logger.log_images(imgs, 4, 3, f"step_{logger.step}_czs_cls_{cls}_cfg_{cfg}")
    logger.generation_end()

@torch.no_grad()
def test(model,
         logger,
         train_dataset,
         num_test_steps=1000,
         ):
    model.eval()
    acc_loss = []
    step = 0
    for [x0, cls] in train_dataset:
        step += 1
        loss = model.train_step(x0.to(model.device), cls.to(model.device))
        acc_loss.append(loss.cpu().item())
        if step >= num_test_steps:
            break
    acc_loss=np.asarray(acc_loss)
    info = f"Test\n" \
           + f"loss:{acc_loss.mean():.4f}+-{acc_loss.std():.4f}" 
    print(info)
    logger.log_text(info, "train_log", newline=True)
