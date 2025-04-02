import torch
from utils import Logger, check_ae


def train(model,
          optim,
          lr_scheduler,
          train_config,
          train_dataset,
          logger: Logger):
    if train_config['train_steps']==0:
        model.eval()
        final_eval_generation(model, logger, verbose=True)
        return

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
        loss.backward()
        optim.step()
        logger.train_step(loss.detach().cpu().item())
        if train_config['use_lr_scheduler']:
            lr_scheduler.step()
        if logger.step % train_config['eval_every_n_steps'] == 0:
            model.eval()
            eval_generation(model, logger)
        if logger.step == train_config['train_steps']:
            model.eval()
            logger.train_end()
            final_eval_generation(model, logger)
            if train_config['save']:
                logger.log_net(model.net.cpu(),f"edm_{logger.step}")
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
        for cfg in [1,2,3,4,5]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_cls_{cls}_cfg_{cfg}")
        for cfg in [1,3,5]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=True,
                                                    n_steps=512,
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_2nd_order_cls_{cls}_cfg_{cfg}")
        for cfg in [1,3,5]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    16,
                                                    use_2nd_order=False,
                                                    n_steps=1024,
                                                    )
                logger.log_images(imgs, 4, 4, f"step_{logger.step}_long_chain_cls_{cls}_cfg_{cfg}")
        for cfg in [1,3,5]:
            for cls in range(5):
                imgs, imgs0 = model.conditional_generation_with_middle_steps(cls,
                                                                            cfg,
                                                                            use_2nd_order=False,
                                                                            batch_size=4,
                                                                            n_steps=512,
                                                                            n_middle_steps=8)
                logger.log_images(imgs, 4, 8, f"step_{logger.step}_mid_cls_{cls}_cfg_{cfg}")
                logger.log_images(imgs0, 4, 8, f"step_{logger.step}_mid_pred_cls_{cls}_cfg_{cfg}")
    else:
        for cfg in [1,3,5]:
            for cls in range(5):
                imgs = model.conditional_generation(cls,
                                                    cfg,
                                                    12,
                                                    use_2nd_order=False,
                                                    n_steps=512,
                                                    )
                logger.log_images(imgs, 4, 3, f"step_{logger.step}_cls_{cls}_cfg_{cfg}")
    logger.generation_end()

@torch.no_grad()
def test(model,
         train_config,
         test_dataset):
    model.eval()
    acc_loss = 0
    step = 0
    for [x0, cls] in test_dataset:
        step += 1
        loss = model.train_step(x0.to(model.device), cls.to(model.device))
        acc_loss += loss.cpu().item()
    info = f"Test step\n" \
           + f"loss:{acc_loss / step:.4f}\n"
    print(info)
    with open(train_config['log_path'], 'a') as f:
        f.write(info)
