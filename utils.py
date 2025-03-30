import torch
import numpy as np
import time
import cv2
import os
import matplotlib.pyplot as plt

@torch.no_grad()
def calculate_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def tensor2bgr(tensor):
    imgs = torch.clip(torch.permute(tensor, [0, 2, 3, 1]).cpu().add(1).mul(127.5), 0, 255)
    return imgs.numpy().astype(np.uint8)[:, :, :, ::-1]

class Logger:
    def __init__(self,
                 log_every_n_steps=100,
                 log_root=None):
        self.log_root = log_root
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.step = 0
        self.log_every_n_steps = log_every_n_steps
        self.time = 0
        self.loss_accum = 0
        self.train_loss = []
        self.train_memory = []
        
    def train_start(self):
        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)
        self.time = time.time()
        torch.cuda.reset_peak_memory_stats()
    
    def train_resume(self):
        self.time = time.time()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def train_step(self, loss):
        self.step += 1
        self.loss_accum += loss
        self.train_loss.append(loss)
        
        # Record peak memory for this step (in GB)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.train_memory.append(peak_mem)
        torch.cuda.reset_peak_memory_stats()
        
        if self.step % self.log_every_n_steps == 0:
            dt = time.time() - self.time
            current_peak_mem = max(self.train_memory[-self.log_every_n_steps:])
            info = (f"Train step {self.step}\n"
                   f"loss: {self.loss_accum/self.log_every_n_steps:.4f}\n"
                   f"tps: {dt/self.log_every_n_steps:.1f}\n"
                   f"peak GPU mem: {current_peak_mem:.1f}GB\n")
            
            print(info)
            self.log_text(info, "train_log")
            self.time = time.time()
            self.loss_accum = 0
    
    def train_end(self):
        # plot train loss curve and memory usage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curve
        ax1.plot(self.train_loss)
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss")
        ax1.set_title("Train Loss")
        
        # Memory curve
        ax2.plot(self.train_memory)
        ax2.set_xlabel("step")
        ax2.set_ylabel("GPU memory (GB)")
        ax2.set_title("Peak GPU Memory")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_root, "train_stats.png"))
        plt.close()
    
    def log_text(self, text, fname="log", newline=True):
        if newline:
            text = "\n"+text
        path = os.path.join(self.log_root, f"{fname}.txt")
        with open(path, 'a') as f:
            f.write(text)

    def generation_start(self):
        self.eval_time = time.time()
        torch.cuda.reset_peak_memory_stats()

    def generation_end(self):
        dt = time.time() - self.eval_time
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        info = (f"Generation time: {dt:.0f}s\n"
                f"Peak GPU memory: {peak_mem:.1f}GB\n")
        
        print(info)
        self.log_text(info, "train_log")

    @torch.no_grad()
    def log_images(self,imgs,nrow,ncol,fname):
        # imgs: torch.Tensor shape(B=nrow*ncol, C, H, W)
        imgs = tensor2bgr(imgs)
        h, w, c = imgs.shape[1:]
        base = np.zeros((h * nrow, w * ncol, c), dtype=np.uint8)
        for i in range(nrow):
            for j in range(ncol):
                base[i * h:i * h + h, j * w:j * w + w, :] = imgs[i * ncol + j]
        fp = os.path.join(self.log_root, f"{fname}.png")
        cv2.imwrite(fp, base)
    
    def log_net(self,net,name):
        torch.save(net.state_dict(),os.path.join(self.log_root,f"{name}.pth"))

@torch.no_grad()
def check_ae(model,x0, batch_size=9):
    # check if the decoder is working
    imgs=model.decode(x0[:batch_size], need_postprocess=False)
    return imgs
