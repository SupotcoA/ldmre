import torch
from torch import nn
import torch.nn.functional as F
from embedding import TimeEmbedding, ClassEmbedding, FourierNoiseEmbedding
from resblock import UpSample, DownSample, ConvNeXtBlock, ResBlock
from norm import AdaptiveLayerNorm, DyT
from attn import MultiHeadAttnBlock


class UneXt(nn.Module):

    # https://github.com/CompVis/taming-transformers/
    # blob/master/taming/modules/diffusionmodules/model.py#L195

    def __init__(self,
                 in_channels=4,
                 emb_dim=64,
                 channels_mult=(1, 1, 2, 4),
                 num_res_blocks=(2, 2, 2, 2),
                 expansion_ratio=4,
                 cond_dim=64,
                 **ignoredkeys):
        super().__init__()

        def make_res_block(*args, **kwargs):
            return ConvNeXtBlock(*args, **kwargs)

        base_dim = emb_dim
        ch_mult = channels_mult

        self.c_dim = cond_dim
        assert ch_mult[0]==1
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        assert len(num_res_blocks)==self.num_resolutions
        self.in_channels = in_channels

        # downsampling
        curr_res = 32
        curr_ch = base_dim
        self.conv_in = nn.Conv2d(in_channels,
                                    curr_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding='same')
        self.norm_in = nn.GroupNorm(16, curr_ch)

        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(make_res_block(emd_dim=curr_ch,
                                            expansion_ratio=expansion_ratio,
                                            cond_dim=cond_dim
                                            ))
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(curr_ch,
                                             ch_mult[i_level+1]*base_dim)
                curr_res = curr_res // 2
                curr_ch = ch_mult[i_level+1]*base_dim
            self.down.append(down)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(make_res_block(emd_dim=curr_ch,
                                            expansion_ratio=expansion_ratio,
                                            cond_dim=cond_dim
                                            ))
            up = nn.Module()
            up.block = block
            if i_level != 0:
                up.upsample = UpSample(curr_ch, ch_mult[i_level-1]*base_dim)
                curr_res = curr_res * 2
                curr_ch = ch_mult[i_level-1]*base_dim
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(16, curr_ch)

        self.conv_out = nn.Conv2d(curr_ch,
                                in_channels,
                                kernel_size=3,
                                stride=1,
                                padding='same')

    def forward(self, x, c=None):
        # downsampling
        hs = []
        h = self.norm_in(self.conv_in(x))
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, c)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                self.down[i_level].downsample(h)
        h = h * (1-1/1.414)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](
                    h+hs.pop()/1.414, c)  # ??
                
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        assert len(hs)==0, f"len(hs) = {len(hs)}"
        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Unet(nn.Module):

    # https://github.com/CompVis/taming-transformers/
    # blob/master/taming/modules/diffusionmodules/model.py#L195

    def __init__(self,
                 in_channels=4,
                 emb_dim=96,
                 channels_mult=(1, 2, 4),
                 num_res_blocks=2,
                 cond_dim=None,
                 attn_resolutions=(8, 16),
                 out_channels=None,
                 **ignoredkeys):
        super().__init__()

        def make_res_block(in_channels,
                           out_channels,
                           cond_dim):
            return ResBlock(in_dim=in_channels,
                            out_dim=out_channels,
                            cond_dim=cond_dim
                            )

        ch = emb_dim
        ch_mult = channels_mult
        out_ch = out_channels if out_channels is not None else in_channels

        self.ch = ch
        self.cond_dim = cond_dim
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels

        # norm before conv in?
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = 32
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(make_res_block(in_channels=block_in,
                                               out_channels=block_out,
                                               cond_dim=cond_dim))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = DownSample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_res_block(in_channels=block_in,
                                             out_channels=block_in,
                                             cond_dim=cond_dim)
        self.mid.attn_1 = MultiHeadAttnBlock(block_in)
        self.mid.block_2 = make_res_block(in_channels=block_in,
                                             out_channels=block_in,
                                             cond_dim=cond_dim)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(make_res_block(in_channels=block_in + skip_in,
                                               out_channels=block_out,
                                               cond_dim=cond_dim))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(MultiHeadAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpSample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(32, block_in)

        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, c=None):

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], c)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, c)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, c)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), c)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        assert len(hs)==0, f"len(hs) = {len(hs)}"
        # end
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class UnetWarp(nn.Module):
    def __init__(self,
                 unet_config):
        super().__init__()
        self.net = Unet(**unet_config)
        self.t_emd = FourierNoiseEmbedding(192)
        self.cls_emb = ClassEmbedding(unet_config['n_class'],
                                      unet_config['cond_dim'])
        self.cond_mlp = nn.Sequential(nn.Linear(192,
                                                unet_config['cond_dim']),
                                        nn.SiLU(),
                                        nn.Linear(unet_config['cond_dim'],
                                                unet_config['cond_dim'])
                                      )
    
    # @torch.compile
    def forward(self, x, t, cls, mask=None):
        # t = cnoise for EDM
        t_emb = self.t_emd(t)
        cls_emb = self.cls_emb(cls)
        if mask is not None:
            cls_emb[mask] = 0
        c = F.silu(self.cond_mlp(t_emb)+cls_emb)
        return self.net(x,c)