from typing import Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# contains code repurposed from https://github.com/magenta/symbolic-music-diffusion/blob/main/models/ncsn.py

def get_timestep_embedding(timesteps: torch.Tensor, embedding_channels: int):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_channels // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_channels % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    assert emb.shape == (timesteps.shape[0], embedding_channels)
    return emb

class TransformerBlock(nn.Module):
    def __init__(self, seq_len: int, embed_channels: int, mlp_dims: int, num_heads: int):
        super().__init__()
        self.embed_channels = embed_channels
        self.seq_len = seq_len
        self.mlp_dims = mlp_dims
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm([self.seq_len, self.embed_channels])
        self.self_attn = nn.MultiheadAttention(self.embed_channels, self.num_heads)
        self.emb_to_mlp = nn.Linear(self.embed_channels, self.mlp_dims)
        self.mlp_to_emb = nn.Linear(self.mlp_dims, self.embed_channels)

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.layer_norm(x)
        x, _ = self.self_attn(x, x, x)
        x = x + shortcut
        shortcut2 = x
        x = self.layer_norm(x)
        x = self.emb_to_mlp(x)
        x = F.gelu(x)
        x = self.mlp_to_emb(x)
        x = x + shortcut2
        return x

class NoiseEncoding(nn.Module):
    """Sinusoidal noise encoding block."""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels

    def forward(self, noise: torch.Tensor):
        # noise.shape = (batch_size, 1)
        # channels.shape = ()
        noise = noise.squeeze(-1)
        assert len(noise.shape) == 1
        half_dim = self.channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = emb.to(device=noise.device)
        emb = 5000 * noise[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        # print('inside noise encoding:', emb.shape)
        assert emb.shape == (noise.shape[0], self.channels)
        return emb

class DenseFiLM(nn.Module):
    def __init__(self, embedding_channels: int, out_channels: int):
        super().__init__()
        self.embed_channels = embedding_channels
        self.out_channels = out_channels
        self.emb_to_emb4 = nn.Linear(self.embed_channels, self.embed_channels * 4)
        self.emb4_to_emb4 = nn.Linear(self.embed_channels * 4, self.embed_channels * 4)
        self.emb4_to_scale = nn.Linear(self.embed_channels * 4, self.out_channels)
        self.emb4_to_shift = nn.Linear(self.embed_channels * 4, self.out_channels)
        self.noise_encoding = NoiseEncoding(self.embed_channels)

    def forward(self, position: torch.Tensor, sequence: bool = False):
        # position.shape = (batch_size,)
        position = position.unsqueeze(-1)
        # print(position.shape)
        assert len(position.shape) == 2
        pos_encoding = self.noise_encoding(position)
        # print('after noise encoding:', pos_encoding.shape)
        pos_encoding = self.emb_to_emb4(pos_encoding)
        # print('after emb to emb4:', pos_encoding.shape)
        pos_encoding = F.silu(pos_encoding)
        # print(pos_encoding.shape)
        pos_encoding = self.emb4_to_emb4(pos_encoding)

        if sequence:
            pos_encoding = pos_encoding[:, None, :]

        scale = self.emb4_to_scale(pos_encoding)
        shift = self.emb4_to_shift(pos_encoding)
        return scale, shift # should be (batch_size, 1, mlp_dims) each

class FeaturewiseAffine(nn.Module):
    """Feature-wise affine layer."""
    def __init__(self):
        super().__init__()

    def forward(self, x, scale: Union[float, torch.Tensor], shift: Union[float, torch.Tensor]):
        # print('[FA] x.shape:', x.shape)
        # # print('[FA] scale.shape:', scale.shape)
        # # print('[FA] shift.shape:', shift.shape)
        res =  scale * x + shift
        # print('[FA] res.shape:', res.shape)
        return res

class DenseResBlock(nn.Module):
    """Fully-connected residual block."""
    def __init__(self, seq_len: int, input_size: int, output_size: int):
        super().__init__()
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.featurewise_affine = FeaturewiseAffine()
        self.in_to_out = nn.Linear(self.input_size, self.output_size)
        self.in_to_out2 = nn.Linear(self.input_size, self.output_size)
        self.out_to_out = nn.Linear(self.output_size, self.output_size)
        self.layer_norm = nn.LayerNorm([self.seq_len, self.input_size])

    def forward(self, x: torch.Tensor, scale: Union[float, torch.Tensor] = 1., shift: Union[float, torch.Tensor] = 0.):
        output = self.layer_norm(x)
        # print('[dense res block] LN x shape:', output.shape)
        output = self.featurewise_affine(output, scale, shift)
        # print('[dense res block] FA x shape:', output.shape)
        output = F.silu(output)
        output = self.in_to_out(output)
        # print('[dense res block] I20 x shape:', output.shape)
        output = self.layer_norm(output)
        # print('[dense res block] LN2 x shape:', output.shape)
        output = self.featurewise_affine(output, scale, shift)
        # print('[dense res block] FA2 x shape:', output.shape)
        output = F.silu(output)
        output = self.out_to_out(output)
        # print('[dense res block] O2O x shape:', output.shape)

        shortcut = x
        if x.shape[-1] != self.output_size:
            shortcut = self.in_to_out2(x)

        return output + shortcut

class DenseFiLMResBlock(nn.Module):
    def __init__(self, seq_len: int, embed_channels: int, mlp_dims: int):
        super().__init__()
        self.embed_channels = embed_channels
        self.mlp_dims = mlp_dims
        self.seq_len = seq_len
        self.dense_film = DenseFiLM(self.embed_channels, self.mlp_dims)
        self.dense_res_block = DenseResBlock(self.seq_len, self.mlp_dims, self.mlp_dims)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        scale, shift = self.dense_film(t, sequence=True)
        # print('dense film x shape:', x.shape)
        x = self.dense_res_block(x, scale=scale, shift=shift)
        # print('dense res block x shape:', x.shape)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_layers = config.transformer.num_layers
        self.num_heads = config.transformer.num_heads
        self.num_mlp_layers = config.transformer.num_mlp_layers
        self.mlp_dims = config.transformer.mlp_dims
        self.input_channels = config.transformer.input_channels
        self.embed_channels = config.transformer.embed_channels
        self.seq_len = config.transformer.seq_len

        self.layer_norm_1 = nn.LayerNorm([self.seq_len, self.embed_channels])
        self.layer_norm_2 = nn.LayerNorm([self.seq_len, self.mlp_dims])
        self.in_to_emb = nn.Linear(self.input_channels, self.embed_channels)
        self.emb_to_mlp = nn.Linear(self.embed_channels, self.mlp_dims)
        self.mlp_to_in = nn.Linear(self.mlp_dims, self.input_channels)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(self.seq_len,
                                                                   self.embed_channels,
                                                                   self.mlp_dims,
                                                                   self.num_heads)
                                                  for _ in range(self.num_layers)])
        self.dense_film_res_blocks = nn.ModuleList([DenseFiLMResBlock(self.seq_len,
                                                                      self.embed_channels,
                                                                      self.mlp_dims)
                                                    for _ in range(self.num_mlp_layers)])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # x: (batch_size, seq_len, embed_dim)
        # t: (batch_size, 1)
        # print('initial x shape:', x.shape)
        temb = get_timestep_embedding(torch.arange(self.seq_len).to(x.device), self.embed_channels)
        temb = temb[None, :, :]
        assert temb.shape[1:] == (self.seq_len, self.embed_channels), temb.shape
        x = self.in_to_emb(x.float())
        # print('emb to emb x shape:', x.shape)
        x = x + temb
        x = self.transformer_blocks(x)
        # print('transformer blocks x shape:', x.shape)

        x = self.layer_norm_1(x)
        x = self.emb_to_mlp(x)
        # print('emb to mlp x shape:', x.shape)

        for dense_film_res_block in self.dense_film_res_blocks:
            x = dense_film_res_block(x, t)
            # print('dense film res x shape:', x.shape)

        x = self.layer_norm_2(x)
        x = self.mlp_to_in(x)
        return x

# def nonlinearity(x):
#     # swish
#     return x*torch.sigmoid(x)


# def Normalize(in_channels):
#     return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# class Upsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             self.conv = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x):
#         x = F.interpolate(
#             x, scale_factor=2.0, mode="nearest")
#         if self.with_conv:
#             x = self.conv(x)
#         return x


# class Downsample(nn.Module):
#     def __init__(self, in_channels, with_conv):
#         super().__init__()
#         self.with_conv = with_conv
#         if self.with_conv:
#             # no asymmetric padding in torch conv, must do it ourselves
#             self.conv = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=3,
#                                         stride=2,
#                                         padding=0)

#     def forward(self, x):
#         if self.with_conv:
#             pad = (0, 1, 0, 1)
#             x = F.pad(x, pad, mode="constant", value=0)
#             x = self.conv(x)
#         else:
#             x = F.avg_pool2d(x, kernel_size=2, stride=2)
#         return x


# class ResnetBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
#                  dropout, temb_channels=512):
#         super().__init__()
#         self.in_channels = in_channels
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut

#         self.norm1 = Normalize(in_channels)
#         self.conv1 = torch.nn.Conv2d(in_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         self.temb_proj = torch.nn.Linear(temb_channels,
#                                          out_channels)
#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv2d(out_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 self.conv_shortcut = torch.nn.Conv2d(in_channels,
#                                                      out_channels,
#                                                      kernel_size=3,
#                                                      stride=1,
#                                                      padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv2d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=1,
#                                                     padding=0)

#     def forward(self, x, temb):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)

#         h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)

#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)

#         return x+h


# class AttnBlock(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels

#         self.norm = Normalize(in_channels)
#         self.q = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.k = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.v = torch.nn.Conv2d(in_channels,
#                                  in_channels,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0)
#         self.proj_out = torch.nn.Conv2d(in_channels,
#                                         in_channels,
#                                         kernel_size=1,
#                                         stride=1,
#                                         padding=0)

#     def forward(self, x):
#         h_ = x
#         h_ = self.norm(h_)
#         q = self.q(h_)
#         k = self.k(h_)
#         v = self.v(h_)

#         # compute attention
#         b, c, h, w = q.shape
#         q = q.reshape(b, c, h*w)
#         q = q.permute(0, 2, 1)   # b,hw,c
#         k = k.reshape(b, c, h*w)  # b,c,hw
#         w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
#         w_ = w_ * (int(c)**(-0.5))
#         w_ = F.softmax(w_, dim=2)

#         # attend to values
#         v = v.reshape(b, c, h*w)
#         w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
#         # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
#         h_ = torch.bmm(v, w_)
#         h_ = h_.reshape(b, c, h, w)

#         h_ = self.proj_out(h_)

#         return x+h_

# class Model(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
#         num_res_blocks = config.model.num_res_blocks
#         attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         in_channels = config.model.in_channels
#         resolution = config.data.image_size
#         resamp_with_conv = config.model.resamp_with_conv
#         num_timesteps = config.diffusion.num_diffusion_timesteps

#         if config.model.type == 'bayesian':
#             self.logvar = nn.Parameter(torch.zeros(num_timesteps))

#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels

#         # timestep embedding
#         self.temb = nn.Module()
#         self.temb.dense = nn.ModuleList([
#             torch.nn.Linear(self.ch,
#                             self.temb_ch),
#             torch.nn.Linear(self.temb_ch,
#                             self.temb_ch),
#         ])

#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+ch_mult
#         self.down = nn.ModuleList()
#         block_in = None
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up)  # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x, t):
#         assert x.shape[2] == x.shape[3] == self.resolution

#         # timestep embedding
#         temb = get_timestep_embedding(t, self.ch)
#         temb = self.temb.dense[0](temb)
#         temb = nonlinearity(temb)
#         temb = self.temb.dense[1](temb)

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](
#                     torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h

# class OldModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
#         num_res_blocks = config.model.num_res_blocks
#         attn_resolutions = config.model.attn_resolutions
#         dropout = config.model.dropout
#         in_channels = config.model.in_channels
#         resolution = config.data.image_size
#         resamp_with_conv = config.model.resamp_with_conv
#         num_timesteps = config.diffusion.num_diffusion_timesteps

#         if config.model.type == 'bayesian':
#             self.logvar = nn.Parameter(torch.zeros(num_timesteps))

#         self.ch = ch
#         self.temb_ch = self.ch*4
#         self.num_resolutions = len(ch_mult)
#         self.num_res_blocks = num_res_blocks
#         self.resolution = resolution
#         self.in_channels = in_channels

#         # timestep embedding
#         self.temb = nn.Module()
#         self.temb.dense = nn.ModuleList([
#             torch.nn.Linear(self.ch,
#                             self.temb_ch),
#             torch.nn.Linear(self.temb_ch,
#                             self.temb_ch),
#         ])

#         # downsampling
#         self.conv_in = torch.nn.Conv2d(in_channels,
#                                        self.ch,
#                                        kernel_size=3,
#                                        stride=1,
#                                        padding=1)

#         curr_res = resolution
#         in_ch_mult = (1,)+ch_mult
#         self.down = nn.ModuleList()
#         block_in = None
#         for i_level in range(self.num_resolutions):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_in = ch*in_ch_mult[i_level]
#             block_out = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks):
#                 block.append(ResnetBlock(in_channels=block_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             down = nn.Module()
#             down.block = block
#             down.attn = attn
#             if i_level != self.num_resolutions-1:
#                 down.downsample = Downsample(block_in, resamp_with_conv)
#                 curr_res = curr_res // 2
#             self.down.append(down)

#         # middle
#         self.mid = nn.Module()
#         self.mid.block_1 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)
#         self.mid.attn_1 = AttnBlock(block_in)
#         self.mid.block_2 = ResnetBlock(in_channels=block_in,
#                                        out_channels=block_in,
#                                        temb_channels=self.temb_ch,
#                                        dropout=dropout)

#         # upsampling
#         self.up = nn.ModuleList()
#         for i_level in reversed(range(self.num_resolutions)):
#             block = nn.ModuleList()
#             attn = nn.ModuleList()
#             block_out = ch*ch_mult[i_level]
#             skip_in = ch*ch_mult[i_level]
#             for i_block in range(self.num_res_blocks+1):
#                 if i_block == self.num_res_blocks:
#                     skip_in = ch*in_ch_mult[i_level]
#                 block.append(ResnetBlock(in_channels=block_in+skip_in,
#                                          out_channels=block_out,
#                                          temb_channels=self.temb_ch,
#                                          dropout=dropout))
#                 block_in = block_out
#                 if curr_res in attn_resolutions:
#                     attn.append(AttnBlock(block_in))
#             up = nn.Module()
#             up.block = block
#             up.attn = attn
#             if i_level != 0:
#                 up.upsample = Upsample(block_in, resamp_with_conv)
#                 curr_res = curr_res * 2
#             self.up.insert(0, up)  # prepend to get consistent order

#         # end
#         self.norm_out = Normalize(block_in)
#         self.conv_out = torch.nn.Conv2d(block_in,
#                                         out_ch,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1)

#     def forward(self, x, t):
#         assert x.shape[2] == x.shape[3] == self.resolution

#         # timestep embedding
#         temb = get_timestep_embedding(t, self.ch)
#         temb = self.temb.dense[0](temb)
#         temb = nonlinearity(temb)
#         temb = self.temb.dense[1](temb)

#         # downsampling
#         hs = [self.conv_in(x)]
#         for i_level in range(self.num_resolutions):
#             for i_block in range(self.num_res_blocks):
#                 h = self.down[i_level].block[i_block](hs[-1], temb)
#                 if len(self.down[i_level].attn) > 0:
#                     h = self.down[i_level].attn[i_block](h)
#                 hs.append(h)
#             if i_level != self.num_resolutions-1:
#                 hs.append(self.down[i_level].downsample(hs[-1]))

#         # middle
#         h = hs[-1]
#         h = self.mid.block_1(h, temb)
#         h = self.mid.attn_1(h)
#         h = self.mid.block_2(h, temb)

#         # upsampling
#         for i_level in reversed(range(self.num_resolutions)):
#             for i_block in range(self.num_res_blocks+1):
#                 h = self.up[i_level].block[i_block](
#                     torch.cat([h, hs.pop()], dim=1), temb)
#                 if len(self.up[i_level].attn) > 0:
#                     h = self.up[i_level].attn[i_block](h)
#             if i_level != 0:
#                 h = self.up[i_level].upsample(h)

#         # end
#         h = self.norm_out(h)
#         h = nonlinearity(h)
#         h = self.conv_out(h)
#         return h
