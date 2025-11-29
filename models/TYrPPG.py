import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
import math
from einops import rearrange

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5,dim=24):
        super(Fusion_Stem, self).__init__()


        self.stem11 = nn.Sequential(nn.Conv2d(3, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, dim//2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(dim//2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2

        return x
    

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class Frequencydomain_FFN(nn.Module):
    def __init__(self, dim, mlp_ratio):
        super().__init__()

        self.scale = 0.02
        self.dim = dim * mlp_ratio

        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))

        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),  
            nn.BatchNorm1d(dim),
        )


    def forward(self, x):
        B, N, C = x.shape
  
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        x_fre = torch.fft.fft(x, dim=1, norm='ortho') # FFT on N dimension

        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r) - \
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.i) + \
            self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r) + \
            torch.einsum('bnc,cc->bnc', x_fre.real, self.i) + \
            self.ib
        )

        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)

        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x

class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        n, d, h, w, c = x.size() 
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out
        # n, d, c = x.size() 
        # fold = c // self.fold_div
        # out = torch.zeros_like(x)
        # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        # return out
        
        # nt, c, h, w = x.size()
        # n_batch = nt // self.n_segment
        # x = x.view(n_batch, self.n_segment, c, h, w)
        # fold = c // self.fold_div
        # out = torch.zeros_like(x)
        # out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        # out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        # out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        # return out.view(nt, c, h, w)
    
# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, output_size)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

class GatedCNNBlock(nn.Module):
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv3d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x # [B, T, H, W, C]
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 4, 1, 2, 3) # [B, T, H, W, C] -> [B, C, T, H, W]
        c = self.conv(c)
        c = c.permute(0, 2, 3, 4, 1) # [B, C, T, H, W] -> [B, T, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut
        # shortcut = x  # [B, T, C]
        # x = self.norm(x)
        # g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        # c = c.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
   
        # c = self.conv(c)  
        # c = c.permute(0, 2, 1)  # [B, C, T] -> [B, T, C]
     
        # x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        # x = self.drop_path(x)
        # return x + shortcut
            


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x    

class TYrPPG(nn.Module):
    def __init__(self, 
                 depth=24, 
                 embed_dim=96, 
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 n_segment=10,  
                 fold_div=3,    
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        super().__init__()

        self.embed_dim = embed_dim
        self.n_segment = n_segment
        self.fold_div = fold_div
        
        self.mlp = Frequencydomain_FFN(dim=embed_dim, mlp_ratio=mlp_ratio)

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()
        self.tsm = TSM(n_segment=n_segment, fold_div=fold_div) 

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([GatedCNNBlock(
            dim = embed_dim, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            kernel_size=7,
            conv_ratio=1.0,
            drop_path=inter_dpr[i],)
                                     
        for i in range(depth)])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        # 初始化
        self.apply(segm_init_weights)
        

    def forward(self, x):
        B, D, C, H, W = x.shape
        
        x = self.Fusion_Stem(x)    #[N*D C H/8 W/8]
        x = x.view(B,D,self.embed_dim//4,H//8,W//8).permute(0,2,1,3,4) #B D C H W -> B C D H W
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask
        
        x = x.permute(0,2,3,4,1) # B D H W C
        # x = torch.mean(x, 4)
        # x = torch.mean(x, 3)
        # x = rearrange(x, 'b c t -> b t c')
        
        for blk in self.blocks:
            short_cut = x
            x = self.tsm(x)
            x = blk(x+short_cut)
            
        x = x.permute(0,4,1,2,3)
        
        x = torch.mean(x, 4)
        x = torch.mean(x, 3)
        x = rearrange(x, 'b c t -> b t c')
        
        rPPG = x.permute(0,2,1) 
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)    #[N, 1, D]
        rPPG = rPPG.squeeze(1)

        return rPPG    
    
    
    