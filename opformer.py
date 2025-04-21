import einops
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = "cuda"

class SpectralConv2d_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, nhead):
        super(SpectralConv2d_Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = nhead
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter(self.scale * (torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat)))

    def compl_mul2d(self, input, weights):
        # (batch, num_patches, in_channel, x,y ), (in_channel, out_channel, x,y, nhead) -> (batch, num_patches, out_channel, x,y, nhead)
        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        num_patches = x.shape[1]
        
        x = torch.permute(x, (0,1,4,2,3))
        x_ft = torch.fft.rfft2(x)
        
        out_ft = torch.zeros(batchsize, num_patches, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, self.nhead, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), dim=(-3, -2))
        x = torch.permute(x, (0,1,3,4,2,5))  # (batch, num_patches, x, y, d_model, nhead)
        return x


def apply_spatial_gaussian_conv(value, kernel):
    """
    value: (B, nH, T, H, W, d)
    kernel: (1, 1, 1, Hk, Wk)
    Returns: value_convolved, same shape as value
    """
    B, nH, T, H, W, d = value.shape
    value = value.permute(0, 1, 5, 2, 3, 4).reshape(B * nH * d * T, H, W)
    
    # Kernel shape
    _, _, kh, kw = kernel.shape

    # Apply conv2d without internal padding
    value_convs = []
    for value_t in value:
        # Compute asymmetric 'same' padding for even-sized kernels
        ph_top = kh // 2
        ph_bottom = kh - ph_top - 1
        pw_left = kw // 2
        pw_right = kw - pw_left - 1

        # Apply circular padding manually
        value_padded = F.pad(
            value_t.unsqueeze(0),  # [1, H, W]
            (pw_left, pw_right, ph_top, ph_bottom),  # (L, R, T, B)
            mode='circular'
        )
        
        value_convs.append(F.conv2d(
            value_padded.unsqueeze(0),
            kernel,  # [1, 1, kh, kw]
            padding=0,
        ))
    return torch.stack(value_convs).reshape(B, nH, d, T, H, W).permute(0, 1, 3, 4, 5, 2)  # (B, nH, T, H, W, d)


def make_gaussian_kernel(sigma, size):
    H, W = size
    y = torch.arange(H, dtype=torch.float32)
    x = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    center_y = (H - 1) / 2
    center_x = (W - 1) / 2
    kernel = torch.exp(-((xx - center_x)**2 + (yy - center_y)**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, H, W).to(device)  # Shape for depthwise conv2d


class ScaledDotProductAttention_Operator(nn.Module):
    def __init__(self, im_size, sigma=1.0):
        super(ScaledDotProductAttention_Operator, self).__init__()
        self.im_size = im_size
        # Build or store needed objects (like above)
        self.scale = nn.Parameter(
            torch.sqrt(torch.tensor(float(im_size[0] * im_size[1]) ** 2, dtype=torch.float32)),
            requires_grad=False
        )
        self.kernel = make_gaussian_kernel(sigma=sigma, size=self.im_size)
        

    def forward(self, query, key, value, key_padding_mask=None):
        B, nH, T, H, W, C = query.shape

        # Compute k_x similarity
        q_ = query.reshape(B, nH, T, -1)
        k_ = key.reshape(B, nH, T, -1)
        kx = torch.matmul(q_, k_.transpose(-1, -2)) / self.scale  # (B, nH, T, T)

        # Apply convolution in spatial domain to each value
        value_convolved = apply_spatial_gaussian_conv(value, self.kernel)  # (B,nH,T,H,W,d)
        output = torch.einsum("b n p q, b n q h w d -> b n p h w d", kx, value_convolved)
        return output


class MultiheadAttentionOperator(nn.Module):
    def __init__(self, d_model, nhead, modes1, modes2, im_size, sigma=1.0):
        super(MultiheadAttentionOperator, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.im_size = im_size

        self.query_operator   = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator     = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.value_operator_x = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead) 
        self.value_operator_y = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Operator(im_size, sigma=sigma)

    def forward(self, z, key_padding_mask=None):
        batch, T, full_H, W, _ = z.size()    # z    : N x T x 2H x W x C
        H = full_H // 2
        x, y = z[:,:,:H], z[:,:,H:] # x, y : N x T x  H x W x C
        
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key   = self.key_operator(x).permute(0,5,1,2,3,4)

        # NOTE: this assumes as in the paper that the value operator is 0'd out -- it is equivalent to
        # using the more general masking approach, but the dimension matching is slightly easier this way
        value = self.value_operator_y(y).permute(0,5,1,2,3,4)

        # NOTE: following assumes attention is computed with only 1 head
        attn_y = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask).reshape(batch, T, H, W, -1)
        output = torch.cat((x, attn_y), dim=2) # N x T x 2H x W x C
        return output


def init_spectral_identity_weights(layer, modes1, modes2, scale=1.0):
    in_channels = layer.in_channels
    out_channels = layer.out_channels
    nhead = layer.nhead

    weight = torch.zeros(in_channels, out_channels, modes1, modes2, nhead, dtype=torch.cfloat)
    for i in range(min(in_channels, out_channels)):
        weight[i, i, :, :, :] = scale

    layer.weights1.data = weight.clone()
    layer.weights2.data = weight.clone()


class TransformerOperatorLayer(nn.Module):#
    def __init__(self, d_model, nhead, patch_size=1, im_size=64, icl_init=False, r_l=None, sigma=1.0):
        super(TransformerOperatorLayer, self).__init__()
        
        modes1 = patch_size // 2 + 1
        modes2 = patch_size // 2 + 1
        
        self.patch_size = patch_size
        self.im_size    = im_size
        self.size_row   = im_size
        self.size_col   = im_size
        self.d_model    = d_model

        self.self_attn = MultiheadAttentionOperator(d_model, nhead, modes1, modes2, im_size, sigma=sigma)
        
        if icl_init:
            init_spectral_identity_weights(self.self_attn.query_operator,   modes1, modes2, scale=1.0)
            init_spectral_identity_weights(self.self_attn.key_operator,     modes1, modes2, scale=1.0)
            init_spectral_identity_weights(self.self_attn.value_operator_x, modes1, modes2, scale=0.0)
            init_spectral_identity_weights(self.self_attn.value_operator_y, modes1, modes2, scale=r_l)

    def forward(self, z, mask=None):
        attn_output = self.self_attn(z, key_padding_mask=mask) # N x T x 2H x W x C
        
        # HACK: hard-coded masking
        N, T, full_H, W, C = attn_output.shape
        attn_output[:,:T-1] = 0
        attn_output[:,T-1,:full_H//2] = 0

        z = z + attn_output
        return z


class TransformerOperator(nn.Module):
    def __init__(self, num_layers, im_size, icl_init=False, sigma=1.0):
        super(TransformerOperator, self).__init__()
        patch_size = im_size[1]

        transformer_layers = []
        for l in range(num_layers):
            transformer_layers.append(TransformerOperatorLayer(
                d_model=1, # assumed scalar field concatenated with coords (1 + 2)
                nhead=1,   # TODO: may want to extend theory to MHA in the future
                patch_size=patch_size,
                im_size=im_size,
                icl_init=icl_init,
                r_l=1e-1, # step size for in-context learning gradient descent
                sigma=sigma, 
            ))

        self.layers = nn.ModuleList(transformer_layers)

    def forward(self, z, coords=None, mask=None): 
        # z : B x T x 2H x W; coords: 2 x 2H x W 
        # Note: Inputs come in as 2H, since z = [x; y] stacked atop one another
        B, T, H, _   = z.shape
        z = z.unsqueeze(-1)  # B x T x H x W x 1
        
        if coords is not None:
            coords_stack = einops.rearrange(coords, "c h w -> h w c") # 2H x W x 2
            coords_stack = coords_stack.unsqueeze(0).unsqueeze(1)     # 1 x 1 x 2H x W x 2
            coords_stack = coords_stack.repeat(B, T, 1, 1, 1)         # B x T x 2H x W x 2
            z = torch.cat((z, coords_stack), dim=-1) # B x T x 2H x W x 3

        zs = []
        for layer in self.layers:
            z = layer(z, mask=mask) # B x T x 2H x W x 3
            zs.append(z[:,-1,H//2:,:,0].cpu().detach().numpy())
        return z[:,-1,H//2:,:,0], zs    # B x 1 x H x W x 1 -- bottom right is prediction (in first channel)