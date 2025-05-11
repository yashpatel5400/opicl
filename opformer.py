import einops
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import kernels

device = "cuda"

class SpectralConv2d_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.nhead = 1
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
        x = torch.permute(x, (0,1,3,4,2,5))[...,0]  # (batch, num_patches, x, y, d_model, nhead) -> just take 0th for head, since only 1
        return x


class RestrictedSpectralOperator(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, scale=1.0):
        super().__init__()
        assert in_channels == out_channels, "Restricted operator assumes square channels"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.nhead = 1  # still assuming 1 head

        # Learnable scalar (shared across all frequencies and channels)
        self.scalar = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def compl_mul2d(self, input, scale):
        B, T, C, H, W = input.shape
        device = input.device

        weight = torch.zeros(C, C, self.modes1, self.modes2, self.nhead, dtype=torch.cfloat, device=device)
        for i in range(C):
            weight[i, i, :, :, 0] = scale

        return torch.einsum("bnixy,ioxyh->bnoxyh", input, weight)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = torch.permute(x, (0, 1, 4, 2, 3))  # (B, T, C, H, W)
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, T, self.out_channels, H, W//2 + 1, self.nhead, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :, :self.modes1, :self.modes2, :] = self.compl_mul2d(
            x_ft[:, :, :, :self.modes1, :self.modes2], self.scalar
        )
        out_ft[:, :, :, -self.modes1:, :self.modes2, :] = self.compl_mul2d(
            x_ft[:, :, :, -self.modes1:, :self.modes2], self.scalar
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), dim=(-3, -2))
        x = torch.permute(x, (0, 1, 3, 4, 2, 5))[..., 0]  # (B, T, H, W, C)
        return x


def apply_spatial_conv_unbatched(value, kernel):
    """
    value: (B, nH, T, H, W, d)
    kernel: (1, 1, 1, Hk, Wk)
    Returns: value_convolved, same shape as value
    """
    B, T, H, W, d = value.shape
    value = value.reshape(B * d * T, H, W)
    
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
    return torch.stack(value_convs).squeeze().unsqueeze(0).unsqueeze(-1)  # (B, T, H, W, d)


def apply_spatial_conv(value, kernel):
    """
    value:  (B, T, H, W, 1)
    kernel: (1, 1, Hk, Wk)
    Returns:
        value_convolved: (B, T, H, W, 1)
    """
    B, T, H, W, _ = value.shape
    kh, kw = kernel.shape[-2:]

    # Merge batch and time dimensions for parallel conv
    x = value.view(B * T, 1, H, W)  # shape: (B*T, 1, H, W)

    # Compute asymmetric 'same' padding
    ph_top    = kh // 2
    ph_bottom = kh - ph_top - 1
    pw_left   = kw // 2
    pw_right  = kw - pw_left - 1

    # Apply circular padding
    x_padded = F.pad(x, (pw_left, pw_right, ph_top, ph_bottom), mode='circular')

    # Apply depthwise 2D convolution
    conv = F.conv2d(x_padded, kernel, padding=0)  # shape: (B*T, 1, H, W)

    # Reshape back to (B, T, H, W, 1)
    return conv.view(B, T, H, W, 1)


def compute_kx(query, key, kx_name, kx_sigma):
    B, T_q, H, W, C = query.shape
    _, T_k, _, _, _ = key.shape
    q_ = query.reshape(query.shape[0], query.shape[1], -1)
    k_ = key.reshape(key.shape[0], key.shape[1], -1)

    if kx_name == "linear":
        kx = torch.matmul(q_, k_.transpose(-1, -2))

    elif kx_name == "laplacian":
        # L2 distance (correct)
        q_exp = q_.unsqueeze(2)  # (B, T, 1, D)
        k_exp = k_.unsqueeze(1)  # (B, 1, T, D)
        l2_dist = torch.sqrt(torch.sum((q_exp - k_exp) ** 2, dim=-1) + 1e-8)  # (B, T, T)
        kx = torch.exp(-l2_dist / kx_sigma)

    elif kx_name == "gradient_rbf":
        # Assume spatial gradients
        q_grad_y = query.diff(dim=2, append=query[:,:,-1:,:,:])  # simple gradient along y
        q_grad_x = query.diff(dim=3, append=query[:,:,:,-1:,:])  # simple gradient along x
        k_grad_y = key.diff(dim=2, append=key[:,:,-1:,:,:])
        k_grad_x = key.diff(dim=3, append=key[:,:,:,-1:,:])

        qg = torch.cat([q_grad_x, q_grad_y], dim=-1).reshape(B, T_q, -1)
        kg = torch.cat([k_grad_x, k_grad_y], dim=-1).reshape(B, T_k, -1)
        qg_norm = (qg ** 2).sum(dim=-1, keepdim=True)
        kg_norm = (kg ** 2).sum(dim=-1, keepdim=True)
        dist_sq = qg_norm + kg_norm.transpose(-1, -2) - 2 * torch.matmul(qg, kg.transpose(-1, -2))
        kx = torch.exp(-dist_sq / (2 * kx_sigma**2))

    elif kx_name == "energy":
        q_energy = (q_ ** 2).sum(dim=-1, keepdim=True)
        k_energy = (k_ ** 2).sum(dim=-1, keepdim=True)
        energy_diff = q_energy - k_energy.transpose(-1, -2)
        kx = torch.exp(-(energy_diff ** 2) / (2 * kx_sigma**2))

    else:
        raise ValueError(f"Unsupported kx_name: {kx_name}")

    return kx


class ScaledDotProductAttention_Operator(nn.Module):
    def __init__(self, im_size, ky_kernel, kx_name='linear', kx_sigma=1.0):
        super(ScaledDotProductAttention_Operator, self).__init__()
        self.im_size = im_size
        self.scale = nn.Parameter(
            torch.sqrt(torch.tensor(float(im_size[0] * im_size[1]) ** 2, dtype=torch.float32)),
            requires_grad=False
        )
        self.ky_kernel = torch.from_numpy(ky_kernel).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
        self.kx_name = kx_name
        self.kx_sigma = kx_sigma

    def forward(self, query, key, value):
        B, T, H, W, C = query.shape

        # Compute k_x similarity
        unmasked_kx = compute_kx(query, key, self.kx_name, self.kx_sigma)

        # Masking: copy from your original if needed
        M = torch.zeros((T, T), device=query.device, dtype=torch.float32)
        M[:T-1, :T-1] = torch.eye(T-1, device=query.device)
        kx = torch.matmul(M, unmasked_kx)

        # Convolve value with spatial kernel
        value_convolved = apply_spatial_conv(value, self.ky_kernel)  # (B, T, H, W, C)
        
        v = value_convolved.permute(0, 2, 3, 4, 1)      # (B, H, W, C, T)
        attn = torch.einsum('bhwct,btq->bhwcq', v, kx)  # (B, H, W, C, T)
        output = attn.permute(0, 4, 1, 2, 3)            # (B, T, H, W, C)

        return output, kx

class AttentionOperator(nn.Module):
    def __init__(self, d_model, modes1, modes2, im_size, ky_kernel, icl_lr, kx_name, kx_sigma):
        super(AttentionOperator, self).__init__()
        self.d_k = d_model
        self.im_size = im_size

        self.query_operator = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2)
        self.key_operator   = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2)
        self.value_operator = RestrictedSpectralOperator(d_model, self.d_k, modes1, modes2, scale=icl_lr)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Operator(im_size, ky_kernel, kx_name, kx_sigma)

    def forward(self, z):
        batch, T, full_H, W, _ = z.size() # z : N x T x 2H x W x C
        H = full_H // 2
        x, y = z[:,:,:H], z[:,:,H:] # x, y : N x T x H x W x C
        
        # NOTE: this assumes as in the paper that the value operator is 0'd out -- it is equivalent to
        # using the more general masking approach, but the dimension matching is slightly easier this way
        query = self.query_operator(x)
        key   = self.key_operator(x)
        value = self.value_operator(y)

        # NOTE: following assumes attention is computed with only 1 head
        attn_results = self.scaled_dot_product_attention(query, key, value)
        attn_y = attn_results[0].reshape(batch, T, H, W, -1)
        output = torch.cat((torch.zeros_like(x), attn_y), dim=2) # N x T x 2H x W x C
        return output, attn_results[1]


def init_spectral_identity_weights(layer, modes1, modes2, scale=1.0):
    in_channels = layer.in_channels
    out_channels = layer.out_channels

    weight = torch.zeros(in_channels, out_channels, modes1, modes2, 1, dtype=torch.cfloat)
    for i in range(min(in_channels, out_channels)):
        weight[i, i, :, :] = scale

    layer.weights1.data = weight.clone()
    layer.weights2.data = weight.clone()


class TransformerOperatorLayer(nn.Module):#
    def __init__(self, d_model, patch_size=1, im_size=64, icl_init=False, r_l=None, ky_kernel=None, kx_name=None, kx_sigma=1.0):
        super(TransformerOperatorLayer, self).__init__()
        
        modes1 = patch_size // 2 + 1
        modes2 = patch_size // 2 + 1
        
        self.patch_size = patch_size
        self.im_size    = im_size
        self.size_row   = im_size
        self.size_col   = im_size
        self.d_model    = d_model

        self.self_attn = AttentionOperator(d_model, modes1, modes2, im_size, ky_kernel, r_l, kx_name, kx_sigma)
        
        if icl_init:
            init_spectral_identity_weights(self.self_attn.query_operator,   modes1, modes2, scale=1.0)
            init_spectral_identity_weights(self.self_attn.key_operator,     modes1, modes2, scale=1.0)

    def forward(self, z):
        z_delta, attn = self.self_attn(z)
        z = z + z_delta # N x T x 2H x W x C
        return z, attn


class TransformerOperator(nn.Module):
    def __init__(self, num_layers, im_size, ky_kernel, kx_name, kx_sigma=1.0, icl_lr=1e-3, icl_init=False):
        super(TransformerOperator, self).__init__()
        patch_size = im_size[1]

        transformer_layers = []
        for l in range(num_layers):
            transformer_layers.append(TransformerOperatorLayer(
                d_model=1, # assumed scalar field concatenated with coords (1 + 2)
                patch_size=patch_size,
                im_size=im_size,
                icl_init=icl_init,
                r_l=icl_lr, # step size for in-context learning gradient descent
                ky_kernel=ky_kernel, 
                kx_name=kx_name,
                kx_sigma=kx_sigma,
            ))

        self.layers = nn.ModuleList(transformer_layers)

    def forward(self, z): 
        # z : B x T x 2H x W; coords: 2 x 2H x W 
        # Note: Inputs come in as 2H, since z = [x; y] stacked atop one another
        B, T, H, _   = z.shape
        z = z.unsqueeze(-1)  # B x T x H x W x 1

        zs = []
        attns = []
        for layer in self.layers:
            z, attn = layer(z) # B x T x 2H x W x 3
            zs.append(z.cpu().detach().numpy())
            attns.append(attn.cpu().detach().numpy())
        return z[:,-1,H//2:,:,0], zs, np.array(attns)    # B x 1 x H x W x 1 -- bottom right is prediction (in first channel)

def test_apply_spatial_conv_batch_equivalence():
    # Test configuration
    B, T, H, W = 2, 3, 8, 8
    torch.manual_seed(0)
    value = torch.randn(B, T, H, W, 1)
    kernel = torch.randn(1, 1, 3, 3)

    # Run original implementation one-by-one
    out_manual = []
    for b in range(B):
        val_b = value[b:b+1]  # (1, T, H, W, 1)
        out_b = apply_spatial_conv_unbatched(val_b, kernel)
        out_manual.append(out_b)
    out_manual = torch.cat(out_manual, dim=0)  # shape (B, T, H, W, 1)

    # Run batched implementation
    out_batch = apply_spatial_conv(value, kernel)

    # Compare
    assert torch.allclose(out_manual, out_batch, atol=1e-5), "Mismatch between manual and batched conv!"
    print("✅ Batched convolution matches manual looped version.")

def test_batch_vs_single_consistency(model, input_tensor):
    """
    model: instance of TransformerOperator
    input_tensor: torch.Tensor of shape (B, T, 2H, W)
    """
    model.eval()
    with torch.no_grad():
        B, T, H2, W = input_tensor.shape
        H = H2 // 2

        # Batch output
        batch_output, _ = model(input_tensor)  # shape: (B, H, W)

        # Individual outputs
        single_outputs = []
        for b in range(B):
            single_input = input_tensor[b:b+1]  # shape (1, T, 2H, W)
            single_output, _ = model(single_input)
            single_outputs.append(single_output.squeeze(0))  # (H, W)

        single_output_stacked = torch.stack(single_outputs)  # (B, H, W)

        # Compare
        diff = torch.norm(batch_output - single_output_stacked)
        print(f"L2 norm difference between batch and individual results: {diff.item():.6e}")

        assert torch.allclose(batch_output, single_output_stacked, atol=1e-5), \
            "Batch and single-inference outputs differ!"

        print("✅ Batch and single-inference outputs match.")

if __name__ == "__main__":
    test_apply_spatial_conv_batch_equivalence()

    H, W = 64, 64
    kernel_maps = kernels.Kernels(H, W)

    kx_sigma = 1.0
    kx_name = "linear"
    kx_name_true = "linear"
    
    kx_true = kernels.get_kx_kernel(kx_name_true, sigma=kx_sigma)
    ky_true = kernel_maps.get_kernel("gaussian")

    im_size = (64, 64)
    device = "cuda"

    kernel_to_preds, kernel_to_errors = {}, {}
    r = .01
    num_layers = 3
    opformer = TransformerOperator(
        num_layers=num_layers, 
        im_size=im_size, 
        ky_kernel=ky_true, 
        kx_name=kx_name, 
        kx_sigma=kx_sigma, 
        icl_lr=-r, 
        icl_init=True).to(device)
    
    B = 4
    z = torch.randn(B, 25, 128, 64).to(device)
    test_batch_vs_single_consistency(opformer, z)