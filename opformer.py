import einops
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


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


class ScaledDotProductAttention_Operator(nn.Module):
    def __init__(self, im_size):
        super(ScaledDotProductAttention_Operator, self).__init__()
        self.scale = nn.Parameter(torch.sqrt(torch.FloatTensor([((im_size[0]*im_size[1])**2)])), requires_grad=False)
        
    def forward(self, query, key, value, key_padding_mask=None):
        scores = torch.einsum("bnpxyd,bnqxyd->bnpq", query, key) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.einsum("bnpq,bnqxyd->bnpxyd", attention_weights, value)
        output = output.permute(0,2,3,4,5,1)
        return output


class MultiheadAttentionOperator(nn.Module):
    def __init__(self, d_model, nhead, modes1, modes2, im_size):
        super(MultiheadAttentionOperator, self).__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.im_size = im_size

        self.query_operator   = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.key_operator     = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator_x = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)
        self.value_operator_y = SpectralConv2d_Attention(d_model, self.d_k, modes1, modes2, nhead)

        self.scaled_dot_product_attention = ScaledDotProductAttention_Operator(im_size)

    def forward(self, z, key_padding_mask=None):
        batch, T, full_H, W, _ = z.size()               # z    : N x T x 2H x W x C
        x, y = z[:,:,:full_H // 2], z[:,:,full_H // 2:] # x, y : N x T x  H x W x C
        
        query = self.query_operator(x).permute(0,5,1,2,3,4)
        key   = self.key_operator(x).permute(0,5,1,2,3,4)

        value_x = self.value_operator_x(x).permute(0,5,1,2,3,4)
        value_y = self.value_operator_y(y).permute(0,5,1,2,3,4)
        value   = torch.cat([value_x, value_y], dim=3)

        output = self.scaled_dot_product_attention(query, key, value, key_padding_mask=key_padding_mask)
        output = output.reshape(batch, T, full_H, W,-1)
        return output


class TransformerOperatorLayer(nn.Module):#
    def __init__(self, d_model, nhead, patch_size=1, im_size=64):
        super(TransformerOperatorLayer, self).__init__()
        
        modes1 = patch_size // 2 + 1
        modes2 = patch_size // 2 + 1
        
        self.patch_size = patch_size
        self.im_size    = im_size
        self.size_row   = im_size
        self.size_col   = im_size
        self.d_model    = d_model

        self.self_attn = MultiheadAttentionOperator(d_model, nhead, modes1, modes2, im_size)

    def forward(self, z, mask=None):
        attn_output = self.self_attn(z, key_padding_mask=mask)
        z = z + attn_output
        return z


class TransformerOperator(nn.Module):
    def __init__(self, num_layers, im_size):
        super(TransformerOperator, self).__init__()
        patch_size = im_size[1]
        transformer_layer = TransformerOperatorLayer(
            d_model=3, # assumed scalar field concatenated with coords (1 + 2)
            nhead=1,   # TODO: may want to extend theory to MHA in the future
            patch_size=patch_size,
            im_size=im_size)
        self.layers = nn.ModuleList([copy.deepcopy(transformer_layer) for _ in range(num_layers)])

    def forward(self, z, coords, mask=None): 
        # z : B x T x 2H x W; coords: 2 x 2H x W 
        # Note: Inputs come in as 2H, since z = [x; y] stacked atop one another
        B, T, H, _   = z.shape
        coords_stack = einops.rearrange(coords, "c h w -> h w c") # 2H x W x 2
        coords_stack = coords_stack.unsqueeze(0).unsqueeze(1)     # 1 x 1 x 2H x W x 2
        coords_stack = coords_stack.repeat(B, T, 1, 1, 1)         # B x T x 2H x W x 2

        z = z.unsqueeze(-1)                      # B x T x H x W x 1
        z = torch.cat((z, coords_stack), dim=-1) # B x T x 2H x W x 3
        for layer in self.layers:
            z = layer(z, mask=mask) # B x T x 2H x W x 3
        return z[:,-1,H//2:,:,0]    # B x 1 x H x W x 1 -- bottom right is prediction (in first channel)