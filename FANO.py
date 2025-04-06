import torch
import torch.nn as nn
import einops

from transformer_custom import TransformerEncoder_Operator
from transformer_custom import TransformerEncoderLayer_Conv


class FANO(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, 
                 learning_rate=0.01, max_sequence_length=100,
                 use_transformer=True,
                 patch=False,
                 nhead=8, 
                 num_layers=6,
                 do_layer_norm=True,
                 patch_size=None,
                 modes=None,
                 im_size=None,
                 activation='relu',
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(FANO, self).__init__()

        transformer_layer = TransformerEncoderLayer_Conv(
            d_model=1, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            do_layer_norm=do_layer_norm,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            modes=modes,
            im_size=im_size,
            batch_first=True)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        self.transformer = TransformerEncoder_Operator(transformer_layer, num_layers=num_layers)

    def forward(self, z, coords): 
        # z : B x T x 2H x W; coords: 2 x 2H x W 
        # Note: Inputs come in as 2H, since z = [x; y] stacked atop one another
        B, T, H, _   = z.shape
        coords_stack = einops.rearrange(coords, "c h w -> h w c") # 2H x W x 2
        coords_stack = coords_stack.unsqueeze(0).unsqueeze(1)     # 1 x 1 x 2H x W x 2
        coords_stack = coords_stack.repeat(B, T, 1, 1, 1)         # B x T x 2H x W x 2

        z = z.unsqueeze(-1)                         # B x T x H x W x 1
        z_tf = torch.cat((z, coords_stack), dim=-1) # B x T x 2H x W x 3
        z_tf = self.transformer(z_tf)               # B x T x 2H x W x d
        return z[:,-1,H//2:] # B x 1 x H x W x 1 -- bottom right is prediction