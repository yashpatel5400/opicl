import torch
import torch.nn as nn
import einops

from transformer_custom import TransformerEncoder_Operator
from transformer_custom import TransformerEncoderLayer_Conv


# Define the neural network model
class FANO(torch.nn.Module):
    def __init__(self, input_dim=1, output_dim=1, domain_dim=1, d_model=32, nhead=8, num_layers=6,
                 learning_rate=0.01, max_sequence_length=100,
                 do_layer_norm=True,
                 use_transformer=True,
                 patch=False,
                 patch_size=None,
                 modes=None,
                 im_size=None,
                 activation='relu',
                 dropout=0.1, norm_first=False, dim_feedforward=2048):
        super(FANO, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.domain_dim = domain_dim
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.use_transformer = use_transformer
        self.patch = patch
        self.patch_size = patch_size
        self.patch_dim = (self.domain_dim+1)*(self.patch_size**2)
        self.im_size = im_size
        

        encoder_layer = TransformerEncoderLayer_Conv(
            d_model=d_model, nhead=nhead,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            do_layer_norm=do_layer_norm,
            dim_feedforward=dim_feedforward,
            patch_size=patch_size,
            modes=modes,
            im_size=im_size,
            batch_first=True)  # when batch first, expects input tensor (batch_size, Seq_len, input_dim)
        self.encoder = TransformerEncoder_Operator(encoder_layer, num_layers=num_layers)

        self.linear_in = nn.Linear(3,d_model)
        self.size_row = self.im_size
        self.size_col = self.im_size
        self.linear_out = nn.Linear(d_model,1)
        self.num_patches = (self.size_row*self.size_col)//(self.patch_size**2)

    def forward(self, x, coords): # x : B x T x H x W; coords: 2 x H x W
        B, T, _, _ = x.shape
        coords_stack = einops.rearrange(coords, "c h w -> h w c") # H x W x 2
        coords_stack = coords_stack.unsqueeze(0).unsqueeze(1)     # 1 x 1 x H x W x 2
        coords_stack = coords_stack.repeat(B, T, 1, 1, 1)         # B x T x H x W x 2

        x = x.unsqueeze(-1) # B x T x H x W x 1

        z = torch.cat((x, coords_stack), dim=-1) # B x T x H x W x 3
        z = self.linear_in(z)                    # B x T x H x W x d
        z = self.encoder(z)                      # B x T x H x W x d
        z = self.linear_out(z)                   # B x T x H x W x 1

        x = z + x      # B x T x H x W x 1
        return x[:,-1] # B x 1 x H x W x 1