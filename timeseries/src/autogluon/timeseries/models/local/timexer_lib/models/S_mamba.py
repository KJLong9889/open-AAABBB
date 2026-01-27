import torch.nn as nn
from mamba_ssm import Mamba

class smamba(nn.Module):
    def __init__(self, d_model=512, d_ff=512, dropout=0.2, activation="relu", use_smamba=False):
        super(smamba, self).__init__()

        if use_smamba:
            # d_ff = d_ff or 4 * d_model
            # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
            # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
            # self.norm1 = nn.LayerNorm(d_model)
            # self.norm2 = nn.LayerNorm(d_model)
            # self.dropout = nn.Dropout(dropout)
            # self.activation = F.relu if activation == "relu" else F.gelu
            self.man = Mamba(
                d_model=d_model,  # Model dimension d_model
                d_state=32,  # SSM state expansion factor
                d_conv=2,  # Local convolution width
                expand=1,  # Block expansion factor)
            )
            self.man2 = Mamba(
                d_model=d_model,  # Model dimension d_model
                d_state=32,  # SSM state expansion factor
                d_conv=2,  # Local convolution width
                expand=1,  # Block expansion factor)
            )

    def forward(self, x):
        new_x = self.man(x) + self.man2(x.flip(dims=[1])).flip(dims=[1])
        return new_x

        # x = x + new_x
        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        # return self.norm2(x + y)


