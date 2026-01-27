import torch
import torch.nn as nn

class PaiFilter(nn.Module):
    def __init__(self):
        super(PaiFilter, self).__init__()
        # self.scale = 0.02
        self.scale = 0.02

        # self.embed_size = self.seq_len
        self.embed_size = 512
        
        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))
        # self.layernorm1 = nn.LayerNorm(512)
        # self.dropout = nn.Dropout(0.2)  #ETTh1=0.5

        # self.fc = nn.Sequential(
        #     nn.Linear(self.embed_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.pred_len)
        # )


    def circular_convolution(self, x, w):
        x = torch.fft.rfft(x, dim=2, norm='ortho')
        w = torch.fft.rfft(w, dim=1, norm='ortho')
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x):
        # z = x
        # z = self.revin_layer(z, 'norm')
        # x = z

        # x = x.permute(0, 2, 1)

        x = self.circular_convolution(x, self.w.to('cuda'))  # B, N, D
        # x = self.layernorm1(x)
        # x = self.dropout(x)

        # x = self.fc(x)


        return x
