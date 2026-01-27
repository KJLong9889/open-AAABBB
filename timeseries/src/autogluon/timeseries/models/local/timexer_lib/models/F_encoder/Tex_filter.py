import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# def visualize_frequency_response(weight, seq_len, save_path="./frequency_response.png"):
#     """
#     weight: 复数张量 [B, freq_bins, D]（TexFilter 输出）
#     seq_len: 原始时间长度 N
#     save_path: 保存路径
#     """
#     # 计算幅值：平均每个通道

#     amplitude = torch.abs(weight).mean(dim=-1).squeeze(0).detach().cpu().numpy()  # [freq_bins]

#     # 生成对应频率
#     freqs = np.fft.rfftfreq(seq_len, d=1.0)  # d=1.0 表示采样间隔为1
#     print("***********", amplitude, freqs)
    
#     # 绘图
#     plt.figure(figsize=(8,4))
#     plt.fill_between(freqs, amplitude, color="#c59f29", alpha=0.7)
#     plt.grid(True, color='gray', alpha=0.3)
#     plt.xlabel("Frequency")
#     plt.ylabel("Amplitude")
#     plt.title("Frequency Response of TexFilter")
#     plt.xlim([0, freqs.max()])
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"[可视化] 频谱图已保存到：{save_path}")


class TexFilter(nn.Module):

    def __init__(self, d_model, dropout):
        super(TexFilter, self).__init__()

        self.embed_size = d_model
        # self.hidden_size = 2048
        # self.dropout = configs.dropout
        # self.band_width = 96
        self.scale = 0.02       #0.02
        # self.sparsity_threshold = 0.01
        self.sparsity_threshold = 0.02      #0.01
      
        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))


        # self.fc = nn.Sequential(
        #     nn.Linear(self.embed_size, self.hidden_size),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_size, self.embed_size)
        # )

        # self.output = nn.Linear(self.embed_size, self.pred_len)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ln_freq = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  #0.2

    def texfilter(self, x):
        B, N, _ = x.shape
        o1_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device='cuda')
        o1_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device='cuda')

        o2_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device='cuda')
        o2_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device='cuda')
        
        x_real = self.ln_freq(x.real)
        x_imag = self.ln_freq(x.imag)
        #第一层复数线性变换 + ReLU
        o1_real = F.relu(
            torch.einsum('bid,d->bid', x_real, self.w[0]) - \
            torch.einsum('bid,d->bid', x_imag, self.w[1]) + \
            self.rb1
        )

        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x_imag, self.w[0]) + \
            torch.einsum('bid,d->bid', x_real, self.w[1]) + \
            self.ib1
        )
        #第二层复数线性变换（没有 ReLU）
        o2_real = F.relu(
                torch.einsum('bid,d->bid', o1_real, self.w1[0]) - \
                torch.einsum('bid,d->bid', o1_imag, self.w1[1]) + \
                self.rb2
        )

        o2_imag = F.relu(
                torch.einsum('bid,d->bid', o1_imag, self.w1[0]) + \
                torch.einsum('bid,d->bid', o1_real, self.w1[1]) + \
                self.ib2
        )
        #把实部和虚部拼接成复数；softshrink 把绝对值小于阈值的元素压成 0 → 稀疏化；最后转为复数张量返回。
        y = torch.stack([o2_real, o2_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        B, N, D = x.shape

        x = torch.fft.rfft(x, dim=1, norm='ortho')  # 时域 → 频域
        weight = self.texfilter(x)  # 学习频域滤波器

        # w = weight[5, :, 0]
        # # 可视化
        # visualize_frequency_response(w, seq_len=N)
        
        x = x * weight  # 点乘调制
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")    # 回到时域      #ortho

        x = self.layernorm1(x)
        x = self.dropout(x)

        return x

