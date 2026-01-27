import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time
from typing import Optional, Dict, Any
import mamba_ssm
import os
import sys
from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.abstract import AbstractTimeSeriesModel
# 1. 获取当前文件 (ag_TimeXer.py) 所在的目录
# ----------------------------------------------------------------------
# 1. 获取当前文件 (ag_TimeXer.py) 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 拼接出 timexer_lib 的绝对路径
lib_path = os.path.join(current_dir, 'timexer_lib')

# 3. 检查文件夹是否存在
if not os.path.exists(lib_path):
    # 如果找不到，打印当前目录结构辅助调试
    print(f"【错误】找不到 timexer_lib 文件夹！")
    print(f"预期位置: {lib_path}")
    print(f"当前目录 {current_dir} 下的文件: {os.listdir(current_dir) if os.path.exists(current_dir) else '目录不存在'}")
    raise ImportError(f"请将 TimeXer 源码文件夹重命名为 'timexer_lib' 并放入 {current_dir} 目录下")

# 4. 将 timexer_lib 加入系统路径 (优先级设为0，确保最先找到)
# 注意：我们直接加入 lib_path，这样代码里就可以直接 import models, layers, utils
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# 5. 尝试导入 TimeXer 模块
try:
    # 因为我们将 lib_path 加入了 sys.path，所以直接从 models 导入即可
    # 这样也能解决 TimeXer 内部 "from layers import ..." 的路径问题
    from models.TimeXer import Model as TimeXerNet
    from utils.timefeatures import time_features
    print("【成功】TimeXer 模块导入成功")
except ImportError as e:
    # 备用方案：有时候需要带上包名 (取决于你的 timexer_lib 内部是否有 __init__.py)
    try:
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        from timexer_lib.models.TimeXer import Model as TimeXerNet
        from timexer_lib.utils.timefeatures import time_features
        print("【成功】TimeXer 模块通过包名导入成功")
    except ImportError as e2:
        raise ImportError(f"TimeXer 导入失败。\n路径尝试1错误: {e}\n路径尝试2错误: {e2}")
    

class TimeXerConfig:
    """
    模拟 TimeXer 原生 args 对象，用于传递参数
    """
    def __init__(self, params: Dict[str, Any], pred_len: int, seq_len: int, enc_in: int, freq: str):
        # 核心维度
        self.seq_len = seq_len
        self.label_len = seq_len // 2  # TimeXer 默认通常用一半作为 label_len
        self.pred_len = pred_len
        
        # 通道数 (AutoGluon 主要是单变量或多变量，这里简化为 enc_in=c_out)
        self.enc_in = enc_in
        self.dec_in = enc_in
        self.c_out = enc_in 
        
        # 模型超参 (从 params 获取，或者使用默认值)
        self.d_model = params.get('d_model', 512)
        self.n_heads = params.get('n_heads', 8)
        self.e_layers = params.get('e_layers', 2)
        self.d_ff = params.get('d_ff', 2048)
        self.dropout = params.get('dropout', 0.1)
        self.embed = params.get('embed', 'timeF')
        self.activation = params.get('activation', 'gelu')
        self.output_attention = False
        self.freq = freq
        
        # TimeXer 特有参数 (根据论文默认值设置)
        self.use_norm = params.get('use_norm', True)
        self.down_sampling_layers = params.get('down_sampling_layers', 0)
        self.down_sampling_method = params.get('down_sampling_method', 'avg')
        self.down_sampling_window = params.get('down_sampling_window', 1)

class TimeSeriesDataset(Dataset):
    """
    极简版 Dataset，适配 AutoGluon 的数据格式
    """
    def __init__(self, data_array, timestamps, seq_len, pred_len, freq):
        self.data = data_array
        self.timestamps = timestamps
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.freq = freq
        
        # 预计算时间特征
        # 将 pandas timestamp 转为 TimeXer 需要的 (N, 4) 矩阵
        df_stamp = pd.DataFrame({"date": timestamps})
        self.data_stamp = time_features(df_stamp, freq=freq) # timeenc=1 for fixed

    def __len__(self):
        # 简单的滑窗逻辑
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark)
        )

class TimeXerModel(AbstractTimeSeriesModel):
    """
    AutoGluon 的 TimeXer 包装器
    """
    
    # 声明支持的功能
    _supports_known_covariates = False 
    _supports_past_covariates = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = None

    def _get_default_hyperparameters(self) -> dict:
        return {
            "d_model": 512,
            "d_ff": 2048,
            "e_layers": 2,
            "n_heads": 8,
            "dropout": 0.1,
            "batch_size": 32,
            "lr": 0.0001,
            "epochs": 10,
            "context_length_ratio": 2, # seq_len = prediction_length * ratio
        }

    def _fit(
        self,
        train_data: TimeSeriesDataFrame,
        val_data: Optional[TimeSeriesDataFrame] = None,
        time_limit: Optional[float] = None,
        **kwargs,
    ) -> None:
        start_time = time.time()
        
        # 1. 数据预处理
        # 填充缺失值，因为 TimeXer 不能处理 NaN
        train_data = train_data.fill_missing_values()
        
        # 获取超参
        params = self._hyperparameters
        seq_len = int(self.prediction_length * params.get("context_length_ratio", 2))
        
        # 2. 准备配置 (Config)
        # 自动推断频率
        
        freq_str = self.freq
        
        # 假设所有序列长度一致且对齐 (简化处理)，或者把所有序列拼在一起训练
        # 这里为了稳健性，我们把所有 item_id 的序列分开做成 samples 然后 concat
        all_samples = []
        all_stamps = []
        
        # 将 AutoGluon 格式转换为 numpy
        # 注意：这里假设是单变量预测。如果是多变量，需要处理 target 列之外的列
        for item_id, group in train_data.groupby(level='item_id'):
            values = group[self.target].values
            timestamps = group.index.get_level_values('timestamp').values
            if len(values) > seq_len + self.prediction_length:
                # 为了简单，这里只取每个序列做训练，实际应该写得更高效
                ds = TimeSeriesDataset(
                    values.reshape(-1, 1), 
                    timestamps, 
                    seq_len, 
                    self.prediction_length, 
                    freq_str
                )
                # 使用 DataLoader 来利用 Dataset 的 __getitem__ 逻辑
                # 但这里我们直接手动抽取所有滑窗可能太慢，
                # 生产环境建议重写 Dataset 让它支持多序列索引
                pass 

        # --- 简化版数据加载 (针对单序列或简单的多序列拼接) ---
        # 我们直接使用第一个序列来初始化模型维度，实际应用需遍历
        sample_item = train_data.iloc[0]
        enc_in = 1 # 单变量
        
        self.args = TimeXerConfig(params, self.prediction_length, seq_len, enc_in, freq_str)
        
        # 3. 初始化模型
        self.model = TimeXerNet(self.args).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=params['lr'])
        criterion = nn.MSELoss()
        
        # 4. 构建训练集
        # 将所有时间序列视为独立样本
        train_datasets = []
        for item_id, group in train_data.groupby(level='item_id'):
            values = group[self.target].values.reshape(-1, 1)
            timestamps = group.index.get_level_values('timestamp')
            if len(values) < seq_len + self.prediction_length:
                continue
            train_datasets.append(TimeSeriesDataset(values, timestamps, seq_len, self.prediction_length, freq_str))
            
        if not train_datasets:
            raise ValueError("数据太短，无法进行训练")
            
        full_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_loader = DataLoader(full_dataset, batch_size=params['batch_size'], shuffle=True)

        # 5. 训练循环
        self.model.train()
        for epoch in range(params['epochs']):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 检查时间限制
                if time_limit and (time.time() - start_time > time_limit):
                    print("Time limit reached.")
                    return

                optimizer.zero_grad()
                
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = batch_x_mark.to(self.device)
                batch_y_mark = batch_y_mark.to(self.device)
                
                # Decoder input: 也就是 exp_long_term_forecasting 里的 dec_inp
                # 策略: label_len 部分填真实值，pred_len 部分填 0
                dec_inp = torch.zeros_like(batch_y[:, -self.prediction_length:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # TimeXer 输出通常不需要取特定维度，直接是 (B, L, D)
                f_dim = -1 if self.args.enc_in > 1 else 0
                outputs = outputs[:, -self.prediction_length:, :]
                batch_y = batch_y[:, -self.prediction_length:, :]

                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch+1} loss: {loss.item()}")

    def _predict(
        self,
        data: TimeSeriesDataFrame,
        known_covariates: Optional[TimeSeriesDataFrame] = None,
        **kwargs,
    ) -> TimeSeriesDataFrame:
        self.model.eval()
        data = data.fill_missing_values()
        results = []
        
        freq_map = {'H': 'h', 'T': 't', 'S': 's', 'M': 'm', 'A': 'a', 'W': 'w', 'D': 'd'}
        freq_str = self.freq
        
        for item_id, group in data.groupby(level='item_id'):
            # 准备输入数据 (取最后 seq_len 长度)
            values = group[self.target].values
            if len(values) < self.args.seq_len:
                # Padding or simple error handling
                pad_len = self.args.seq_len - len(values)
                values = np.pad(values, (pad_len, 0), mode='edge')
            else:
                values = values[-self.args.seq_len:]
                
            # 准备时间戳 (需要生成未来的时间戳)
            last_timestamp = group.index.get_level_values('timestamp')[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp, 
                periods=self.prediction_length + 1, 
                freq=self.freq
            )[1:] # 不包含最后一个已知点
            
            # 构造输入 Tensor
            seq_x = torch.FloatTensor(values.reshape(1, -1, 1)).to(self.device)
            
            # 构造 Time Features
            # 需要过去的时间戳 + 未来的时间戳
            past_timestamps = group.index.get_level_values('timestamp')[-self.args.seq_len:]
            all_timestamps = pd.concat([
                pd.Series(past_timestamps), 
                pd.Series(future_timestamps)
            ])
            df_stamp = pd.DataFrame({"date": all_timestamps})
            data_stamp = time_features(df_stamp, timeenc=1, freq=freq_str)
            data_stamp = torch.FloatTensor(data_stamp).unsqueeze(0).to(self.device)
            
            seq_x_mark = data_stamp[:, :self.args.seq_len, :]
            seq_y_mark = data_stamp[:, self.args.seq_len:, :]
            
            # Decoder Input
            dec_inp = torch.zeros(1, self.prediction_length, 1).float().to(self.device)
            dec_inp = torch.cat([seq_x[:, -self.args.label_len:, :], dec_inp], dim=1)
            
            with torch.no_grad():
                outputs = self.model(seq_x, seq_x_mark, dec_inp, seq_y_mark)
                preds = outputs[:, -self.prediction_length:, 0].cpu().numpy().flatten()
            
            # 构造输出 DataFrame
            item_df = pd.DataFrame({
                "item_id": item_id,
                "timestamp": future_timestamps,
                "mean": preds
            })
            results.append(item_df)
            
        return TimeSeriesDataFrame(
            pd.concat(results), 
            static_features=data.static_features
        )
