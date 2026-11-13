# 文件：models/mpnet.py
# 用途：MPNet 两频带处理、专属卷积分支、分窗矩阵池化、SPD网络及混合优化器。
# 来源：1-oldVersion/MPNetPHD_code/green_utils/MPNet.py、MPNet_NoMP.py、prepocess_in_model.py。
# 修改位置：固定频带改 FilterBank，网络及池化改 MPNet，训练默认值改 get_defaults。
# 调用：factory.create_method(dataset_name, data_info, config)，DA 在 FilterBank 之前执行。
"""公共波形 -> 两频带 -> 卷积/耦合 -> 分窗二阶矩 -> 池化 -> SPD 层 -> logits。

默认按用户给出的论文设置表：重叠频带、4窗、独立分支、未做trace归一化的二阶矩。
分支独立性是按“两个专属分支”的描述解释，尚未核验论文原文；共享开关保留。
公共 CAR、截窗和标准化沿用 data/，本方法只补充专属频带。
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch
from torch import nn
from models.spdnet_utils import BiMap, ReEig, LogEig, MixOptimizer
from models.spdnet_utils.functional import sym_logm, sym_expm


class FilterBank(nn.Module):
    """五阶 Butterworth 零相位分频，批量处理样本/通道，不改变标签。

    固定 SciPy SOS 滤波在 CPU 执行，随后返回原设备；可学习处理在网络中。
    系数由记录的 bands/order/sfreq 重建，不保存训练统计。
    """
    def __init__(self, sfreq, bands=((4, 17), (13, 40)), order=5):
        super().__init__()
        if not bands:
            raise ValueError("MPNet 至少需要一个频带")
        self.coefficients = tuple(butter(order, band, btype="bandpass", fs=sfreq,
                                         output="sos") for band in bands)

    def forward(self, x):
        if x.requires_grad:
            raise ValueError("固定 SciPy 滤波不支持输入梯度；可学习滤波应放在网络内部")
        array = x.detach().cpu().numpy().astype(np.float64, copy=False)
        filtered = np.stack([sosfiltfilt(sos, array, axis=-1)
                             for sos in self.coefficients], axis=1)
        return torch.as_tensor(np.ascontiguousarray(filtered), device=x.device, dtype=x.dtype)


class Conv1dWithConstraint(nn.Conv1d):
    @torch.no_grad()
    def project_weights(self):
        self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=1))


def window_matrices(x, n_windows, trace_normalize=False):
    """(B,F,C,T) -> (B,窗*F,C,C)，沿用原窗口优先、频带其次的顺序。

    默认计算 HH^T/(T_w-1)，不减均值，不除trace；末窗保留全部余数点。
    trace_normalize=True 仅用于旧代码的迹归一化消融。
    """
    length = x.shape[-1] // n_windows
    if length < 2:
        raise ValueError("MPNet 每个窗口至少需要两个采样点")
    if n_windows == 1:
        matrices = (x @ x.transpose(-1, -2)).unsqueeze(2) / (x.shape[-1] - 1)
    else:
        first = x[..., :length * (n_windows - 1)].reshape(
            *x.shape[:-1], n_windows - 1, length).transpose(-2, -3)
        last = x[..., length * (n_windows - 1):].unsqueeze(-3)
        matrices = torch.cat((first @ first.transpose(-1, -2) / (length - 1),
                              last @ last.transpose(-1, -2) / (last.shape[-1] - 1)), dim=2)
    if trace_normalize:
        trace = matrices.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        matrices = matrices / trace.clamp_min(torch.finfo(x.dtype).tiny)
    return matrices.transpose(1, 2).flatten(1, 2)


class MPNet(nn.Module):
    def __init__(self, n_chans, n_classes, n_bands=2, hidden_channels=60,
                 temporal_kernel=32, n_windows=4, pooling="EM", dropout=0.5,
                 shared_filters=False, cov_eps=0., trace_normalize=False):
        super().__init__()
        if pooling not in {"EM", "RM", "WEM", "WRM", "none"}:
            raise ValueError(f"未知池化：{pooling}")
        if n_windows < 1:
            raise ValueError("n_windows 至少为1")
        self.n_bands, self.n_windows, self.pooling = n_bands, n_windows, pooling
        self.shared_filters, self.cov_eps = shared_filters, cov_eps
        self.trace_normalize = trace_normalize

        def branch():
            return nn.Sequential(Conv1dWithConstraint(n_chans, hidden_channels, 1),
                                 nn.BatchNorm1d(hidden_channels),
                                 Conv1dWithConstraint(hidden_channels, hidden_channels, temporal_kernel),
                                 nn.BatchNorm1d(hidden_channels))

        # 默认两个频带各用一套卷积和BN；True 可对照旧代码的共享方式。
        self.filters = nn.ModuleList([branch() for _ in range(1 if shared_filters else n_bands)])

        self.activation = nn.Sequential(
            nn.ELU(), 
            nn.Dropout(dropout)
            )

        dimensions = [hidden_channels, max(1, hidden_channels // 2), max(1, hidden_channels // 4)]

        self.spd_layers = nn.Sequential(BiMap((dimensions[0], dimensions[1])), 
                                        ReEig(5e-4),
                                        BiMap((dimensions[1], dimensions[2])), 
                                        ReEig(5e-4),
                                        LogEig(ndim=dimensions[2]))
        
        self.node_count = n_windows * (n_bands + 1)
        if pooling in {"WEM", "WRM"}:
            self.pool_weights = nn.Parameter(torch.ones(self.node_count))
        features = dimensions[-1] * (dimensions[-1] + 1) // 2
        if pooling == "none":
            features *= self.node_count
        self.classifier = nn.Sequential(nn.Linear(features, 64), 
                                        nn.ELU(),
                                        nn.Dropout(dropout), 
                                        nn.Linear(64, n_classes))


    def forward(self, x):
        # 在共享卷积的所有调用前只投影一次，避免中途修改反传需要的权重。
        for layer in self.filters.modules():
            if isinstance(layer, Conv1dWithConstraint):
                layer.project_weights()
        filtered = [self.filters[0 if self.shared_filters else index](x[:, index])
                    for index in range(self.n_bands)]
        coupled = torch.stack(filtered, dim=1).sum(1)
        activated = torch.stack([self.activation(coupled)] +
                                [self.activation(value) for value in filtered], dim=1)
        matrices = window_matrices(activated, self.n_windows, self.trace_normalize)

        if self.cov_eps:
            matrices = matrices + self.cov_eps * torch.eye(
                matrices.shape[-1], device=x.device, dtype=x.dtype)
            
        if self.pooling == "none":
            # NoMP 原来12次调用同一个 SPDNet；这里用批量维一次完成。
            features = self.spd_layers(matrices).flatten(1)

        else:
            values = sym_logm.apply(matrices, True) if self.pooling in {"RM", "WRM"} else matrices
            if self.pooling in {"WEM", "WRM"}:
                pooled = (values * self.pool_weights.softmax(0)[None, :, None, None]).sum(1)
            else:
                pooled = values.mean(1)
            if self.pooling in {"RM", "WRM"}:
                pooled = sym_expm.apply(pooled, True)
            features = self.spd_layers(pooled)

        return self.classifier(features)


class ClassificationLoss(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits, targets):
        ce = self.criterion(logits, targets)
        return {"total": ce, "ce": ce}


def get_defaults(dataset_name):
    return {"weight_decay": 0., "recipe": "mpnet-user-table-aligned-mixed-riemannian",
            "preprocess_kwargs": {"bands": [[4, 17], [13, 40]], 
                                  "order": 5},
            "model_kwargs": {"hidden_channels": 60, 
                             "temporal_kernel": 32,
                             "n_windows": 4, 
                             "pooling": "EM", 
                             "dropout": 0.5,
                             "shared_filters": False,
                             "cov_eps": 0.,
                             "trace_normalize": False}}


def build_optimizer(parameters, config):
    return MixOptimizer(parameters, 
                        optimizer=torch.optim.Adam, 
                        lr=config["lr"],
                        weight_decay=config["weight_decay"], 
                        **config["optimizer_kwargs"])


def create_method(dataset_name, data_info, config):
    preprocessor = FilterBank(data_info["sfreq"], 
                              **config["preprocess_kwargs"])
    kwargs = {**config["model_kwargs"], 
              "n_chans": data_info["n_chans"],
              "n_classes": data_info["n_classes"], 
              "n_bands": len(preprocessor.coefficients)}
    
    if data_info["n_times"] - kwargs["temporal_kernel"] + 1 < 2 * kwargs["n_windows"]:
        raise ValueError("MPNet 时间卷积后样本不足；请减小 temporal_kernel/n_windows 或延长分类窗")
    return {"model": MPNet(**kwargs), "preprocessor": preprocessor,
            "loss_func": ClassificationLoss(**config["loss_kwargs"]),
            "optimizer_builder": build_optimizer}
