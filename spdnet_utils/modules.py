# 用途：快速 SPD 层；模型自身的卷积、小波和分类头放在 models/ 对应文件。
import math
from typing import Tuple
from einops import einsum, rearrange
from einops.layers.torch import Rearrange
import geoopt
from geoopt.manifolds import Sphere, Stiefel
from geoopt.tensor import ManifoldParameter
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Number

# 显式从本工具包导入
from . import functional


class CovariancePool(nn.Module):

    def __init__(self, alpha=None, unitvar=False):
        super().__init__()
        self.pooldim = -1
        self.chandim = -2
        self.alpha = alpha
        self.unitvar = unitvar

    def forward(self, X: Tensor) -> Tensor:
        X0 = X - X.mean(dim=self.pooldim, keepdim=True)
        if self.unitvar:
            X0 = X0 / X0.std(dim=self.pooldim, keepdim=True)
            X0.nan_to_num_(0)

        n_t = X0.shape[self.pooldim]
        denom = n_t - 1 if n_t > 1 else 1
        C = (X0 @ X0.transpose(-2, -1)) / denom
        if self.alpha is not None:
            Cd = C.diagonal(dim1=self.pooldim, dim2=self.pooldim - 1)
            Cd += self.alpha
        return C


class ReverseGradient(nn.Module):

    def __init__(self, scaling=1.0):
        super().__init__()
        self.scaling_ = scaling

    def forward(self, X: Tensor) -> Tensor:
        return functional.reverse_gradient.apply(X, self.scaling_)


class BiMap(nn.Module):

    def __init__(
        self,
        shape: Tuple[int, ...] or torch.Size,
        W0: Tensor = None,
        manifold="stiefel",
        **kwargs,
    ):
        super().__init__()
        if manifold == "stiefel":
            assert shape[-2] >= shape[-1]
            mf = Stiefel()
        elif manifold == "sphere":
            mf = Sphere()
            shape = list(shape)
            shape[-1], shape[-2] = shape[-2], shape[-1]
        else:
            raise NotImplementedError(f"Unsupported manifold: {manifold}")

        self.W = ManifoldParameter(torch.empty(shape, **kwargs), manifold=mf)
        if W0 is not None:
            with torch.no_grad():
                self.W.copy_(W0)
        else:
            self.reset_parameters()

    def forward(self, X: Tensor) -> Tensor:
        if isinstance(self.W.manifold, Sphere):
            return self.W @ X @ self.W.transpose(-2, -1)
        return self.W.transpose(-2, -1) @ X @ self.W

    @torch.no_grad()
    def reset_parameters(self):
        if isinstance(self.W.manifold, Stiefel):
            W = torch.rand(
                self.W.shape, dtype=self.W.dtype, device=self.W.device
            )
            self.W.copy_(torch.linalg.qr(W, mode="reduced")[0])
        elif isinstance(self.W.manifold, Sphere):
            W = torch.empty(
                self.W.shape, dtype=self.W.dtype, device=self.W.device
            )
            bound = math.sqrt(3) * 1.0 / W.shape[-1]
            W.uniform_(-bound, bound)
            self.W.copy_(W / W.norm(dim=-1, keepdim=True))


class ReEig(nn.Module):

    def __init__(self, threshold: Number = 5e-4):
        super().__init__()
        # 使用 register_buffer 保证阈值设备与当前模型（CPU/GPU）严格一致
        self.register_buffer(
            "threshold", torch.tensor([threshold], dtype=torch.float32)
        )

    def forward(self, X: Tensor) -> Tensor:
        return functional.sym_reeig.apply(X, self.threshold)


class LogEig(nn.Module):

    def __init__(self, ndim: int, tril: bool = True, sqrt: bool = True):
        super().__init__()
        self.tril = tril
        self.sqrt = sqrt
        self.ndim = ndim
        if self.tril:
            ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
            ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
            self.register_buffer(
                "ixs",
                torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.embed(functional.sym_logm.apply(X))

    def embed(self, X: Tensor) -> Tensor:
        if self.tril:
            x_vec = X[..., self.ixs[0], self.ixs[1]]
            if self.sqrt:
                x_vec[..., self.ndim :] *= math.sqrt(2)
        else:
            x_vec = X.flatten(start_dim=-2)
        return x_vec


class TrilEmbedder(nn.Module):

    def forward(self, X: Tensor) -> Tensor:
        ndim = X.shape[-1]
        ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)
        ixs = torch.cat((ixs_diag[None, :].tile((2, 1)), ixs_lower), dim=1)

        x_vec = X[..., ixs[0], ixs[1]]
        x_vec[..., ndim:] *= math.sqrt(2)
        return x_vec

    def inverse_transform(self, x_vec: Tensor) -> Tensor:
        ndim = int(-0.5 + math.sqrt(0.25 + 2 * x_vec.shape[-1]))
        ixs_lower = torch.tril_indices(ndim, ndim, offset=-1)
        ixs_diag = torch.arange(start=0, end=ndim, dtype=torch.long)

        X = torch.zeros(
            x_vec.shape[:-1] + (ndim, ndim),
            device=x_vec.device,
            dtype=x_vec.dtype,
        )
        X[..., ixs_lower[0], ixs_lower[1]] = x_vec[..., ndim:] / math.sqrt(2)
        X[..., ixs_lower[1], ixs_lower[0]] = x_vec[..., ndim:] / math.sqrt(2)
        X[..., ixs_diag, ixs_diag] = x_vec[..., :ndim]
        return X


class SPDDropout(nn.Module):

    def __init__(self, drop_prob=0.5, use_scaling=True, epsilon=1e-5):
        super().__init__()
        self.drop_prob = drop_prob
        self.use_scaling = use_scaling
        self.epsilon = epsilon

    def forward(self, x):
        return functional.dropout_spd(x, drop_prob=self.drop_prob, training=self.training,
                                      use_scaling=self.use_scaling, epsilon=self.epsilon)


class SPDMixUp(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        batch_size = x.shape[0]
        if self.alpha > 0:
            lams = [
                np.random.beta(self.alpha, self.alpha)
                for _ in range(batch_size)
            ]
        else:
            lams = [1.0 for _ in range(batch_size)]

        # 动态绑定设备，避免写死 .cuda()
        lams = torch.tensor(lams, device=x.device, dtype=x.dtype)
        index = torch.randperm(batch_size, device=x.device)

        x_a, x_b = x, x[index]
        mixed_x = functional.geodesic_logeuclid(x_a, x_b, alpha=lams)
        return mixed_x, lams, index


class SCMPool(nn.Module):

    def __init__(self, demean=False):
        super().__init__()
        self.demean = demean

    def forward(self, x: Tensor) -> Tensor:
        n_t = x.shape[-1]
        if self.demean:
            x = x - x.mean(dim=-1, keepdim=True)
        denom = n_t - 1 if n_t > 1 else 1
        return x @ x.transpose(-1, -2) / denom


class RemoveInterbandCovariance(nn.Module):
    """保留原 EEGSPDNet 的频带/电极掩码规则，随模型迁移设备。"""
    def __init__(self, elecs_grouped, n_filters, n_elecs, keep_inter_elec=False):
        super().__init__()
        self.register_buffer("mask", functional.create_interband_mask(
            elecs_grouped, n_filters, n_elecs, keep_inter_elec))

    def forward(self, x):
        return x * self.mask


class Shrinkage(nn.Module):
    """可学习迹收缩；使用对称单位阵目标，不引入原 GREEN 的非对称随机噪声。"""
    def __init__(self, dim, initial=-3.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(initial)))
        self.register_buffer("identity", torch.eye(dim))

    def forward(self, x):
        scale = x.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
        alpha = self.alpha.sigmoid()
        return (1 - alpha) * x + alpha * scale * self.identity


class LogMap(nn.Module):
    """GREEN 的 log 域参考中心；每个 batch 只分解一次输入矩阵。"""
    def __init__(self, dim, momentum=0.9, reg=1e-4, reference="logeuclid"):
        super().__init__()
        if reference not in {"identity", "logeuclid"}:
            raise ValueError(reference)
        self.reference, self.reg = reference, reg
        # 原 GREEN 权重为 [1, 1-momentum]，这里直接使用其归一化结果。
        self.batch_weight = (1 - momentum) / (2 - momentum)
        self.register_buffer("running_log", torch.zeros(dim, dim))

    def forward(self, x):
        logged = functional.logm(x, self.reg)
        if self.reference == "identity":
            return logged
        if self.training:
            with torch.no_grad():
                self.running_log.lerp_(logged.detach().mean(0), self.batch_weight)
        return logged - self.running_log


class StiefelAdam:
    """历史 Stiefel 包装器；新方法统一使用 optimizers.MixOptimizer。"""

    def __init__(self, optimizer, set_zero=True):
        self.optimizer = optimizer
        self.state = {}
        self.set_zero = set_zero

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)
        self.state.clear()

    @staticmethod
    def _orthogonal_projection(x, u):
        return u - x @ geoopt.linalg.sym(x.transpose(-1, -2) @ u)

    @staticmethod
    def _retraction(x, u):
        q, r = torch.linalg.qr(x + u)
        unflip = geoopt.linalg.extract_diag(r).sign().add(0.5).sign()
        q *= unflip[..., None, :]
        return q

    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for point in group["params"]:
                grad = point.grad
                if grad is None:
                    continue

                if isinstance(point, ManifoldParameter):
                    if id(point) not in self.state:
                        self.state[id(point)] = point.data.clone()
                    else:
                        self.state[id(point)].fill_(0).add_(point.data.clone())

                    egrad2rgrad = self._orthogonal_projection(
                        grad.data, point.data
                    )
                    grad.data.fill_(0).add_(egrad2rgrad)

                    if self.set_zero:
                        point.data.fill_(0)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for point in group["params"]:
                grad = point.grad
                if grad is None:
                    continue

                if isinstance(point, ManifoldParameter):
                    trans = self._retraction(point.data, self.state[id(point)])
                    point.data.fill_(0).add_(trans)

        return loss
