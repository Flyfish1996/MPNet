# 用途：参考 Tensor-CSPNet 的混合优化结构，按参数空间分流更新。
# geoopt.ManifoldParameter 保留现有工具包的类型；支持 Stiefel、Sphere 和 SPD。
"""普通参数用 Adam；流形参数用对应黎曼梯度与回缩/指数映射。"""

import torch
from geoopt.tensor import ManifoldParameter
from geoopt.manifolds import Stiefel


class MixOptimizer:
    def __init__(self, parameters, optimizer=torch.optim.Adam, lr=1e-3, **kwargs):
        self.parameters = list(dict.fromkeys(p for p in parameters if p.requires_grad))
        self.lr = lr
        self.manifold_parameters = [p for p in self.parameters if isinstance(p, ManifoldParameter)]
        self.other_parameters = [p for p in self.parameters if not isinstance(p, ManifoldParameter)]
        self.optim = optimizer(self.other_parameters, lr=lr, **kwargs) if self.other_parameters else None

    def zero_grad(self, set_to_none=True):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.detach_()
                parameter.grad.zero_()

    def step(self):
        if self.optim is not None:
            self.optim.step()
        with torch.no_grad():
            for parameter in self.manifold_parameters:
                if parameter.grad is None:
                    continue
                manifold = parameter.manifold
                tangent = manifold.egrad2rgrad(parameter, parameter.grad)
                if isinstance(manifold, Stiefel):
                    basis, triangular = torch.linalg.qr(parameter - self.lr * tangent, mode="reduced")
                    signs = triangular.diagonal(dim1=-2, dim2=-1).sign()
                    updated = basis * torch.where(signs == 0, torch.ones_like(signs), signs).unsqueeze(-2)
                elif manifold.name == "SymmetricPositiveDefinite":
                    updated = manifold.expmap(parameter, -self.lr * tangent)
                else:
                    updated = manifold.retr(parameter, -self.lr * tangent)
                parameter.copy_(updated)

    def state_dict(self):
        return {"optim": self.optim.state_dict() if self.optim is not None else None, "lr": self.lr}

    def load_state_dict(self, state):
        if self.optim is not None:
            self.optim.load_state_dict(state["optim"])
        self.lr = state["lr"]
