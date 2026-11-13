# 用途：导出当前工具包实际提供的批量 SPD 算子、网络层及混合优化器。

# 1. 导出 functional 的关键算子
from .batchnorm import (
    AdaMomSPDBatchNorm,
    BatchNormDispersion,
    BatchNormTestStatsMode,
    SPDBatchNormImpl,
    SymmetricPositiveDefinite,
)
from .functional import (
    dropout_spd,
    ensure_sym,
    geodesic_logeuclid,
    spd_2point_interpolation,
    spd_mean_kracher_flow,
    sym2vec_n_unique,
    sym_abseig,
    sym_expm,
    sym_invm,
    sym_invsqrtm,
    sym_invsqrtm2,
    sym_logm,
    sym_powm,
    sym_reeig,
    sym_sqrtm,
    logm,
)

# 2. 导出 modules 的网络层与优化器
from .modules import (
    BiMap,
    CovariancePool,
    LogEig,
    ReEig,
    RemoveInterbandCovariance,
    ReverseGradient,
    SCMPool,
    SPDDropout,
    SPDMixUp,
    StiefelAdam,
    TrilEmbedder,
    Shrinkage,
    LogMap,
)

# 3. 欧氏参数和流形参数分别更新；供独立方法文件构造优化器。
from .optimizers import MixOptimizer
