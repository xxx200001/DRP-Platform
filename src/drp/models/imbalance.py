"""
样本极度不均衡处理（规范 3.2）。

规范原话："高危样本少，必须使用 focal loss、分层抽样、权重平衡"。
本模块把三种手段全部实现，并明确各自的适用位置与代价 ——
不均衡处理不是"三个都开就更准"，用错了反而掉点：

【手段 1】scale_pos_weight / 类权重（默认首选）
    把正样本的损失权重放大到 n_neg/n_pos。实现最简单、对排序指标
    （AUC）最稳。代价：输出概率不再是标定概率（整体被抬高），
    必须配合概率校准（lgbm.py 的 calibration）才能用于风险分层。

【手段 2】focal loss（正样本 < 3% 或难例主导时用）
    在类权重之外额外把"已经分对的容易样本"的损失打折（(1-pt)^γ），
    迫使模型把容量花在难例上。对极端不均衡 + 难例边界复杂的场景
    优于纯权重法。代价同上且更甚：raw score 完全失去概率含义，
    校准从"建议"变成"强制"（lgbm.py 会在 focal 模式下拒绝关闭校准）。
    仅 LightGBM 后端支持（需要自定义目标函数）。

【手段 3】多数类分层欠采样（仅在数据量大到训不动时用）
    保留全部正样本，负样本按患者级抽样到指定比例。
    代价：负样本信息有损失；且改变了先验，校准更加必要。
    3 万条量级完全没必要欠采样 —— 提供它是为了百万级线上回流样本
    （规范 1.3）到来之后的迭代训练。

【所有手段共同的红线】
    只允许作用于训练集。对验证/测试集做任何重加权或重采样，
    评估结果就不再反映真实人群 —— 这属于 leakage.py 里的泄露 4 变体。
    undersample_majority 因此强制要求传入训练掩码内的标签，并在
    docstring 与日志里反复强调。

关于 focal loss 的 Hessian（实现说明，改这段代码前必读）：
    focal loss 对 raw score 的二阶导解析式冗长且分支多，是社区实现里
    出 bug 概率最高的地方（常见后果：训练发散或早停失效，且很难察觉）。
    本实现采用【解析一阶导 + 中心差分二阶导】：
        hess(z) ≈ (grad(z+ε) - grad(z-ε)) / 2ε
    每轮迭代多算两次 sigmoid，向量化后开销可忽略，换来的是数值上
    与解析梯度严格自洽（单元测试用有限差分对损失本身做了双重验证）。
    Hessian 下限裁剪到 1e-6 保证 LightGBM 的牛顿步稳定。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 手段 1：权重平衡
# ---------------------------------------------------------------------------
def compute_scale_pos_weight(y: np.ndarray | pd.Series, cap: float = 100.0) -> float:
    """
    n_neg / n_pos，直接喂给 LightGBM 的 scale_pos_weight。

    cap：极端情况下（阳性率 <0.5%）不加盖的权重会让单个正样本的梯度
    大到破坏数值稳定，实践上超过 ~100 的权重收益也趋于零，此时应该
    换 focal loss 而不是继续加权重。
    """
    y = np.asarray(y, dtype=float)
    _check_binary(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0:
        raise ValueError("训练标签中没有正样本，无法训练。请检查标签构建与切分。")
    spw = n_neg / n_pos
    if spw > cap:
        logger.warning(
            "scale_pos_weight=%.1f 超过上限 %.0f，已截断。阳性率仅 %.3f%%，"
            "建议改用 focal loss（LGBMConfig.imbalance='focal'）。",
            spw, cap, 100 * n_pos / (n_pos + n_neg),
        )
        spw = cap
    return spw


def balanced_sample_weight(y: np.ndarray | pd.Series) -> np.ndarray:
    """逐样本权重版的类平衡（正样本权重 = n_neg/n_pos，负样本 = 1）。
    供不支持 scale_pos_weight 参数的后端（如 sklearn 回退）使用。"""
    y = np.asarray(y, dtype=float)
    spw = compute_scale_pos_weight(y)
    w = np.ones_like(y, dtype=float)
    w[y == 1] = spw
    return w


# ---------------------------------------------------------------------------
# 手段 3：多数类欠采样
# ---------------------------------------------------------------------------
def undersample_majority(
    y_train: np.ndarray | pd.Series,
    max_neg_pos_ratio: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """
    多数类（负样本）欠采样。返回【训练集内部的保留位置索引】。

    只允许作用于训练集 —— 传进来的必须是切分后的训练标签，
    绝不能是全量标签（原因见模块 docstring 红线段）。

    保留全部正样本；负样本随机抽到 max_neg_pos_ratio 倍。
    若当前比例已低于阈值则原样返回（不做任何事，幂等）。
    """
    y = np.asarray(y_train, dtype=float)
    _check_binary(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    target_neg = int(len(pos_idx) * max_neg_pos_ratio)
    if len(neg_idx) <= target_neg:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    keep_neg = rng.choice(neg_idx, size=target_neg, replace=False)
    keep = np.sort(np.concatenate([pos_idx, keep_neg]))
    logger.info(
        "多数类欠采样: 负样本 %d -> %d (neg:pos = %.1f:1)。"
        "注意训练先验已改变，概率校准必须开启。",
        len(neg_idx), target_neg, max_neg_pos_ratio,
    )
    return keep


# ---------------------------------------------------------------------------
# 手段 2：focal loss（LightGBM 自定义目标函数）
# ---------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """数值稳定 sigmoid：正负分支分别处理，避免 exp 溢出。"""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class FocalBinary:
    """
    二分类 focal loss，可直接作为 LightGBM 的 objective 回调::

        booster = lgb.train({...}, dtrain, fobj 由 objective= 传入,
                            feval=focal.eval_metric)

    FL(z) = -α_t · (1 - p_t)^γ · log(p_t)
        p = σ(z);  p_t = y·p + (1-y)·(1-p);  α_t = y·α + (1-y)·(1-α)

    α : 类权重项。注意 α 与 scale_pos_weight 二选一，叠加会双重加权。
    γ : 难例聚焦强度。γ=0 退化为加权交叉熵；常用 1.5~3。

    使用 focal 后 raw score 不再有概率含义（见模块 docstring），
    predict 时必须 σ(z) 后再过校准器 —— lgbm.py 已强制处理。
    """

    alpha: float = 0.25
    gamma: float = 2.0
    eps: float = 1e-12
    fd_eps: float = 1e-4  # Hessian 中心差分步长

    # ---- 损失本身（供 eval 与单元测试） ----
    def loss(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = _sigmoid(np.asarray(z, dtype=float))
        y = np.asarray(y, dtype=float)
        pt = np.clip(y * p + (1 - y) * (1 - p), self.eps, 1 - self.eps)
        at = y * self.alpha + (1 - y) * (1 - self.alpha)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    # ---- 解析一阶导 ----
    def grad(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        y = np.asarray(y, dtype=float)
        p = _sigmoid(z)
        pt = np.clip(y * p + (1 - y) * (1 - p), self.eps, 1 - self.eps)
        at = y * self.alpha + (1 - y) * (1 - self.alpha)
        # dFL/dpt = α_t [ γ(1-pt)^{γ-1} ln(pt) - (1-pt)^γ / pt ]
        dl_dpt = at * (
            self.gamma * (1 - pt) ** (self.gamma - 1) * np.log(pt)
            - (1 - pt) ** self.gamma / pt
        )
        # dpt/dz = (2y-1)·p(1-p)
        dpt_dz = (2 * y - 1) * p * (1 - p)
        return dl_dpt * dpt_dz

    # ---- 中心差分二阶导（原因见模块 docstring） ----
    def hess(self, z: np.ndarray, y: np.ndarray) -> np.ndarray:
        e = self.fd_eps
        h = (self.grad(z + e, y) - self.grad(z - e, y)) / (2 * e)
        return np.maximum(h, 1e-6)

    # ---- LightGBM 接口 ----
    def __call__(self, z: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        """objective 回调：入参为 (raw_score, lgb.Dataset)，返回 (grad, hess)。"""
        y = dataset.get_label()
        return self.grad(z, y), self.hess(z, y)

    def eval_metric(self, z: np.ndarray, dataset) -> tuple[str, float, bool]:
        """feval 回调，用于训练日志/早停监控 focal 损失本身。"""
        y = dataset.get_label()
        return "focal_loss", float(np.mean(self.loss(z, y))), False


# ---------------------------------------------------------------------------
def _check_binary(y: np.ndarray) -> None:
    if np.isnan(y).any():
        raise ValueError(
            "标签含 NaN。删失样本必须先用 labels.usable_mask() 剔除，"
            "禁止把 NaN 标签直接送进模型/权重计算。"
        )
    bad = ~np.isin(y, (0.0, 1.0))
    if bad.any():
        raise ValueError(f"标签必须是 0/1，发现异常值: {np.unique(y[bad])[:5]}")
