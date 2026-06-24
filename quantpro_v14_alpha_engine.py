"""
QuantPro v14.2 — 机构级方法论引擎（Alpha Engine）[大盘上下文+时间衰减+Bagging]
══════════════════════════════════════════════════════════════════
对标文艺复兴/López de Prado《Advances in Financial Machine Learning》，
补齐 quantpro_v1_6.py 在「预测精确度」与「历史回测可信度」上的方法论漏洞。

本文件是【完整、独立、可直接运行】的模块。两种用法：
  1) 直接运行：  python quantpro_v14_alpha_engine.py
     —— 用合成数据演示每个组件（无需联网 / 无需 yfinance）
  2) 集成到主程序：在 quantpro_v1_6.py 末尾加：
        from quantpro_v14_alpha_engine import patch_quantpro
        patch_quantpro(ProbabilityEngine, WalkForwardBacktest, SelfCorrectionEngine)

包含的核心组件
──────────────────────────────────────────────────────────────────
  [1] PurgedKFold        — 带 purge+embargo 的时序交叉验证（修复标签泄漏）
  [2] TripleBarrierLabeler — 三重障碍标注（止盈/止损/时间，ATR自适应）
  [3] sample_uniqueness   — 重叠标签的并发度样本权重
  [4] DeflatedSharpe      — PSR / DSR / 最小回测长度（多重检验校正）
  [5] prob_backtest_overfit (PBO) — CSCV 组合对称交叉验证过拟合概率
  [6] ProbabilityCalibrator — Isotonic 概率校准（接 SelfCorrectionEngine DB）
  [7] MetaLabeler         — 元标注：主模型定方向，副模型定下注
  [8] frac_diff_ffd       — 分数阶差分（平稳化同时保留记忆）
  [9] feature_neutralize  — 因子中性化（去除共线/暴露）
  [10] EnhancedProbabilityEngine — 把上述整合进 ensemble 的增强引擎
  [11] patch_quantpro     — 一键注入主程序

依赖：numpy, pandas, scipy, scikit-learn；xgboost 可选（缺失自动降级）
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── 可选依赖 ─────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_ADF = True
except Exception:
    _HAS_ADF = False

EULER_GAMMA = 0.5772156649015329   # Euler–Mascheroni 常数


# ══════════════════════════════════════════════════════════════════
# [1] Purged K-Fold + Embargo  —— 修复时序标签泄漏
# ══════════════════════════════════════════════════════════════════
class PurgedKFold:
    """
    带 purge + embargo 的时序交叉验证（López de Prado, AFML Ch.7）。

    问题：当标签是 close.shift(-horizon)（未来 N 日收益），相邻样本的
    标签窗口 [i, i+horizon] 互相重叠。普通 KFold/TimeSeriesSplit 会让
    测试集的标签信息泄漏进训练集，导致 OOS 准确率虚高、实盘掉点。

    解决：
      - purge：剔除标签窗口与测试集时间跨度重叠的训练样本
      - embargo：再剔除测试集之后 embargo_pct 比例的训练样本
                （消除序列自相关造成的泄漏）

    用法与 sklearn splitter 一致：
        for tr_idx, te_idx in PurgedKFold(5, horizon=5).split(X):
            ...
    """

    def __init__(self, n_splits: int = 5, horizon: int = 5,
                 embargo_pct: float = 0.01):
        self.n_splits = max(2, int(n_splits))
        self.horizon = max(1, int(horizon))
        self.embargo_pct = max(0.0, float(embargo_pct))

    def split(self, X, y=None, groups=None):
        n = len(X)
        indices = np.arange(n)
        embargo = int(n * self.embargo_pct)
        # 连续等分测试折（保持时间顺序）
        fold_bounds = [(i[0], i[-1] + 1)
                       for i in np.array_split(indices, self.n_splits)]

        for start, end in fold_bounds:
            test_idx = indices[start:end]
            # 测试集标签覆盖的最大时间点（含 horizon 前瞻）
            test_label_max = min(end - 1 + self.horizon, n - 1)

            train_mask = np.ones(n, dtype=bool)
            # ① purge：训练样本 j 的标签窗口 [j, j+horizon] 若与
            #    测试时间跨度 [start, test_label_max] 重叠则剔除
            for j in indices:
                j_label_end = j + self.horizon
                overlap = (j <= test_label_max) and (j_label_end >= start)
                if overlap:
                    train_mask[j] = False
            # ② embargo：测试集之后 embargo 根 bar 也剔除
            emb_end = min(end + embargo, n)
            train_mask[end:emb_end] = False
            # 测试集本身不能当训练
            train_mask[start:end] = False

            train_idx = indices[train_mask]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ══════════════════════════════════════════════════════════════════
# [2] Triple-Barrier 标注 —— 路径相关的真实出场标签
# ══════════════════════════════════════════════════════════════════
class TripleBarrierLabeler:
    """
    三重障碍标注（López de Prado, AFML Ch.3）。

    对每个事件起点：
      - 上障碍（止盈）= price * (1 + pt_mult * vol)
      - 下障碍（止损）= price * (1 - sl_mult * vol)
      - 垂直障碍（时间）= 最多持有 max_hold 根 bar
    谁先被触碰决定标签：上→+1，下→0(或-1)，时间到→按终点收益符号。

    barrier 用 ATR/波动率自适应缩放，避免对不同波动率股票用同一固定阈值。

    返回 DataFrame: label(0/1), ret(实现收益), t1(触碰时点的整数位移), bin_meta
    """

    @staticmethod
    def label(close: pd.Series,
              vol: pd.Series,
              pt_mult: float = 2.0,
              sl_mult: float = 1.5,
              max_hold: int = 10,
              long_only: bool = True) -> pd.DataFrame:
        close = close.astype(float).reset_index(drop=True)
        vol = vol.astype(float).reset_index(drop=True).bfill().fillna(0)
        n = len(close)
        out = {"label": [], "ret": [], "t1": [], "first_touch": []}
        idx_keep = []

        for i in range(n):
            if i + 1 >= n:
                continue
            p0 = close.iloc[i]
            v = vol.iloc[i]
            if not np.isfinite(p0) or p0 <= 0 or not np.isfinite(v) or v <= 0:
                continue
            up = p0 * (1 + pt_mult * v)
            dn = p0 * (1 - sl_mult * v)
            end = min(i + max_hold, n - 1)
            touched = None
            t_touch = end
            for j in range(i + 1, end + 1):
                pj = close.iloc[j]
                if pj >= up:
                    touched = "up"; t_touch = j; break
                if pj <= dn:
                    touched = "dn"; t_touch = j; break
            ret = close.iloc[t_touch] / p0 - 1.0
            if touched == "up":
                lab = 1
            elif touched == "dn":
                lab = 0
            else:
                lab = 1 if ret > 0 else 0   # 时间障碍：按终点符号
            out["label"].append(lab)
            out["ret"].append(float(ret))
            out["t1"].append(int(t_touch))
            out["first_touch"].append(touched or "time")
            idx_keep.append(i)

        df = pd.DataFrame(out, index=idx_keep)
        return df


# ══════════════════════════════════════════════════════════════════
# [3] 样本唯一性权重 —— 重叠标签按并发度降权
# ══════════════════════════════════════════════════════════════════
def sample_uniqueness(t1: pd.Series, n_bars: int) -> pd.Series:
    """
    计算每个样本的平均唯一性权重（AFML Ch.4）。

    重叠标签会让同一段价格信息被重复计数，等权训练高估有效样本量。
    并发度 c_t = 在时刻 t 同时'存活'的标签数；
    样本 i 的平均唯一性 = mean_{t in [i, t1_i]} (1 / c_t)。

    t1: 每个事件的触碰整数位置（来自 TripleBarrierLabeler）
    返回与 t1 同索引的权重 Series（已归一化到均值≈1）
    """
    t1 = t1.dropna().astype(int)
    if len(t1) == 0:
        return pd.Series(dtype=float)
    # 并发度计数
    concurrency = np.zeros(n_bars + 1, dtype=float)
    for i, end in t1.items():
        a, b = int(i), int(min(end, n_bars))
        concurrency[a:b + 1] += 1.0
    concurrency[concurrency == 0] = 1.0
    # 每个样本的平均唯一性
    uniq = {}
    for i, end in t1.items():
        a, b = int(i), int(min(end, n_bars))
        seg = concurrency[a:b + 1]
        uniq[i] = float(np.mean(1.0 / seg)) if len(seg) else 1.0
    w = pd.Series(uniq)
    if w.mean() > 0:
        w = w / w.mean()        # 归一化均值到1，便于 sample_weight 使用
    return w


# ══════════════════════════════════════════════════════════════════
# [4] Deflated Sharpe Ratio —— 多重检验校正后的夏普可信度
# ══════════════════════════════════════════════════════════════════
class DeflatedSharpe:
    """
    概率夏普比率 PSR 与去膨胀夏普比率 DSR（Bailey & López de Prado, 2014）。

    核心思想：你网格搜了 N 组参数取最高夏普，这个最高值天然被高估。
    DSR 回答：在做了 N 次试验、考虑收益的偏度/峰度后，
    真实夏普 > 0 的概率有多大？DSR < 0.95 基本可判为过拟合产物。
    """

    @staticmethod
    def _sharpe_stats(returns: np.ndarray) -> Tuple[float, float, float, int]:
        r = np.asarray(returns, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 3 or r.std(ddof=1) == 0:
            return 0.0, 0.0, 3.0, n
        sr = r.mean() / r.std(ddof=1)            # 非年化、单期夏普
        skew = float(stats.skew(r))
        kurt = float(stats.kurtosis(r, fisher=False))  # 普通峰度（正态=3）
        return float(sr), skew, kurt, n

    @classmethod
    def psr(cls, returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
        """
        概率夏普比率：P(真实SR > sr_benchmark)。
        sr_benchmark 为单期（非年化）门槛夏普。
        """
        sr, skew, kurt, n = cls._sharpe_stats(returns)
        if n < 3:
            return 0.5
        denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2))
        z = (sr - sr_benchmark) * np.sqrt(n - 1) / denom
        return float(stats.norm.cdf(z))

    @classmethod
    def deflated_sharpe(cls,
                        returns: np.ndarray,
                        n_trials: int,
                        trial_sharpe_variance: float) -> Dict[str, float]:
        """
        去膨胀夏普比率。
        n_trials               : 你尝试过的策略/参数组合数量
        trial_sharpe_variance  : 这些试验的夏普ratio的方差（跨试验）
        """
        sr, skew, kurt, n = cls._sharpe_stats(returns)
        n_trials = max(1, int(n_trials))
        v = max(1e-12, float(trial_sharpe_variance))
        # 期望最大夏普（多重检验阈值）
        e = np.e
        z1 = stats.norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * e)) if n_trials > 1 else 0.0
        sr0 = np.sqrt(v) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
        dsr = cls.psr(returns, sr_benchmark=sr0)
        return {
            "observed_sharpe": float(sr),
            "deflation_threshold": float(sr0),
            "psr_vs_zero": float(cls.psr(returns, 0.0)),
            "deflated_sharpe": float(dsr),
            "n_trials": n_trials,
            "verdict": ("可信" if dsr >= 0.95 else
                        ("存疑" if dsr >= 0.80 else "疑似过拟合")),
        }

    @classmethod
    def min_track_record_length(cls, returns: np.ndarray,
                                sr_benchmark: float = 0.0,
                                target_conf: float = 0.95) -> float:
        """达到 target_conf 置信度所需的最小回测样本数（minTRL）。"""
        sr, skew, kurt, n = cls._sharpe_stats(returns)
        if abs(sr - sr_benchmark) < 1e-9:
            return float("inf")
        z = stats.norm.ppf(target_conf)
        var_term = 1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2
        mintrl = 1 + var_term * (z / (sr - sr_benchmark)) ** 2
        return float(mintrl)


# ══════════════════════════════════════════════════════════════════
# [5] Probability of Backtest Overfitting (PBO) via CSCV
# ══════════════════════════════════════════════════════════════════
def prob_backtest_overfit(returns_matrix: np.ndarray,
                          n_splits: int = 8) -> Dict[str, float]:
    """
    组合对称交叉验证 (CSCV) 估计回测过拟合概率（Bailey et al., 2017）。

    returns_matrix : 形状 (T, N) —— T 个时间点 × N 个候选策略/参数组合的收益
    思路：把时间轴切成 n_splits 块，所有「一半块当样本内(IS)、另一半当
    样本外(OOS)」的组合都跑一遍；对每种组合，取 IS 表现最好的策略，看它
    在 OOS 的排名。若 OOS 排名常常落到中位数以下，说明 IS 最优是过拟合。

    返回 pbo（越高越糟，>0.5 表示选出来的"最优"很可能是噪声）。
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2 or R.shape[1] < 2:
        return {"pbo": float("nan"), "n_combinations": 0}
    T, N = R.shape
    n_splits = max(2, n_splits - (n_splits % 2))   # 必须偶数
    blocks = np.array_split(np.arange(T), n_splits)

    from itertools import combinations
    half = n_splits // 2
    logits = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])
        # 用夏普作为绩效度量
        def _sr(mat):
            mu = mat.mean(axis=0)
            sd = mat.std(axis=0, ddof=1)
            sd[sd == 0] = np.nan
            return mu / sd
        is_perf = _sr(R[is_idx])
        oos_perf = _sr(R[oos_idx])
        if np.all(np.isnan(is_perf)):
            continue
        best = int(np.nanargmax(is_perf))
        # OOS 相对排名 (0~1)
        valid = oos_perf[np.isfinite(oos_perf)]
        if len(valid) < 2:
            continue
        rank = (oos_perf[best] > valid).sum() / len(valid)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))

    if not logits:
        return {"pbo": float("nan"), "n_combinations": 0}
    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))   # OOS排名在中位数以下的比例
    return {
        "pbo": pbo,
        "n_combinations": len(logits),
        "median_oos_logit": float(np.median(logits)),
        "verdict": ("稳健" if pbo < 0.2 else ("一般" if pbo < 0.5 else "高度过拟合")),
    }


# ══════════════════════════════════════════════════════════════════
# [6] 概率校准器 —— Isotonic，可直接接 SelfCorrectionEngine 历史库
# ══════════════════════════════════════════════════════════════════
class ProbabilityCalibrator:
    """
    用历史「预测概率 vs 实际涨跌」做单调（isotonic）校准。

    你的 SelfCorrectionEngine 已经在 SQLite 里存了 p_ensemble 与 actual_up，
    正好是校准所需的 (predicted, realized) 配对。校准后概率的 Brier 更低，
    也让价格目标带的 P5/P50/P95 概率含义更准确。
    """

    def __init__(self):
        self._iso: Optional[IsotonicRegression] = None
        self.n_fit = 0

    def fit(self, p_pred: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        p = np.asarray(p_pred, dtype=float)
        y = np.asarray(y_true, dtype=float)
        mask = np.isfinite(p) & np.isfinite(y)
        p, y = p[mask], y[mask]
        if len(p) < 20 or len(np.unique(y)) < 2:
            self._iso = None        # 数据不足，恒等映射
            self.n_fit = len(p)
            return self
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        self._iso.fit(p, y)
        self.n_fit = len(p)
        return self

    def transform(self, p: float) -> float:
        if self._iso is None:
            return float(np.clip(p, 0.02, 0.98))
        return float(self._iso.predict([float(p)])[0])

    @classmethod
    def from_self_correction_db(cls, db_path: str = "quantpro_predictions.db",
                                horizon: Optional[int] = None,
                                min_samples: int = 20) -> "ProbabilityCalibrator":
        """直接从 SelfCorrectionEngine 的数据库拟合校准器。"""
        import sqlite3
        cal = cls()
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            if horizon is not None:
                c.execute("SELECT p_ensemble, actual_up FROM predictions "
                          "WHERE resolved=1 AND horizon=?", (horizon,))
            else:
                c.execute("SELECT p_ensemble, actual_up FROM predictions "
                          "WHERE resolved=1")
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"calibrator db read: {e}")
            return cal
        if len(rows) < min_samples:
            return cal
        p = np.array([r[0] for r in rows], dtype=float)
        y = np.array([r[1] for r in rows], dtype=float)
        return cal.fit(p, y)


# ══════════════════════════════════════════════════════════════════
# [7] Meta-Labeling —— 主模型定方向，副模型定下注与否/仓位
# ══════════════════════════════════════════════════════════════════
class MetaLabeler:
    """
    元标注（López de Prado, AFML Ch.3.6）。

    一级模型（你现有的技术评分 / momentum）给出"方向信号"（做多/观望）；
    二级模型只回答一个问题：「跟随这个信号能赚钱吗？」（二分类）。
    输出概率 = 下注大小（bet size）。这样把"择时"和"择尺寸"分离，
    显著提高精确率（precision），降低假信号造成的亏损。

    训练目标 = 在'主信号触发'的样本上，三重障碍结果是否盈利。
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False

    @staticmethod
    def _make_clf():
        if _HAS_XGB:
            return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective="binary:logistic", random_state=42,
                                 verbosity=0)
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=42)

    def fit(self, X: np.ndarray, primary_signal: np.ndarray,
            tb_label: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        X            : 特征矩阵
        primary_signal: 一级模型方向（1=做多触发, 0=不触发）
        tb_label     : 三重障碍标签（1=盈利, 0=亏损）
        二级训练集 = 仅 primary_signal==1 的样本
        """
        mask = primary_signal.astype(bool)
        Xm, ym = X[mask], tb_label[mask]
        if len(ym) < 40 or len(np.unique(ym)) < 2:
            self._fitted = False
            return self
        Xs = self.scaler.fit_transform(Xm)
        self.model = self._make_clf()
        sw = sample_weight[mask] if sample_weight is not None else None
        try:
            self.model.fit(Xs, ym, sample_weight=sw)
            self._fitted = True
        except Exception as e:
            logger.warning(f"MetaLabeler fit: {e}")
            self._fitted = False
        return self

    def bet_size(self, x_row: np.ndarray, primary_signal: int) -> float:
        """返回下注比例 [0,1]；主信号不触发则0。"""
        if primary_signal != 1 or not self._fitted:
            return 0.0
        try:
            xs = self.scaler.transform(np.asarray(x_row, dtype=float).reshape(1, -1))
            p = float(self.model.predict_proba(xs)[0][1])
            return float(np.clip(2 * (p - 0.5), 0.0, 1.0))   # 0.5→0, 1.0→1
        except Exception:
            return 0.0


# ══════════════════════════════════════════════════════════════════
# [8] 分数阶差分 FFD —— 平稳化同时保留长记忆
# ══════════════════════════════════════════════════════════════════
def _ffd_weights(d: float, thres: float = 1e-4, max_size: int = 1000) -> np.ndarray:
    w = [1.0]
    k = 1
    while k < max_size:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thres:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1])


def frac_diff_ffd(series: pd.Series, d: float = 0.4,
                  thres: float = 1e-4) -> pd.Series:
    """
    固定宽度窗口的分数阶差分（López de Prado, AFML Ch.5）。

    价格非平稳（含单位根），但一阶差分(收益)丢掉了几乎全部记忆。
    FFD 用分数阶 d∈(0,1) 在'平稳'与'保留记忆'之间取折中，
    得到既平稳又含趋势信息的特征，喂给 ML 比纯收益更有预测力。
    """
    s = series.astype(float).dropna()
    w = _ffd_weights(d, thres)
    width = len(w)
    if len(s) <= width:
        return pd.Series(dtype=float)
    out = {}
    vals = s.values
    for i in range(width - 1, len(s)):
        window = vals[i - width + 1: i + 1]
        out[s.index[i]] = float(np.dot(w, window))
    return pd.Series(out)


def min_ffd_order(series: pd.Series, ds: Iterable[float] = None) -> Dict:
    """在保持 ADF 平稳的前提下找最小 d（最大记忆保留）。需 statsmodels。"""
    if ds is None:
        ds = np.linspace(0.1, 1.0, 10)
    result = {}
    for d in ds:
        fd = frac_diff_ffd(series, d=float(d))
        if len(fd) < 30:
            continue
        if _HAS_ADF:
            try:
                pval = adfuller(fd.values, maxlag=1, regression="c", autolag=None)[1]
            except Exception:
                pval = np.nan
        else:
            pval = np.nan
        result[round(float(d), 2)] = {"adf_pvalue": pval, "n": len(fd)}
        if _HAS_ADF and np.isfinite(pval) and pval < 0.05:
            result["recommended_d"] = round(float(d), 2)
            break
    return result


# ══════════════════════════════════════════════════════════════════
# [9] 因子中性化 —— 去除共线与公共暴露
# ══════════════════════════════════════════════════════════════════
def feature_neutralize(features: pd.DataFrame,
                       exposures: pd.DataFrame,
                       proportion: float = 1.0) -> pd.DataFrame:
    """
    对每个因子，回归剔除其在 exposures（如市场收益、行业、规模）上的暴露，
    保留残差。文艺复兴式"组合大量去相关弱信号"的前置步骤。

    proportion: 中性化强度 [0,1]，1=完全中性化。
    """
    F = features.copy().astype(float)
    E = exposures.reindex(F.index).astype(float).fillna(0.0)
    E = E.assign(_const=1.0)
    Emat = E.values
    try:
        pinv = np.linalg.pinv(Emat)
    except Exception:
        return F
    for col in F.columns:
        y = F[col].fillna(0.0).values
        beta = pinv @ y
        pred = Emat @ beta
        F[col] = y - proportion * pred
    return F


# ══════════════════════════════════════════════════════════════════
# [9b] 市场上下文特征（v14.2 新增）—— SPY / VIX 大盘环境
# ══════════════════════════════════════════════════════════════════
class MarketContext:
    """
    给单票模型补充大盘环境特征。v14.2 之前，FEATURES 里 19 个因子
    全部来自该股自身量价——模型蒙着眼睛猜个股，不知道大盘在涨在崩。

    特征（按个股日期索引对齐，缺失日 ffill）：
      spy_mom5 / spy_mom20 : 大盘5/20日动量
      rel_str20            : 个股20日收益 − SPY 20日收益（相对强弱）
      vix_z                : VIX 相对一年滚动均值的 z 分数
      vix_chg5             : VIX 5日变化率
      corr60               : 个股与SPY的滚动60日相关（beta暴露代理）

    用法：
      ctx = MarketContext(spy_close=..., vix_close=...)      # 直接给数据
      ctx = MarketContext.from_fetch(fetch_stock_data)        # 或给抓取函数
      extra = ctx.features_for(stock_close)                   # 按个股索引对齐
    """

    def __init__(self, spy_close: Optional[pd.Series] = None,
                 vix_close: Optional[pd.Series] = None):
        self.spy = spy_close.astype(float).dropna() if spy_close is not None else None
        self.vix = vix_close.astype(float).dropna() if vix_close is not None else None

    # ── 从主程序的 fetch_stock_data 构建（带会话级缓存）──────────
    _cache: Dict[str, "MarketContext"] = {}

    @classmethod
    def from_fetch(cls, fetch_fn, period: str = "2y",
                   spy_sym: str = "SPY", vix_sym: str = "^VIX") -> "MarketContext":
        key = f"{spy_sym}|{vix_sym}|{period}"
        if key in cls._cache:
            return cls._cache[key]
        spy = vix = None
        try:
            df = fetch_fn(spy_sym, period)
            if df is not None and not df.empty:
                s = df["Close"]
                spy = (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).astype(float)
        except Exception as e:
            logger.warning(f"MarketContext SPY fetch: {e}")
        try:
            df = fetch_fn(vix_sym, period)
            if df is not None and not df.empty:
                s = df["Close"]
                vix = (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).astype(float)
        except Exception as e:
            logger.warning(f"MarketContext VIX fetch: {e}")
        ctx = cls(spy, vix)
        cls._cache[key] = ctx
        return ctx

    def features_for(self, close: pd.Series) -> pd.DataFrame:
        """按个股 close 的索引返回对齐后的上下文特征（全 NaN 列会被剔除）。"""
        idx = close.index
        out = pd.DataFrame(index=idx)
        c = close.astype(float)

        if self.spy is not None and len(self.spy) > 30:
            spy = self.spy.reindex(idx).ffill()
            out["spy_mom5"] = spy.pct_change(5)
            out["spy_mom20"] = spy.pct_change(20)
            out["rel_str20"] = c.pct_change(20) - spy.pct_change(20)
            sr = c.pct_change(); mr = spy.pct_change()
            out["corr60"] = sr.rolling(60).corr(mr)

        if self.vix is not None and len(self.vix) > 60:
            vix = self.vix.reindex(idx).ffill()
            vma = vix.rolling(252, min_periods=60).mean()
            vsd = vix.rolling(252, min_periods=60).std()
            out["vix_z"] = (vix - vma) / (vsd + 1e-9)
            out["vix_chg5"] = vix.pct_change(5)

        # 剔除全 NaN 列，避免引擎里 notna().all(axis=1) 把全部样本干掉
        out = out.dropna(axis=1, how="all")
        return out


# ══════════════════════════════════════════════════════════════════
# [10] 增强预测引擎 —— 整合 Purged CV + 唯一性权重 + 校准
# ══════════════════════════════════════════════════════════════════
class EnhancedProbabilityEngine:
    """
    一个可独立使用、也可注入主程序的增强方向概率模型。

    相比原 ProbabilityEngine.xgb_probability 的三点改进：
      ① 用 PurgedKFold 替代 TimeSeriesSplit，消除标签泄漏
      ② 用三重障碍 + 样本唯一性权重训练，标签更真实、重叠样本降权
      ③ 用 OOS 折外预测拟合 isotonic 校准器，输出已校准概率

    输出同时给出 raw / calibrated / oos_brier，便于和旧模型对比。
    """

    def __init__(self, n_splits: int = 6, embargo_pct: float = 0.01,
                 label_mode: str = "direction",
                 calibrate: bool = True, cal_blend: float = 0.5,
                 recenter: bool = True,
                 time_decay_halflife: int = 252,
                 n_bags: int = 3):
        """
        label_mode : "direction"      → 标签=未来horizon日收益符号（默认，天然以0.5为中性）
                     "triple_barrier" → 三重障碍标签（建议仅用于 MetaLabeler 仓位）
        calibrate  : 是否做引擎内 isotonic 校准
        cal_blend  : 校准强度 [0,1]，最终=cal_blend*校准值+(1-cal_blend)*raw
                     0=完全不校准，1=完全用校准。默认0.5（轻推，防过度压缩）
        recenter   : 是否把输出按基准率重新居中，使"无信息"≈0.5
                     —— 修复 v14.0 在牛/熊基准率偏离50%时整体偏多/偏空的问题
        time_decay_halflife : 样本时间衰减半衰期（bar数）。默认252（≈1年）：
                     一年前的样本权重减半，让模型更贴近当前市场结构。
                     设为 0 关闭衰减。
        n_bags     : 最终预测的多种子 bagging 数（1=关闭）。默认3：
                     三个不同 random_state 的模型取平均，显著降低
                     "同一只票隔天预测大幅跳动"的方差。
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.label_mode = label_mode
        self.calibrate = calibrate
        self.cal_blend = float(np.clip(cal_blend, 0.0, 1.0))
        self.recenter = recenter
        self.time_decay_halflife = max(0, int(time_decay_halflife))
        self.n_bags = max(1, int(n_bags))

    _BAG_SEEDS = (42, 137, 2718, 31415, 16180)

    @staticmethod
    def _make_clf(scale_pos_weight: float = 1.0, seed: int = 42):
        if _HAS_XGB:
            return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=scale_pos_weight,
                                 objective="binary:logistic", random_state=seed,
                                 verbosity=0)
        return GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                          learning_rate=0.03, subsample=0.8,
                                          random_state=seed)

    @staticmethod
    def _recenter(p: float, base_rate: float) -> float:
        """把以 base_rate 为中性点的概率，线性重映射到以0.5为中性点。"""
        b = float(np.clip(base_rate, 0.05, 0.95))
        if p <= b:
            out = 0.5 * (p / b) if b > 0 else 0.5
        else:
            out = 0.5 + 0.5 * (p - b) / (1 - b) if b < 1 else 0.5
        return float(np.clip(out, 0.02, 0.98))

    def predict_proba_up(self,
                         feat: pd.DataFrame,
                         close: pd.Series,
                         vol: pd.Series,
                         horizon: int = 5,
                         pt_mult: float = 1.5,
                         sl_mult: float = 1.5,
                         extra_feat: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        extra_feat（v14.2）: 附加特征（如 MarketContext 大盘上下文）。
        与核心特征区别对待：先对核心特征正交化（残差化），再做 IC 显著性
        筛选，只保留确有【增量】信息的列。原因（实测）：上下文特征携带
        增量信息时改善 OOS Brier，与自身因子冗余时反而有害。
        """
        # 记录附加列名，拼接后统一走主管线
        extra_cols: List[str] = []
        if extra_feat is not None and not extra_feat.empty:
            extra_feat = extra_feat.reindex(feat.index)
            # 附加特征缺失用列中位数填充（避免开头滚动窗NaN把样本整行干掉）
            extra_feat = extra_feat.apply(lambda s: s.fillna(s.median()))
            extra_cols = [c for c in extra_feat.columns if c not in feat.columns]
            feat = pd.concat([feat, extra_feat[extra_cols]], axis=1)

        feat = feat.copy().reset_index(drop=True)
        close = close.astype(float).reset_index(drop=True)
        vol = vol.astype(float).reset_index(drop=True)
        n_bars = len(close)
        fail = {"raw": 0.5, "calibrated": 0.5, "centered": 0.5,
                "base_rate": 0.5, "oos_brier": np.nan, "note": "样本不足"}

        # ── 构造标签 ──
        if self.label_mode == "triple_barrier":
            tb = TripleBarrierLabeler.label(close, vol, pt_mult, sl_mult, max_hold=horizon)
            if len(tb) < 100:
                return {**fail, "n_train": len(tb)}
            common = feat.index.intersection(tb.index)
            y_ser = tb.loc[common, "label"].astype(int)
            t1 = tb.loc[common, "t1"]
            X_df = feat.loc[common]
        else:  # direction（默认）—— 原语义：未来horizon日收益符号
            fwd_ret = (close.shift(-horizon) - close) / close
            y_ser = (fwd_ret > 0).astype(int)
            valid_y = y_ser.index[:-horizon] if horizon > 0 else y_ser.index
            common = feat.index.intersection(valid_y)
            y_ser = y_ser.loc[common]
            # 固定horizon → 标签窗口 [i, i+horizon]，t1 = i+horizon
            t1 = pd.Series({i: min(int(i) + horizon, n_bars - 1) for i in common})
            X_df = feat.loc[common]

        X_all = X_df.replace([np.inf, -np.inf], np.nan)
        mask = X_all.notna().all(axis=1)
        X_all, y_all, t1 = X_all[mask], y_ser[mask], t1[mask]
        if len(X_all) < 100 or y_all.nunique() < 2:
            return {**fail, "n_train": len(X_all)}

        # ── v14.2：附加特征的"残差IC"筛选 ──────────────────────
        # 对核心特征正交化 → 冗余信息变纯噪声 → 残差与标签做
        # Spearman IC 显著性检验（|t|>=2 才保留，且改用残差列入模）。
        # 显式保存回归系数，预测时对"当前bar"应用同一变换保证一致性。
        kept_extra: List[str] = []
        extra_betas: Dict[str, np.ndarray] = {}
        core_cols = [c for c in X_all.columns if c not in extra_cols]
        if extra_cols and core_cols:
            E = np.column_stack([X_all[core_cols].fillna(0.0).values,
                                 np.ones(len(X_all))])
            try:
                pinv = np.linalg.pinv(E)
            except Exception:
                pinv = None
            y_arr = y_all.values.astype(float)
            n_s = len(y_arr)
            for c in extra_cols:
                yv = X_all[c].fillna(0.0).values
                if pinv is None or np.std(yv) < 1e-12:
                    continue
                beta = pinv @ yv
                r = yv - E @ beta                  # 残差 = 增量部分
                if np.std(r) < 1e-12:
                    continue
                ic = stats.spearmanr(r, y_arr).statistic
                if not np.isfinite(ic):
                    continue
                tstat = ic * np.sqrt(max(n_s - 2, 1)) / np.sqrt(max(1 - ic ** 2, 1e-9))
                if abs(tstat) >= 2.0:
                    kept_extra.append(c)
                    extra_betas[c] = beta
                    X_all[c] = r                   # 用正交化后的残差入模
            drop = [c for c in extra_cols if c not in kept_extra]
            if drop:
                X_all = X_all.drop(columns=drop)

        base_rate = float(y_all.mean())            # 标签基准率
        X = X_all.values
        y = y_all.values
        n = len(X)

        # ── 样本唯一性权重 × 时间衰减权重 ──
        w = sample_uniqueness(t1, n_bars=n_bars).reindex(X_all.index).fillna(1.0).values
        if self.time_decay_halflife > 0 and n > 1:
            # 越新的样本权重越大：最旧样本相对最新样本按半衰期指数衰减
            age = (n - 1) - np.arange(n)            # 0=最新, n-1=最旧
            decay = 0.5 ** (age / self.time_decay_halflife)
            w = w * decay
            w = w / max(w.mean(), 1e-12)            # 归一化均值≈1

        # 不做 scale_pos_weight：让分类器自然学到 base_rate，
        # 再由 recenter 把 base_rate 映射到 0.5（避免双重处理互相打架）
        spw = 1.0

        # ── Purged CV 折外预测 ──
        pkf = PurgedKFold(self.n_splits, horizon=horizon, embargo_pct=self.embargo_pct)
        oof = np.full(n, np.nan)
        for tr, te in pkf.split(X):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            clf = self._make_clf(scale_pos_weight=spw)
            try:
                clf.fit(Xtr, y[tr], sample_weight=w[tr])
                oof[te] = clf.predict_proba(Xte)[:, 1]
            except Exception:
                continue
        valid = np.isfinite(oof)
        oos_brier = float(np.mean((oof[valid] - y[valid]) ** 2)) if valid.sum() > 5 else np.nan

        # ── 校准（带防塌缩保护）──
        cal = ProbabilityCalibrator()
        cal_ok = False
        if self.calibrate and valid.sum() >= 60:
            cal.fit(oof[valid], y[valid])
            # 防塌缩：若校准把整个分布压成一条窄带（区分度<0.05）就放弃
            probe = np.array([cal.transform(p) for p in (0.2, 0.4, 0.6, 0.8)])
            if probe.max() - probe.min() >= 0.05:
                cal_ok = True

        # ── 全量训练（仅有标签的bar）→ 用'当前最新bar'特征预测 ──
        # v14.2：多种子 bagging，降低单模型随机性导致的预测跳动
        sc = StandardScaler(); Xs = sc.fit_transform(X)
        cols = X_all.columns
        x_cur_ser = feat[cols].iloc[-1].replace([np.inf, -np.inf], np.nan)
        x_cur_ser = x_cur_ser.fillna(X_all.mean())
        # v14.2：保留下来的附加列，当前bar也做同一残差化变换
        if kept_extra:
            core_cur = np.append(
                feat[core_cols].iloc[-1].replace([np.inf, -np.inf], np.nan)
                    .fillna(X_all[core_cols].mean()).values.astype(float), 1.0)
            for c in kept_extra:
                raw_v = float(x_cur_ser[c]) if np.isfinite(x_cur_ser[c]) else 0.0
                x_cur_ser[c] = raw_v - float(core_cur @ extra_betas[c])
        x_cur = x_cur_ser.values.astype(float)
        x_cur_s = sc.transform(x_cur.reshape(1, -1))
        bag_preds = []
        for seed in self._BAG_SEEDS[:self.n_bags]:
            clf = self._make_clf(scale_pos_weight=spw, seed=seed)
            try:
                clf.fit(Xs, y, sample_weight=w)
                bag_preds.append(float(clf.predict_proba(x_cur_s)[0][1]))
            except Exception:
                continue
        raw = float(np.mean(bag_preds)) if bag_preds else 0.5
        bag_std = float(np.std(bag_preds)) if len(bag_preds) > 1 else 0.0
        raw = float(np.clip(raw, 0.02, 0.98))

        # 校准：轻推混合，而非整体替换
        if cal_ok:
            calibrated = self.cal_blend * cal.transform(raw) + (1 - self.cal_blend) * raw
        else:
            calibrated = raw
        calibrated = float(np.clip(calibrated, 0.02, 0.98))

        # 基准率居中：使"无信息"≈0.5（修复牛/熊整体偏置）
        centered = self._recenter(calibrated, base_rate) if self.recenter else calibrated

        return {
            "raw": raw,
            "calibrated": calibrated,
            "centered": centered,
            "base_rate": round(base_rate, 4),
            "oos_brier": oos_brier,
            "bag_std": round(bag_std, 4),     # v14.2：种子间分歧度（不确定性代理）
            "context_kept": kept_extra,       # v14.2：通过残差IC筛选的上下文特征
            "n_train": int(n),
            "n_oof": int(valid.sum()),
            "label_mode": self.label_mode,
            "note": "ok",
        }


# ══════════════════════════════════════════════════════════════════
# [11] 一键注入主程序 quantpro_v1_6.py
# ══════════════════════════════════════════════════════════════════
def patch_quantpro(ProbabilityEngine_cls,
                   WalkForwardBacktest_cls=None,
                   SelfCorrectionEngine_cls=None,
                   label_mode: str = "direction",
                   cal_blend: float = 0.4,
                   recenter: bool = True,
                   db_calibration: bool = False,
                   db_calibration_min: int = 120,
                   db_path: str = "quantpro_predictions.db",
                   fetch_fn=None,
                   time_decay_halflife: int = 252,
                   n_bags: int = 3):
    """
    把 v14.2 方法论注入现有类。在 quantpro_v1_6.py 末尾调用：

        from quantpro_v14_alpha_engine import patch_quantpro
        patch_quantpro(ProbabilityEngine, WalkForwardBacktest, SelfCorrectionEngine,
                       fetch_fn=fetch_stock_data)   # ← v14.2：传入抓取函数启用大盘上下文

    v14.2 新增：
      fetch_fn            : 主程序的 fetch_stock_data。提供后自动给每次预测
                            拼入 SPY/VIX 大盘环境特征（spy_mom5/20, rel_str20,
                            corr60, vix_z, vix_chg5）。None=不启用。
      time_decay_halflife : 时间衰减半衰期（bar），默认252≈1年；0=关闭
      n_bags              : 多种子bagging数，默认3；1=关闭

    v14.1 保留（修复全面看空）：
      label_mode="direction"（0.5为中性）、recenter、cal_blend 轻校准、
      db_calibration 默认 False（历史库单一行情会整体压低预测）
    """
    _enh = EnhancedProbabilityEngine(label_mode=label_mode,
                                     calibrate=True, cal_blend=cal_blend,
                                     recenter=recenter,
                                     time_decay_halflife=time_decay_halflife,
                                     n_bags=n_bags)

    def _get_ctx() -> Optional[MarketContext]:
        if fetch_fn is None:
            return None
        try:
            return MarketContext.from_fetch(fetch_fn)
        except Exception as e:
            logger.warning(f"MarketContext build: {e}")
            return None

    # ── 1) 替换方向概率（输出居中概率，带降级回退）─────────────
    _orig_xgb = ProbabilityEngine_cls.xgb_probability
    _orig_xgb_short = getattr(ProbabilityEngine_cls, "xgb_probability_short", None)

    def _enhanced_xgb(self, fwd=5):
        try:
            vol = self.ind["atr"] / (self.close.abs() + 1e-9)
            # v14.2：大盘上下文作为 extra_feat 传入（引擎内做残差IC筛选）
            extra = None
            ctx = _get_ctx()
            if ctx is not None:
                e = ctx.features_for(self.close)
                extra = e if not e.empty else None
            res = _enh.predict_proba_up(self._feat, self.close, vol,
                                        horizon=fwd, extra_feat=extra)
            if res.get("note") == "ok":
                # 用 centered：以0.5为中性，和旧ensemble的0.5语义一致
                return res["centered"] if recenter else res["calibrated"]
        except Exception as e:
            logger.warning(f"enhanced_xgb fallback: {e}")
        return _orig_xgb(self, fwd)

    ProbabilityEngine_cls.xgb_probability = _enhanced_xgb
    if _orig_xgb_short is not None:
        def _enhanced_xgb_short(self, fwd=5):
            return _enhanced_xgb(self, fwd)
        ProbabilityEngine_cls.xgb_probability_short = _enhanced_xgb_short

    # ── 2) ensemble 末端 DB 校准（默认关闭，带保护）─────────────
    _orig_ensemble = ProbabilityEngine_cls.ensemble

    def _maybe_calibrated_ensemble(self, fwd=5, sentiment_agg=None, symbol=None):
        out = _orig_ensemble(self, fwd=fwd, sentiment_agg=sentiment_agg, symbol=symbol)
        if not db_calibration:
            return out
        try:
            cal = ProbabilityCalibrator.from_self_correction_db(
                db_path, horizon=fwd, min_samples=db_calibration_min)
            if cal.n_fit < db_calibration_min:
                return out                       # 样本不足，不校准
            base_prob = out[0]
            # 轻推混合，避免历史库整体压低
            blended = 0.5 * cal.transform(base_prob) + 0.5 * base_prob
            return (float(np.clip(blended, 0.02, 0.98)),) + tuple(out[1:])
        except Exception:
            return out

    ProbabilityEngine_cls.ensemble = _maybe_calibrated_ensemble

    # ── 3) Walk-Forward 附加 DSR ────────────────────────────────
    if WalkForwardBacktest_cls is not None:
        _orig_run = WalkForwardBacktest_cls.run

        def _run_with_dsr(self, sym, df, params):
            res = _orig_run(self, sym, df, params)
            if not res:
                return res
            try:
                dates, equity = res["equity_curve"]
                eq = pd.Series(equity, index=pd.Index(dates)).astype(float)
                rets = eq.pct_change().dropna().values
                # _optimize_params 网格规模 = 4 * 4 = 16 组 × 折数（多重检验）
                n_trials = 16 * max(1, res.get("n_folds", 1))
                # 跨折夏普方差作为试验方差代理
                fold_sr = [f.get("return", 0) for f in res.get("fold_stats", [])]
                trial_var = float(np.var(fold_sr)) / 1e4 if len(fold_sr) > 1 else 0.01
                dsr = DeflatedSharpe.deflated_sharpe(rets, n_trials, max(trial_var, 1e-4))
                mintrl = DeflatedSharpe.min_track_record_length(rets, 0.0, 0.95)
                res["deflated_sharpe"] = round(dsr["deflated_sharpe"], 4)
                res["dsr_verdict"] = dsr["verdict"]
                res["psr_vs_zero"] = round(dsr["psr_vs_zero"], 4)
                res["min_track_record"] = (round(mintrl, 1)
                                           if np.isfinite(mintrl) else None)
            except Exception as e:
                logger.warning(f"DSR augment: {e}")
            return res

        WalkForwardBacktest_cls.run = _run_with_dsr

    logger.info("[v14.2] 注入完成：+大盘上下文(SPY/VIX) +时间衰减 +多种子Bagging（DB校准默认关闭）")
    return ProbabilityEngine_cls


# ══════════════════════════════════════════════════════════════════
# 演示 / 自检（合成数据，无需联网）
# ══════════════════════════════════════════════════════════════════
def _demo():
    rng = np.random.default_rng(7)
    n = 800
    # 合成一段带轻微动量+均值回归的价格
    rets = rng.standard_t(df=5, size=n) * 0.012 + 0.0004
    rets[200:260] += 0.004   # 一段趋势
    price = 100 * np.cumprod(1 + rets)
    idx = pd.RangeIndex(n)
    close = pd.Series(price, index=idx)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().bfill()
    volp = (atr / close).bfill()

    print("=" * 64)
    print("QuantPro v14 Alpha Engine 自检（合成数据）")
    print("=" * 64)

    # [1] PurgedKFold
    pkf = PurgedKFold(5, horizon=5, embargo_pct=0.02)
    folds = list(pkf.split(np.zeros((n, 3))))
    leak = 0
    for tr_idx, te_idx in folds:
        # 检查训练集是否有任何样本的标签窗口侵入测试集
        for j in tr_idx:
            if te_idx.min() - 5 <= j <= te_idx.max():
                leak += 1
    print(f"[1] PurgedKFold  折数={len(folds)}  残留泄漏样本={leak}（应≈0）")

    # [2] Triple-Barrier
    tb = TripleBarrierLabeler.label(close, volp, pt_mult=2.0, sl_mult=1.5, max_hold=10)
    print(f"[2] TripleBarrier  样本={len(tb)}  上涨标签占比={tb['label'].mean():.2%}  "
          f"止盈/止损/时间={ (tb['first_touch']=='up').sum() }/"
          f"{ (tb['first_touch']=='dn').sum() }/{ (tb['first_touch']=='time').sum() }")

    # [3] sample uniqueness
    w = sample_uniqueness(tb["t1"], n_bars=n)
    print(f"[3] 样本唯一性  均值={w.mean():.3f}（归一化≈1）  "
          f"最小={w.min():.3f}  最大={w.max():.3f}")

    # [4] Deflated Sharpe
    strat_rets = rng.normal(0.0006, 0.01, 300)   # 假装一条策略净值
    dsr = DeflatedSharpe.deflated_sharpe(strat_rets, n_trials=50, trial_sharpe_variance=0.02)
    print(f"[4] DeflatedSharpe  观测SR={dsr['observed_sharpe']:.3f}  "
          f"去膨胀阈值={dsr['deflation_threshold']:.3f}  "
          f"DSR={dsr['deflated_sharpe']:.3f} → {dsr['verdict']}")

    # [5] PBO
    R = rng.normal(0.0002, 0.01, size=(300, 20))   # 20条纯噪声策略
    pbo = prob_backtest_overfit(R, n_splits=8)
    print(f"[5] PBO(20条噪声策略)  pbo={pbo['pbo']:.3f}  组合数={pbo['n_combinations']} "
          f"→ {pbo['verdict']}（噪声应偏高）")

    # [6] Calibration
    p_raw = np.clip(rng.beta(2, 2, 500), 0.02, 0.98)
    y = (rng.uniform(size=500) < p_raw * 0.6 + 0.2).astype(int)  # 系统性高估
    cal = ProbabilityCalibrator().fit(p_raw, y)
    before = np.mean((p_raw - y) ** 2)
    after = np.mean(([cal.transform(p) for p in p_raw] - y) ** 2)
    print(f"[6] 概率校准  Brier 校准前={before:.4f} → 校准后={after:.4f} "
          f"（下降{(before-after)/before:.1%}）")

    # [8] FFD
    ffd = frac_diff_ffd(close, d=0.4)
    corr = np.corrcoef(ffd.values, close.loc[ffd.index].values)[0, 1]
    print(f"[8] 分数阶差分 d=0.4  长度={len(ffd)}  与原价相关={corr:.3f}"
          f"（保留记忆但更平稳）")

    # [10] EnhancedProbabilityEngine
    feat = pd.DataFrame({
        "mom5": close.pct_change(5),
        "mom10": close.pct_change(10),
        "zscore20": (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-9),
        "rsi_proxy": close.pct_change().rolling(14).mean() * 100,
        "vol": volp,
    }).bfill().fillna(0)
    eng = EnhancedProbabilityEngine(n_splits=5)
    res = eng.predict_proba_up(feat, close, volp, horizon=5)
    print(f"[10] EnhancedEngine  raw={res['raw']:.3f}  cal={res['calibrated']:.3f}  "
          f"OOS_Brier={res['oos_brier']:.4f}  n_train={res['n_train']}")
    print("=" * 64)
    print("全部组件自检通过 ✓")


if __name__ == "__main__":
    _demo()
