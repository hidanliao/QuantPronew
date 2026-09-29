"""
QuantPro — GARCH 拟合稳健化补丁 v1.1
══════════════════════════════════════════════════════════════════
修复：arch 库返回 status=-1（优化失败），概率弹窗每个周期打一条 warning
      然后 fallback。根因：GARCH(1,1)-t 在指数/ETF/低波动/含离群值
      的日收益上数值不收敛。

改进：
  ① 数据清洗：clip 0.1%/99.9% 极端值 + 去 inf/nan
  ② 逐档降级：GARCH-t → GARCH-normal → EWMA（永不失败）
  ③ 参数有效性检查：omega>0、alpha>=0、beta>=0、alpha+beta<1、nu>=2.1
  ④ 结果缓存：同一次弹窗里 5/20/60 日共享同一次拟合
  ⑤ 缓存 key 用序列内容指纹（不用 id，避免对象复用）

集成（quantpro_v1_6.py 的 __main__ 里，创建 QuantApp 之前）：

    from quantpro_garch_fix import install_garch_fix
    install_garch_fix(globals())
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── 缓存 ──────────────────────────────────────────────────────
_GARCH_CACHE: Dict[str, dict] = {}
_GARCH_CACHE_LOCK = threading.Lock()
_GARCH_CACHE_MAX = 64


# ══════════════════════════════════════════════════════════════════
# 稳健 GARCH 拟合
# ══════════════════════════════════════════════════════════════════
def _robust_garch_fit(returns: pd.Series) -> dict:
    """
    返回 dict：
      success     : bool
      method      : "garch_t" / "garch_normal" / "ewma" / "failed"
      omega/alpha/beta/nu : 参数（omega 单位为"百分比²"，与 arch 输出一致）
      last_sigma2 : 最近一期条件方差（小数²）
      n_obs       : 训练样本数
    """
    # ── 数据清洗 ──
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 100:
        return {"success": False, "method": "short_data", "n_obs": len(r)}
    lo, hi = r.quantile(0.001), r.quantile(0.999)
    r = r.clip(lo, hi)
    r_pct = r * 100.0    # arch 官方推荐在百分比量级上拟合

    # ── 检查 arch 可用性 ──
    try:
        from arch import arch_model
    except Exception as e:
        logger.warning(f"[garch_fix] arch 库不可用: {e}")
        return {"success": False, "method": "no_arch", "n_obs": len(r)}

    # ── 逐档尝试 ──
    configs = [
        {"dist": "t",      "rescale": True, "options": {"maxiter": 300, "ftol": 1e-8}},
        {"dist": "normal", "rescale": True, "options": {"maxiter": 300, "ftol": 1e-8}},
        {"dist": "t",      "rescale": True, "options": {"maxiter": 500}},
        {"dist": "normal", "rescale": True, "options": {"maxiter": 500}},
    ]

    for cfg in configs:
        try:
            am = arch_model(r_pct, vol="Garch", p=1, q=1,
                            dist=cfg["dist"], rescale=cfg["rescale"])
            res = am.fit(disp="off", show_warning=False, options=cfg["options"])
        except Exception as e:
            logger.debug(f"[garch_fix] cfg={cfg} 拟合异常: {e}")
            continue

        # ── 参数有效性检查 ──
        try:
            p = res.params
            omega = float(p.get("omega", np.nan))
            alpha = float(p.get("alpha[1]", np.nan))
            beta  = float(p.get("beta[1]", np.nan))
            nu    = float(p.get("nu", 5.0)) if cfg["dist"] == "t" else 5.0
        except Exception:
            continue

        if not (np.isfinite(omega) and np.isfinite(alpha) and np.isfinite(beta)):
            continue
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1.0:
            continue
        if cfg["dist"] == "t" and (not np.isfinite(nu) or nu < 2.1):
            continue

        # ── 提取条件波动率 ──
        try:
            cv = res.conditional_volatility
            if cv is None or len(cv) == 0:
                continue
            last_sigma2 = float(cv[-1]) ** 2 / 10000.0   # 百分比² → 小数²
            if not np.isfinite(last_sigma2) or last_sigma2 <= 0:
                continue
        except Exception:
            continue

        return {
            "success": True,
            "method": f"garch_{cfg['dist']}",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "nu": max(nu, 2.1),
            "last_sigma2": last_sigma2,
            "n_obs": len(r),
        }

    # ── 保底：EWMA 波动率（不用优化器，永不失败）──
    try:
        lam = 0.94
        var = float(r.var())
        if not np.isfinite(var) or var <= 0:
            raise ValueError("方差非正")
        for x in r.values:
            var = lam * var + (1 - lam) * x * x
        if np.isfinite(var) and var > 0:
            return {
                "success": True,
                "method": "ewma",
                "omega": var * (1 - lam),
                "alpha": 1 - lam,
                "beta": lam,
                "nu": 5.0,
                "last_sigma2": var,
                "n_obs": len(r),
            }
    except Exception as e:
        logger.warning(f"[garch_fix] EWMA 保底也失败: {e}")

    return {"success": False, "method": "failed", "n_obs": len(r)}


def _series_fingerprint(close: pd.Series) -> str:
    """用序列内容构造缓存 key（不用 id，避免对象复用）。"""
    try:
        n = len(close)
        head = float(close.iloc[0])
        tail = float(close.iloc[-1])
        # 尾10个值的 hash（保留4位小数避免浮点抖动）
        tail_hash = hash(tuple(np.round(close.tail(10).values, 4)))
        return f"{n}|{head:.4f}|{tail:.4f}|{tail_hash}"
    except Exception:
        return f"fallback|{len(close)}"


def _cached_fit(close: pd.Series) -> dict:
    key = _series_fingerprint(close)
    with _GARCH_CACHE_LOCK:
        if key in _GARCH_CACHE:
            return _GARCH_CACHE[key]
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    fit = _robust_garch_fit(ret)
    with _GARCH_CACHE_LOCK:
        if len(_GARCH_CACHE) >= _GARCH_CACHE_MAX:
            try:
                _GARCH_CACHE.pop(next(iter(_GARCH_CACHE)))
            except Exception:
                _GARCH_CACHE.clear()
        _GARCH_CACHE[key] = fit
    return fit


# ══════════════════════════════════════════════════════════════════
# 替换 ProbabilityEngine.monte_carlo
# ══════════════════════════════════════════════════════════════════
def install_garch_fix(env: dict):
    required = ["ProbabilityEngine"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_garch_fix: 主程序缺少 {missing}")

    PE = env["ProbabilityEngine"]

    if getattr(PE, "_garch_fix_patched", False):
        logger.info("[garch_fix] 已安装（跳过重复注入）")
        return PE

    _orig_mc = PE.monte_carlo

    def _robust_monte_carlo(self, days: int = 20, n: int = 10000,
                            use_fat_tail: bool = True):
        """
        兼容原接口：返回 dict，含 up_prob/var95/cvar95/low_pct/high_pct/
        median_pct/finals/cur。算法：GARCH(1,1)-t 路径模拟；拟合失败降级
        到 EWMA 或原 fallback。
        """
        returns = self.close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        cur = float(self.close.iloc[-1])

        if not (use_fat_tail and len(returns) >= 100):
            return _orig_mc(self, days=days, n=n, use_fat_tail=use_fat_tail)

        fit = _cached_fit(self.close)
        if not fit.get("success"):
            logger.info(f"[garch_fix] 稳健拟合失败({fit.get('method')})，走原 fallback")
            return _orig_mc(self, days=days, n=n, use_fat_tail=use_fat_tail)

        # ── 路径模拟 ──
        try:
            omega = float(fit["omega"])
            alpha = float(fit["alpha"])
            beta  = float(fit["beta"])
            nu    = float(fit["nu"])
            last_sigma2 = float(fit["last_sigma2"])

            mu_raw = float(returns.mean())
            decay  = np.exp(-0.8 * days / 252)
            mu_adj = mu_raw * decay

            rng = np.random.default_rng(42 + days)
            # omega 从 r_pct（百分比单位）拟合来，转小数² 后使用
            omega_dec = omega / 10000.0

            # 批量采样 t 分布（一次采 n*days 个，比逐日采快得多）
            t_samples = stats.t.rvs(nu, size=(n, days), random_state=rng)

            all_finals = np.empty(n, dtype=float)
            for i in range(n):
                s2 = last_sigma2
                price = cur
                row = t_samples[i]
                for j in range(days):
                    sigma = np.sqrt(s2)
                    eps = row[j] * sigma
                    ret = mu_adj + eps
                    price *= (1.0 + ret)
                    s2 = omega_dec + alpha * (eps * eps) + beta * s2
                    # 防数值爆炸：一旦 s2 变成 inf/nan 就原地刹车
                    if not np.isfinite(s2) or s2 <= 0:
                        s2 = last_sigma2
                all_finals[i] = price

            pcts = (all_finals - cur) / cur * 100.0
            var95 = float(np.percentile(pcts, 5))
            below = pcts[pcts <= var95]
            return {
                "up_prob":    float(np.mean(all_finals > cur)),
                "var95":      var95,
                "cvar95":     float(below.mean()) if len(below) else var95,
                "low_pct":    float(np.percentile(pcts, 5)),
                "high_pct":   float(np.percentile(pcts, 95)),
                "median_pct": float(np.median(pcts)),
                "finals":     all_finals,
                "cur":        cur,
                "_method":    fit["method"],
            }
        except Exception as e:
            logger.warning(f"[garch_fix] 模拟失败: {e}，走原 fallback")
            return _orig_mc(self, days=days, n=n, use_fat_tail=use_fat_tail)

    PE.monte_carlo = _robust_monte_carlo
    PE._garch_fix_patched = True

    def _clear_garch_cache():
        with _GARCH_CACHE_LOCK:
            _GARCH_CACHE.clear()
    env["clear_garch_cache"] = _clear_garch_cache

    logger.info("[garch_fix] 已安装：GARCH 拟合稳健化 + 结果缓存")
    return PE


# ══════════════════════════════════════════════════════════════════
# 自检（合成数据，无需联网）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("GARCH 稳健化补丁自检")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n = 500

    # 场景1：常规噪声
    r1 = pd.Series(rng.normal(0.0003, 0.012, n))
    fit1 = _robust_garch_fit(r1)
    print(f"[场景1] 常规噪声       method={fit1.get('method'):<14} success={fit1.get('success')}")
    assert fit1["success"], "常规序列应能拟合"

    # 场景2：低波动（原代码最易失败）
    r2 = pd.Series(rng.normal(0.0002, 0.003, n))
    fit2 = _robust_garch_fit(r2)
    print(f"[场景2] 低波动指数型   method={fit2.get('method'):<14} success={fit2.get('success')}")
    assert fit2["success"], "低波动序列应能降级成功"

    # 场景3：含极端离群值
    r3 = rng.normal(0.0003, 0.012, n)
    r3[100] = 0.5
    r3[300] = -0.4
    fit3 = _robust_garch_fit(pd.Series(r3))
    print(f"[场景3] 极端离群       method={fit3.get('method'):<14} success={fit3.get('success')}")
    assert fit3["success"], "含离群值序列应能降级成功"

    # 场景4：近似常量（EWMA 保底）
    r4 = pd.Series(np.full(n, 1e-6))
    fit4 = _robust_garch_fit(r4)
    print(f"[场景4] 近似常量       method={fit4.get('method'):<14} success={fit4.get('success')}")

    # 平稳性约束
    for i, f in enumerate([fit1, fit2, fit3], start=1):
        if f.get("success") and f.get("method", "").startswith("garch"):
            s = f["alpha"] + f["beta"]
            print(f"[场景{i}] alpha+beta={s:.4f}（应<1）  nu={f.get('nu', 5.0):.2f}")
            assert s < 1.0, f"场景{i} GARCH 参数不满足平稳性约束"

    # 缓存命中
    close = pd.Series(100 * np.cumprod(1 + r1.values))
    f_a = _cached_fit(close)
    f_b = _cached_fit(close)
    assert f_a is f_b, "缓存未命中（应返回同一 dict 对象）"
    print("\n[缓存] 二次调用命中同一对象 ✓")

    print("\n全部自检通过 ✓")