"""
QuantPro — 量价分析模块（Volume-Price Analysis）v1.1
══════════════════════════════════════════════════════════════════
主程序 __main__ 第⑦步会 import 这个文件，本模块补齐其功能。

功能：
  概率弹窗新增「量价诊断」tab：
    - OBV 能量潮（含 MA20）
    - 60日滚动量价相关（健康趋势 vs 背离）
    - VWAP 偏离%（低买/高卖参考）
    - Amihud 非流动性 + Garman-Klass 年化波动（双轴）
    - 量价背离扫描（近90日顶背离/底背离）
    - 主力资金流向指标（up_vol - down_vol）/ (up_vol + down_vol)
    - 数值摘要卡 + 文字研判

集成（quantpro_v1_6.py 的 __main__ 里，创建 QuantApp 之前）：

    from quantpro_volume_analysis import install_volume_analysis
    install_volume_analysis(globals())
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Callable

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout,
)
from PyQt5.QtCore import Qt

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 模块级依赖缓存（install 时注入，避免循环 import）
# ══════════════════════════════════════════════════════════════════
_DEPS: Dict[str, object] = {}


# ══════════════════════════════════════════════════════════════════
# 计算层（纯函数）
# ══════════════════════════════════════════════════════════════════
def _safe_col(df: pd.DataFrame, name: str) -> pd.Series:
    s = df[name]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.squeeze().astype(float)


def compute_volume_features(df: pd.DataFrame, window: int = 20) -> Dict:
    """
    从 OHLCV 计算量价诊断特征。
    """
    if df is None or df.empty or len(df) < 20:
        return {"error": "数据不足"}

    try:
        close = _safe_col(df, "Close")
        high  = _safe_col(df, "High")
        low   = _safe_col(df, "Low")
        open_ = _safe_col(df, "Open")
        vol   = _safe_col(df, "Volume")
    except KeyError as e:
        return {"error": f"缺少列: {e}"}

    # ── OBV ──
    sign = np.sign(close.diff()).fillna(0)
    obv = (sign * vol).cumsum()
    obv_ma = obv.rolling(window).mean()

    # ── 上下行成交量 ──
    up_vol = vol.where(close > close.shift(1), 0).rolling(window).sum()
    dn_vol = vol.where(close < close.shift(1), 0).rolling(window).sum()
    vol_imbalance = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-9)

    # ── 60日滚动量价相关 ──
    ret = close.pct_change()
    vol_chg = vol.pct_change()
    roll_corr = ret.rolling(60).corr(vol_chg)

    # ── VWAP 偏离 ──
    vwap_20 = (close * vol).rolling(window).sum() / vol.rolling(window).sum().replace(0, np.nan)
    vwap_dev = (close - vwap_20) / vwap_20.replace(0, np.nan)

    # ── Amihud 非流动性 ──
    dollar_vol = (close * vol).replace(0, np.nan)
    amihud = (ret.abs() / dollar_vol).rolling(window).mean() * 1e6

    # ── Roll 有效价差 ──
    ret_lag = ret.shift(1)
    roll_cov = ret.rolling(window).cov(ret_lag)
    roll_spread = 2 * np.sqrt(np.maximum(-roll_cov, 0))

    # ── Garman-Klass 波动率 ──
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gk_vol = np.sqrt(gk.rolling(window).mean().clip(lower=0) * 252)

    # ── 量价背离扫描（近90日）──
    price_high = close.rolling(window).max()
    price_low  = close.rolling(window).min()
    vol_ma     = vol.rolling(window).mean()
    near_high  = close >= price_high * 0.97
    near_low   = close <= price_low * 1.03
    low_vol    = vol < vol_ma * 0.8
    bearish    = near_high & low_vol
    bullish    = near_low  & low_vol

    div_list = []
    for d in close.index[-90:]:
        if bearish.get(d, False):
            div_list.append((d, "bearish"))
        elif bullish.get(d, False):
            div_list.append((d, "bullish"))

    # ── 汇总最新值 ──
    last = -1
    def _last(sr):
        try:
            v = float(sr.iloc[last])
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    summary = {
        "obv":               _last(obv),
        "obv_vs_ma":         "↑强势" if obv.iloc[last] > obv_ma.iloc[last] else "↓弱势",
        "vol_imbalance_20d": _last(vol_imbalance),
        "roll_corr_60d":     _last(roll_corr),
        "vwap_dev_pct":      _last(vwap_dev) * 100 if np.isfinite(_last(vwap_dev)) else np.nan,
        "amihud_illiq":      _last(amihud),
        "roll_spread":       _last(roll_spread),
        "gk_vol_annual":     _last(gk_vol),
        "n_divergence_90d":  len(div_list),
    }

    # 量价关系诊断
    rc = summary["roll_corr_60d"]
    if np.isfinite(rc):
        if rc > 0.3:
            summary["regime_hint"] = "量价同向（健康趋势）"
        elif rc < -0.3:
            summary["regime_hint"] = "量价背离（警惕反转）"
        else:
            summary["regime_hint"] = "量价弱相关（震荡）"
    else:
        summary["regime_hint"] = "--"

    return {
        "index":         close.index,
        "close":         close,
        "volume":        vol,
        "obv":           obv,
        "obv_ma":        obv_ma,
        "roll_corr":     roll_corr,
        "vwap_dev":      vwap_dev,
        "amihud":        amihud,
        "roll_spread":   roll_spread,
        "gk_vol":        gk_vol,
        "vol_imbalance": vol_imbalance,
        "divergence_flags": div_list,
        "summary":       summary,
    }


# ══════════════════════════════════════════════════════════════════
# UI：量价诊断 Tab
# ══════════════════════════════════════════════════════════════════
def _install_volume_tab(ProbabilityDialog):
    """给概率弹窗追加一个「量价诊断」tab。"""

    # ── 覆盖 _build_ui：原版执行完后，追加量价 tab ──
    _orig_build_ui = ProbabilityDialog._build_ui

    def _build_ui_with_volume(self):
        _orig_build_ui(self)
        try:
            _build_tab_volume(self)
        except Exception as e:
            logger.warning(f"[volume_analysis] 添加量价诊断tab失败: {e}")

    ProbabilityDialog._build_ui = _build_ui_with_volume

    # ── 覆盖 _run：原版执行完后，刷新量价数据 ──
    _orig_run = ProbabilityDialog._run

    def _run_with_volume(self):
        _orig_run(self)
        try:
            _refresh_volume_tab(self)
        except Exception as e:
            logger.warning(f"[volume_analysis] 刷新量价诊断失败: {e}")

    ProbabilityDialog._run = _run_with_volume


def _build_tab_volume(self):
    T = _DEPS["T"]

    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(8)

    # ── 顶部：数值摘要卡 ──
    summary_card = QFrame()
    summary_card.setStyleSheet(
        f"QFrame{{background:{T.BG2};border-radius:10px;"
        f"border:1px solid {T.BORDER};}}")
    grid = QGridLayout(summary_card)
    grid.setContentsMargins(14, 12, 14, 12)
    grid.setSpacing(6)

    self._vol_summary_labels = {}

    keys = [
        ("OBV 能量",      "obv_vs_ma"),
        ("20日资金流",    "vol_imbalance_20d"),
        ("60日量价相关",  "roll_corr_60d"),
        ("VWAP 偏离%",    "vwap_dev_pct"),
        ("Amihud 非流动", "amihud_illiq"),
        ("GK 年化波动",   "gk_vol_annual"),
        ("90日背离次数",  "n_divergence_90d"),
        ("量价研判",      "regime_hint"),
    ]
    for i, (label, key) in enumerate(keys):
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
        val = QLabel("--")
        val.setStyleSheet(f"color:{T.TEXT_H};font-size:11pt;font-weight:700;border:none;")
        row = i // 4
        col = (i % 4) * 2
        grid.addWidget(lbl, row, col)
        grid.addWidget(val, row, col + 1)
        self._vol_summary_labels[key] = val

    lay.addWidget(summary_card)

    # ── 主图 ──
    self._vol_canvas = FigureCanvas(
        Figure(figsize=(9, 5.5), facecolor=T.MPL_BG))
    lay.addWidget(self._vol_canvas, 1)

    # ── 结论小字 ──
    self._vol_note = QLabel("加载中...")
    self._vol_note.setWordWrap(True)
    self._vol_note.setStyleSheet(
        f"color:{T.TEXT_2};font-size:8.5pt;background:{T.BG2};"
        f"border:1px solid {T.BORDER};border-radius:6px;padding:8px;")
    lay.addWidget(self._vol_note)

    # 加进 tab（优先用主程序的滚动区包裹，保持风格一致）
    if hasattr(self, "_add_scroll_tab"):
        self._add_scroll_tab(w, "量价诊断")
    else:
        self.tabs.addTab(w, "量价诊断")


def _refresh_volume_tab(self):
    T = _DEPS["T"]
    fetch = _DEPS["fetch_stock_data"]

    df = fetch(self.sym, "6mo")
    if df is None or df.empty:
        self._vol_note.setText("无数据")
        return

    feats = compute_volume_features(df, window=20)
    if "error" in feats:
        self._vol_note.setText(feats["error"])
        return

    # ── 填数值摘要 ──
    s = feats["summary"]
    for key, lbl in self._vol_summary_labels.items():
        v = s.get(key, "--")
        if isinstance(v, float):
            if key == "vol_imbalance_20d":
                text = f"{v:+.3f}"
                col = T.GREEN if v > 0.1 else (T.RED if v < -0.1 else T.TEXT_H)
            elif key == "roll_corr_60d":
                if np.isnan(v):
                    text = "--"; col = T.TEXT_H
                else:
                    text = f"{v:+.3f}"
                    col = T.GREEN if v > 0.3 else (T.RED if v < -0.3 else T.YELLOW)
            elif key == "vwap_dev_pct":
                if np.isnan(v):
                    text = "--"; col = T.TEXT_H
                else:
                    text = f"{v:+.2f}%"
                    col = T.GREEN if v < 0 else T.RED
            elif key == "amihud_illiq":
                text = f"{v:.3f}" if np.isfinite(v) else "--"; col = T.TEXT_H
            elif key == "gk_vol_annual":
                text = f"{v*100:.1f}%" if np.isfinite(v) else "--"; col = T.TEXT_H
            else:
                text = f"{v:g}"; col = T.TEXT_H
        else:
            text = str(v); col = T.TEXT_H
        lbl.setText(text)
        lbl.setStyleSheet(
            f"color:{col};font-size:11pt;font-weight:700;border:none;")

    # ── 绘图 ──
    _draw_volume_charts(self, feats)

    # ── 结论 ──
    hint = s.get("regime_hint", "--")
    obv_t = s.get("obv_vs_ma", "--")
    vi = s.get("vol_imbalance_20d", 0.0)
    nd = s.get("n_divergence_90d", 0)
    concl = (f"【量价关系】{hint}　|　OBV：{obv_t}　|　"
             f"20日资金净流入指标：{vi:+.3f}　|　近90日背离信号：{nd}次")
    if nd >= 3:
        concl += "　⚠ 背离频繁，注意趋势可持续性"
    self._vol_note.setText(concl)


def _draw_volume_charts(self, feats):
    T = _DEPS["T"]
    get_rc = _DEPS["_get_mpl_rc"]
    apply_font = _DEPS["_apply_font_to_figure"]
    mpl_style = _DEPS["_mpl_style"]

    with mpl.rc_context(get_rc()):
        fig = self._vol_canvas.figure
        fig.clear()
        fig.patch.set_facecolor(T.MPL_BG)
        idx = feats["index"]

        gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.25)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # ① OBV
        ax1.plot(idx, feats["obv"].values, color=T.ACCENT, lw=1.2, label="OBV")
        ax1.plot(idx, feats["obv_ma"].values, color=T.GOLD, lw=1.0,
                 ls="--", label="MA20")
        mpl_style(ax1, title="OBV 能量潮", ylabel="OBV")
        ax1.legend(fontsize=7, facecolor=T.BG2, labelcolor=T.TEXT_H,
                   edgecolor=T.BORDER)

        # ② 60日滚动量价相关
        rc = feats["roll_corr"]
        ax2.axhline(0, color=T.TEXT_3, lw=0.8)
        ax2.axhline(0.3, color=T.GREEN, lw=0.6, ls=":")
        ax2.axhline(-0.3, color=T.RED, lw=0.6, ls=":")
        ax2.plot(idx, rc.values, color=T.CYAN, lw=1.2)
        ax2.fill_between(idx, 0, rc.values,
                         where=(rc.values > 0), color=T.GREEN, alpha=0.12)
        ax2.fill_between(idx, 0, rc.values,
                         where=(rc.values < 0), color=T.RED, alpha=0.12)
        ax2.set_ylim(-1.05, 1.05)
        mpl_style(ax2, title="60日滚动量价相关", ylabel="Corr")

        # ③ VWAP 偏离
        vd = feats["vwap_dev"] * 100
        ax3.axhline(0, color=T.TEXT_3, lw=0.8)
        ax3.plot(idx, vd.values, color=T.PURPLE, lw=1.2)
        ax3.fill_between(idx, 0, vd.values,
                         where=(vd.values > 0), color=T.RED, alpha=0.10)
        ax3.fill_between(idx, 0, vd.values,
                         where=(vd.values < 0), color=T.GREEN, alpha=0.10)
        mpl_style(ax3, title="VWAP 偏离%", ylabel="%")

        # ④ 流动性与波动率（双轴）
        am = feats["amihud"]
        ax4.plot(idx, am.values, color=T.ORANGE, lw=1.0, label="Amihud")
        ax4.set_ylabel("Amihud ×1e6", color=T.TEXT_2, fontsize=8)
        ax4b = ax4.twinx()
        gk = feats["gk_vol"] * 100
        ax4b.plot(idx, gk.values, color=T.CYAN, lw=1.0, label="GK Vol%")
        ax4b.set_ylabel("GK年化波动%", color=T.TEXT_2, fontsize=8)
        ax4b.tick_params(colors=T.TEXT_2, labelsize=7)
        mpl_style(ax4, title="流动性 & 波动率", xlabel="")
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                   facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)

        fig.autofmt_xdate()
        apply_font(fig)

    self._vol_canvas.draw()


# ══════════════════════════════════════════════════════════════════
# 一键安装
# ══════════════════════════════════════════════════════════════════
def install_volume_analysis(env: dict):
    required = ["ProbabilityDialog", "fetch_stock_data", "T",
                "_get_mpl_rc", "_apply_font_to_figure", "_mpl_style"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_volume_analysis: 主程序缺少 {missing}")

    # 把依赖缓存下来，供 UI 回调使用（避免任何 import 主程序）
    for k in required:
        _DEPS[k] = env[k]

    PD = env["ProbabilityDialog"]
    if getattr(PD, "_volume_analysis_installed", False):
        logger.info("[volume_analysis] 已安装（跳过重复注入）")
        return PD

    _install_volume_tab(PD)
    PD._volume_analysis_installed = True
    logger.info("[volume_analysis] 已安装：概率弹窗新增「量价诊断」tab")
    return PD


# ══════════════════════════════════════════════════════════════════
# 自检（合成数据，无需联网/无需主程序）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("量价分析模块自检")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n = 200
    idx = pd.bdate_range("2024-01-01", periods=n)
    ret = rng.normal(0.0003, 0.015, n)
    close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.integers(1_000_000, 5_000_000, n).astype(float),
                    index=idx)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": vol,
    })

    feats = compute_volume_features(df)
    assert "error" not in feats, f"计算失败: {feats.get('error')}"
    s = feats["summary"]

    print(f"OBV:           {s['obv_vs_ma']}")
    print(f"20日资金流指标: {s['vol_imbalance_20d']:+.3f}")
    print(f"60日量价相关:  {s['roll_corr_60d']:+.3f}")
    print(f"VWAP偏离:      {s['vwap_dev_pct']:+.2f}%")
    print(f"Amihud:        {s['amihud_illiq']:.4f}")
    print(f"GK年化波动:    {s['gk_vol_annual']*100:.1f}%")
    print(f"背离信号:      {s['n_divergence_90d']}次")
    print(f"研判:          {s['regime_hint']}")

    # 数值健全性检查
    assert np.isfinite(s["roll_corr_60d"]), "量价相关应为有限值"
    assert np.isfinite(s["amihud_illiq"]), "Amihud 应为有限值"
    assert np.isfinite(s["gk_vol_annual"]), "GK 波动率应为有限值"
    assert -1.05 <= s["roll_corr_60d"] <= 1.05, "相关系数越界"
    assert 0 <= s["vol_imbalance_20d"] <= 1 or -1 <= s["vol_imbalance_20d"] <= 0, \
        "资金流指标越界"

    # 边界：数据不足
    tiny = df.iloc[:5]
    r = compute_volume_features(tiny)
    assert "error" in r, "数据不足时应返回 error"

    print("\n自检通过 ✓")