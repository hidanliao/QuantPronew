"""
QuantPro v11.0 — 价格目标区间补丁（含漂移衰减修复）
======================================
在原始 quantpro_v9.py 基础上，增加以下功能：

  [K线图叠加层] ─────────────────────────────────────────
  1. Monte Carlo P5/P50/P95 未来价格带（灰色/绿色/红色渐变区间）
  2. ATR动态支撑/阻力位标注（±1ATR, ±2ATR）
  3. Fibonacci回撤位（0%, 23.6%, 38.2%, 50%, 61.8%, 100%）
  4. 预测起点竖线标注

  [价格目标卡片] ────────────────────────────────────────
  5. 概率Tab新增"价格目标区间"面板
     - 5日 / 20日 / 60日 目标价格（P5 悲观 / P50 中性 / P95 乐观）
     - 涨跌幅百分比
     - ATR止损参考价
     - Fibonacci关键位

  [诚实性注释] ──────────────────────────────────────────
  所有预测标注"统计区间，非精确预测"免责提示
  区间宽度本身即不确定性的量化体现

使用方法：
  将本文件中的类和函数复制/替换到 quantpro_v9.py 中
  或直接 import 后 monkey-patch ProbabilityDialog

======================================
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from scipy import stats
import mplfinance as mpf
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 价格目标计算引擎
# ══════════════════════════════════════════════════════════════════

class PriceTargetEngine:
    """
    价格目标区间计算
    核心理念：提供统计区间而非点预测，诚实量化不确定性
    """

    def __init__(self, close: pd.Series, ind: pd.DataFrame, current_price: float):
        self.close = close.dropna().astype(float)
        self.ind = ind
        self.current_price = current_price
        self.returns = self.close.pct_change().dropna()

    # ── Monte Carlo 价格区间 ─────────────────────────────────────
    def monte_carlo_price_bands(
        self,
        horizons: List[int] = [5, 20, 60],
        n_sims: int = 10000
    ) -> Dict[int, Dict[str, float]]:
        """
        Fat-tail Monte Carlo → 每个时间窗口的价格区间
        返回 {horizon: {p5, p25, p50, p75, p95, up_prob}}
        """
        r = self.returns
        if len(r) < 30:
            return {}

        # 拟合 t 分布（重尾）
        try:
            df_t, loc, scale = stats.t.fit(r)
            df_t = max(df_t, 2.1)  # 保证方差有限
        except Exception:
            df_t, loc, scale = 5.0, float(r.mean()), float(r.std())

        # ── v11 修复：漂移率衰减（避免长期 MC 预测虚高）──────────
        # 历史日均漂移 → 年化 → 指数衰减 → 折回日度
        mu_raw = float(loc)                            # 历史日均收益
        mu_annual = mu_raw * 252                       # 年化漂移
        # 超过1年的部分，漂移向0收敛（Ornstein-Uhlenbeck风格折扣）
        # 短期（5日）基本不变，60日只保留约30%漂移
        result = {}

        for horizon in horizons:
            # 漂移衰减：horizon越长，漂移折扣越大
            decay_factor = np.exp(-0.8 * horizon / 252)   # 60日≈0.83折扣
            loc_adj = mu_raw * decay_factor                 # 折后日度漂移
            # 使用随机种子不同于固定42，使各horizon路径独立
            rng_seed = 42 + horizon
            # t 分布随机游走（使用衰减后漂移）
            sim_returns = stats.t.rvs(
                df_t, loc=loc_adj, scale=scale,
                size=(n_sims, horizon),
                random_state=rng_seed
            )
            final_prices = self.current_price * np.cumprod(1 + sim_returns, axis=1)[:, -1]

            pct_changes = (final_prices / self.current_price - 1) * 100

            result[horizon] = {
                'p5':       float(np.percentile(final_prices, 5)),
                'p25':      float(np.percentile(final_prices, 25)),
                'p50':      float(np.percentile(final_prices, 50)),
                'p75':      float(np.percentile(final_prices, 75)),
                'p95':      float(np.percentile(final_prices, 95)),
                'p5_pct':   float(np.percentile(pct_changes, 5)),
                'p25_pct':  float(np.percentile(pct_changes, 25)),
                'p50_pct':  float(np.percentile(pct_changes, 50)),
                'p75_pct':  float(np.percentile(pct_changes, 75)),
                'p95_pct':  float(np.percentile(pct_changes, 95)),
                'up_prob':  float(np.mean(final_prices > self.current_price)),
                'df_t':     df_t,
                'n_sims':   n_sims,
            }

        return result

    # ── ATR 支撑 / 阻力 ─────────────────────────────────────────
    def atr_levels(self, multipliers: List[float] = [1.0, 1.5, 2.0]) -> Dict[str, float]:
        """
        基于 ATR 的动态支撑阻力位
        常用于止损/目标价设置
        """
        if 'atr' not in self.ind.columns:
            return {}
        atr_val = float(self.ind['atr'].iloc[-1])
        if np.isnan(atr_val) or atr_val <= 0:
            return {}

        levels = {'atr_value': atr_val}
        for m in multipliers:
            levels[f'resist_{m}x'] = round(self.current_price + m * atr_val, 4)
            levels[f'support_{m}x'] = round(self.current_price - m * atr_val, 4)

        # ATR止损建议（2ATR止损是常用机构标准）
        levels['stop_loss_atr2'] = round(self.current_price - 2 * atr_val, 4)
        levels['stop_loss_atr2_pct'] = round(-2 * atr_val / self.current_price * 100, 2)

        return levels

    # ── Fibonacci 回撤 ───────────────────────────────────────────
    def fibonacci_levels(self, lookback: int = 60) -> Dict[str, float]:
        """
        近期高低点的 Fibonacci 回撤/扩展位
        lookback: 回看天数确定高低点
        """
        window = self.close.iloc[-lookback:] if len(self.close) >= lookback else self.close
        if len(window) < 10:
            return {}

        high = float(window.max())
        low  = float(window.min())
        diff = high - low
        if diff <= 0:
            return {}

        fibs = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        # 扩展位
        ext_fibs = [1.272, 1.618]

        levels = {
            'fib_high': high,
            'fib_low':  low,
        }

        # 回撤（从高点向低点）
        for f in fibs:
            price = round(high - f * diff, 4)
            label = f'fib_{int(f*1000):04d}'  # fib_0000, fib_0236, ...
            levels[label] = price

        # 扩展位（突破后目标）
        for f in ext_fibs:
            price = round(low + f * diff, 4)
            label = f'fib_ext_{int(f*1000):04d}'
            levels[label] = price

        return levels

    # ── 综合目标价摘要 ───────────────────────────────────────────
    def summary(self) -> Dict:
        """汇总所有目标价信息"""
        mc_bands = self.monte_carlo_price_bands([5, 20, 60])
        atr_lvl  = self.atr_levels()
        fib_lvl  = self.fibonacci_levels()

        return {
            'current_price': self.current_price,
            'mc_bands':      mc_bands,
            'atr_levels':    atr_lvl,
            'fib_levels':    fib_lvl,
        }


# ══════════════════════════════════════════════════════════════════
# K线图叠加绘制器
# ══════════════════════════════════════════════════════════════════

class KlineOverlayPainter:
    """
    在 mplfinance 绘制的 K 线图基础上叠加价格目标层
    """

    # 设计令牌（与 QuantPro T 一致 — 抹茶绿浅色主题）
    GREEN   = '#16a34a'
    RED     = '#dc2626'
    GOLD    = '#a9822f'
    ACCENT  = '#1b7a43'
    PURPLE  = '#7c5cb0'
    ORANGE  = '#d97706'
    CYAN    = '#0f9c9c'
    TEXT_2  = '#6d7d70'
    TEXT_3  = '#9aab9c'
    BG2     = '#ffffff'
    BORDER  = '#dde5db'

    FIB_COLORS = {
        'fib_0000':  '#dc2626',   # 0%   — 高点，红
        'fib_0236':  '#ea580c',   # 23.6%
        'fib_0382':  '#d97706',   # 38.2%
        'fib_0500':  '#16a34a',   # 50%  — 绿
        'fib_0618':  '#1b7a43',   # 61.8%— 黄金比例，主品牌抹茶绿
        'fib_0786':  '#7c5cb0',   # 78.6%
        'fib_1000':  '#0f9c9c',   # 100% — 低点，青
    }
    FIB_LABELS = {
        'fib_0000': '0%',
        'fib_0236': '23.6%',
        'fib_0382': '38.2%',
        'fib_0500': '50%',
        'fib_0618': '61.8%',
        'fib_0786': '78.6%',
        'fib_1000': '100%',
    }

    @staticmethod
    def overlay_price_targets(
        ax,                        # matplotlib Axes（主K线轴）
        df: pd.DataFrame,          # OHLCV DataFrame（已绘制的K线数据）
        target_data: Dict,         # PriceTargetEngine.summary() 返回值
        horizons: List[int] = [5, 20, 60],
        show_mc: bool = True,
        show_atr: bool = True,
        show_fib: bool = True,
        currency: str = '$',
    ):
        """
        在已有的 K 线图 axes 上叠加目标区间
        """
        if df.empty or not target_data:
            return

        cp   = target_data.get('current_price', 0)
        mc   = target_data.get('mc_bands', {})
        atr  = target_data.get('atr_levels', {})
        fib  = target_data.get('fib_levels', {})

        # x 轴最后位置（数值索引）
        n_bars  = len(df)
        x_last  = n_bars - 1    # 最后一根K线的 x 位置（mplfinance 用整数索引）

        # ── 1. Monte Carlo 价格带 ─────────────────────────────
        if show_mc and mc:
            horizon_styles = {
                5:  {'color': KlineOverlayPainter.ACCENT,  'ls': '-',  'lw': 1.2, 'label': '5日'},
                20: {'color': KlineOverlayPainter.GREEN,   'ls': '--', 'lw': 1.4, 'label': '20日'},
                60: {'color': KlineOverlayPainter.GOLD,    'ls': ':',  'lw': 1.6, 'label': '60日'},
            }
            for h in horizons:
                if h not in mc:
                    continue
                band = mc[h]
                st   = horizon_styles.get(h, horizon_styles[20])
                col  = st['color']
                x_end = x_last + h   # 投影到未来

                # P50 中位线
                ax.plot(
                    [x_last, x_end],
                    [cp, band['p50']],
                    color=col, ls=st['ls'], lw=st['lw'], alpha=0.9,
                    label=f"{st['label']} P50 {currency}{band['p50']:.2f} ({band['p50_pct']:+.1f}%)"
                )
                # P25–P75 区间（主体概率带）
                ax.fill_between(
                    [x_last, x_end],
                    [cp, band['p25']],
                    [cp, band['p75']],
                    color=col, alpha=0.10
                )
                # P5–P95 区间（尾部范围）
                ax.fill_between(
                    [x_last, x_end],
                    [cp, band['p5']],
                    [cp, band['p95']],
                    color=col, alpha=0.05
                )
                # P5 / P95 端点标注
                up_col = KlineOverlayPainter.GREEN if band['p95_pct'] >= 0 else KlineOverlayPainter.RED
                dn_col = KlineOverlayPainter.RED   if band['p5_pct']  <= 0 else KlineOverlayPainter.GREEN
                ax.annotate(
                    f"  {currency}{band['p95']:.2f}\n  ({band['p95_pct']:+.1f}%)",
                    xy=(x_end, band['p95']), color=up_col, fontsize=7, va='bottom',
                    annotation_clip=False
                )
                ax.annotate(
                    f"  {currency}{band['p5']:.2f}\n  ({band['p5_pct']:+.1f}%)",
                    xy=(x_end, band['p5']), color=dn_col, fontsize=7, va='top',
                    annotation_clip=False
                )

            # 当前价竖线
            ax.axvline(x_last, color=KlineOverlayPainter.TEXT_3,
                       lw=1.2, ls='--', alpha=0.7, label='当前价')

        # ── 2. ATR 支撑/阻力 ──────────────────────────────────
        if show_atr and atr:
            atr_val = atr.get('atr_value', 0)
            atr_defs = [
                ('resist_2.0x', KlineOverlayPainter.RED,    '阻力 +2ATR', ':',  0.75),
                ('resist_1.5x', KlineOverlayPainter.ORANGE, '阻力 +1.5ATR',':', 0.6),
                ('resist_1.0x', KlineOverlayPainter.GOLD,   '阻力 +1ATR', '-.',  0.7),
                ('support_1.0x',KlineOverlayPainter.GREEN,  '支撑 -1ATR', '-.',  0.7),
                ('support_1.5x',KlineOverlayPainter.CYAN,   '支撑 -1.5ATR',':',  0.6),
                ('support_2.0x',KlineOverlayPainter.ACCENT, '支撑 -2ATR (止损)',':',0.75),
            ]
            for key, col, label, ls, alpha in atr_defs:
                price = atr.get(key)
                if price is None:
                    continue
                ax.axhline(price, color=col, ls=ls, lw=0.9, alpha=alpha,
                           label=f"{label}: {currency}{price:.2f}")
                # 右侧小标签
                ax.annotate(
                    f' {label.split()[0]}: {currency}{price:.2f}',
                    xy=(n_bars - 1, price),
                    xytext=(5, 0), textcoords='offset points',
                    color=col, fontsize=6.5, va='center', alpha=0.85,
                    annotation_clip=False
                )

        # ── 3. Fibonacci 回撤 ──────────────────────────────────
        if show_fib and fib:
            for fib_key, fib_label in KlineOverlayPainter.FIB_LABELS.items():
                price = fib.get(fib_key)
                if price is None:
                    continue
                col = KlineOverlayPainter.FIB_COLORS.get(fib_key, '#888888')
                ax.axhline(price, color=col, ls='-', lw=0.7, alpha=0.55)
                ax.annotate(
                    f' Fib {fib_label}: {currency}{price:.2f}',
                    xy=(0, price),
                    xytext=(6, 2), textcoords='offset points',
                    color=col, fontsize=6, va='bottom', alpha=0.75,
                    annotation_clip=False
                )

        # ── 图例（移到绘图区外·右侧，不再压缩K线）──────────────
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                fontsize=6.5, ncol=1,
                facecolor=KlineOverlayPainter.BG2,
                labelcolor=KlineOverlayPainter.TEXT_2,
                edgecolor=KlineOverlayPainter.BORDER,
                framealpha=0.9,
                loc='upper left',
                bbox_to_anchor=(1.005, 1.0),   # 锚到坐标轴右侧外部
                borderaxespad=0.0,
            )
            try:
                # 给右侧图例留出空间，避免被裁掉
                ax.figure.subplots_adjust(right=0.80)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════
# 价格目标卡片 Widget（替换 / 扩展原 ProbabilityDialog._build_tab_prob）
# ══════════════════════════════════════════════════════════════════

try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
        QGridLayout, QSizePolicy, QScrollArea
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QColor, QFont
    HAS_QT = True
except ImportError:
    HAS_QT = False


if HAS_QT:

    # ── 设计令牌（与 QuantPro T 同步 — 抹茶绿浅色主题）──
    class _T:
        BG0='#f4f7f2'; BG1='#ffffff'; BG2='#ffffff'; BG3='#eef3ea'; BG4='#dceadb'
        ACCENT='#1b7a43'; ACCENT2='#2f9e5c'; GOLD='#a9822f'
        GREEN='#16a34a'; RED='#dc2626'; YELLOW='#d97706'
        TEXT_H='#16241a'; TEXT_1='#33413a'; TEXT_2='#6d7d70'; TEXT_3='#9aab9c'
        BORDER='#dde5db'; PURPLE='#7c5cb0'; CYAN='#0f9c9c'; ORANGE='#d97706'

    class PriceTargetCard(QFrame):
        """
        单个时间窗口的价格目标卡片
        显示：P5/P50/P95 目标价 + 涨跌幅 + 上涨概率
        """

        def __init__(self, horizon: int, band: dict, currency: str = '$', parent=None):
            super().__init__(parent)
            self.setStyleSheet(
                f"QFrame{{background:{_T.BG3};border-radius:10px;"
                f"border:1px solid {_T.BORDER};}}"
            )
            self._build(horizon, band, currency)

        def _build(self, horizon: int, band: dict, currency: str):
            lay = QVBoxLayout(self)
            lay.setContentsMargins(14, 12, 14, 12)
            lay.setSpacing(6)

            # ─ 标题行 ─
            title_row = QHBoxLayout()
            title = QLabel(f"{horizon}日预测区间")
            title.setStyleSheet(
                f"color:{_T.TEXT_2};font-size:9pt;font-weight:600;border:none;"
            )
            up_pct = band.get('up_prob', 0.5) * 100
            up_col = _T.GREEN if up_pct >= 55 else (_T.RED if up_pct <= 45 else _T.YELLOW)
            up_lbl = QLabel(f"上涨概率 {up_pct:.0f}%")
            up_lbl.setStyleSheet(
                f"color:{up_col};font-size:9pt;font-weight:700;border:none;"
            )
            up_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            title_row.addWidget(title)
            title_row.addStretch()
            title_row.addWidget(up_lbl)
            lay.addLayout(title_row)

            # ─ 三档价格 ─
            grid = QGridLayout()
            grid.setSpacing(4)

            rows_def = [
                ('乐观 P95', 'p95', 'p95_pct', _T.GREEN),
                ('中性 P50', 'p50', 'p50_pct', _T.TEXT_H),
                ('悲观 P5',  'p5',  'p5_pct',  _T.RED),
            ]

            for r_idx, (label_str, price_key, pct_key, col) in enumerate(rows_def):
                price = band.get(price_key, 0)
                pct   = band.get(pct_key,   0)

                lbl = QLabel(label_str)
                lbl.setStyleSheet(
                    f"color:{_T.TEXT_3};font-size:8.5pt;border:none;"
                )
                price_lbl = QLabel(f"{currency}{price:.2f}")
                price_lbl.setStyleSheet(
                    f"color:{col};font-size:11pt;font-weight:700;border:none;"
                )
                price_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                pct_lbl = QLabel(f"{pct:+.1f}%")
                pct_lbl.setStyleSheet(
                    f"color:{col};font-size:9pt;font-weight:600;border:none;"
                )
                pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

                grid.addWidget(lbl,       r_idx, 0)
                grid.addWidget(price_lbl, r_idx, 1)
                grid.addWidget(pct_lbl,   r_idx, 2)

            lay.addLayout(grid)

            # ─ 免责小字 ─
            disc = QLabel("* t分布MC + 漂移衰减，统计区间，非精确预测")
            disc.setStyleSheet(
                f"color:{_T.TEXT_3};font-size:7.5pt;border:none;"
            )
            lay.addWidget(disc)


    class ATRFibPanel(QFrame):
        """
        ATR 止损参考 + Fibonacci 关键位面板
        """

        def __init__(self, atr_levels: dict, fib_levels: dict,
                     currency: str = '$', parent=None):
            super().__init__(parent)
            self.setStyleSheet(
                f"QFrame{{background:{_T.BG2};border-radius:8px;"
                f"border:1px solid {_T.BORDER};}}"
            )
            self._build(atr_levels, fib_levels, currency)

        def _build(self, atr: dict, fib: dict, currency: str):
            lay = QHBoxLayout(self)
            lay.setContentsMargins(14, 10, 14, 10)
            lay.setSpacing(20)

            # ── ATR ──
            atr_box = QVBoxLayout()
            atr_hdr = QLabel("ATR 止损 / 目标")
            atr_hdr.setStyleSheet(
                f"color:{_T.GOLD};font-size:9.5pt;font-weight:700;border:none;"
            )
            atr_box.addWidget(atr_hdr)

            atr_defs = [
                (f"阻力 +2ATR", 'resist_2.0x', _T.RED),
                (f"阻力 +1ATR", 'resist_1.0x', _T.ORANGE),
                (f"支撑 -1ATR", 'support_1.0x', _T.GREEN),
                (f"止损 -2ATR", 'stop_loss_atr2', _T.ACCENT),
            ]
            atr_val = atr.get('atr_value', 0)
            if atr_val:
                atr_box.addWidget(self._kv(
                    f"ATR值", f"{currency}{atr_val:.3f}", _T.TEXT_2
                ))
            for label, key, col in atr_defs:
                val = atr.get(key)
                if val is not None:
                    extra = ""
                    if key == 'stop_loss_atr2':
                        extra = f"  ({atr.get('stop_loss_atr2_pct', 0):+.1f}%)"
                    atr_box.addWidget(self._kv(label, f"{currency}{val:.2f}{extra}", col))
            lay.addLayout(atr_box)

            # 分割线
            sep = QFrame()
            sep.setFrameShape(QFrame.VLine)
            sep.setStyleSheet(
                f"background:{_T.BORDER};max-width:1px;border:none;"
            )
            lay.addWidget(sep)

            # ── Fibonacci ──
            fib_box = QVBoxLayout()
            fib_hdr = QLabel("Fibonacci 回撤位（近60日）")
            fib_hdr.setStyleSheet(
                f"color:{_T.PURPLE};font-size:9.5pt;font-weight:700;border:none;"
            )
            fib_box.addWidget(fib_hdr)

            fib_defs = [
                ('fib_0000', '0.0%',   '#dc2626'),
                ('fib_0236', '23.6%',  '#ea580c'),
                ('fib_0382', '38.2%',  '#d97706'),
                ('fib_0500', '50.0%',  '#16a34a'),
                ('fib_0618', '61.8%*', '#1b7a43'),
                ('fib_1000', '100%',   '#0f9c9c'),
            ]
            for key, label, col in fib_defs:
                val = fib.get(key)
                if val is not None:
                    fib_box.addWidget(self._kv(f"Fib {label}", f"{currency}{val:.2f}", col))
            lay.addLayout(fib_box)

        @staticmethod
        def _kv(label: str, value: str, col: str) -> QLabel:
            lbl = QLabel(
                f"<span style='color:{_T.TEXT_3};font-size:8.5pt;'>{label}: </span>"
                f"<span style='color:{col};font-size:9pt;font-weight:600;'>{value}</span>"
            )
            lbl.setTextFormat(Qt.RichText)
            lbl.setStyleSheet("border:none;")
            return lbl


    class PriceTargetWidget(QWidget):
        """
        完整的价格目标面板
        嵌入到 ProbabilityDialog 的 tab_prob 中，
        放置在原有概率卡片下方
        """

        def __init__(self, target_data: dict, currency: str = '$', parent=None):
            super().__init__(parent)
            self._build(target_data, currency)

        def _build(self, data: dict, currency: str):
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 8, 0, 0)
            root.setSpacing(10)

            mc = data.get('mc_bands', {})
            atr = data.get('atr_levels', {})
            fib = data.get('fib_levels', {})
            cp  = data.get('current_price', 0)

            # ─ 标题 ─
            hdr_row = QHBoxLayout()
            hdr = QLabel("价格目标区间（GARCH/t分布 MC · 漂移衰减 · 10,000次模拟）")
            hdr.setStyleSheet(
                f"color:{_T.GOLD};font-size:10pt;font-weight:700;border:none;"
            )
            cp_lbl = QLabel(f"当前价 {currency}{cp:.2f}")
            cp_lbl.setStyleSheet(
                f"color:{_T.TEXT_2};font-size:9pt;border:none;"
            )
            cp_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            hdr_row.addWidget(hdr)
            hdr_row.addStretch()
            hdr_row.addWidget(cp_lbl)
            root.addLayout(hdr_row)

            # ─ MC 卡片行 ─
            card_row = QHBoxLayout()
            card_row.setSpacing(8)
            for h in [5, 20, 60]:
                band = mc.get(h)
                if band:
                    card = PriceTargetCard(h, band, currency)
                    card_row.addWidget(card)
            root.addLayout(card_row)

            # ─ ATR + Fib ─
            if atr or fib:
                panel = ATRFibPanel(atr, fib, currency)
                root.addWidget(panel)

            # ─ 全局免责 ─
            disc_frame = QFrame()
            disc_frame.setStyleSheet(
                f"QFrame{{background:{_T.BG1};border-radius:6px;"
                f"border:1px solid {_T.BORDER};}}"
            )
            disc_lay = QVBoxLayout(disc_frame)
            disc_lay.setContentsMargins(12, 8, 12, 8)
            disc_text = QLabel(
                "⚠️  <b>统计区间说明</b>：以上价格区间由历史收益率的 t 分布 Monte Carlo 模拟得出，"
                "反映不确定性范围而非精确预测。区间越宽表示波动率越高、未来越不确定。"
                "P5/P95 分别代表悲观/乐观的极端情景（各占5%概率）。"
                "<b>本工具不构成投资建议，请结合基本面和自身风险承受能力独立判断。</b>"
            )
            disc_text.setWordWrap(True)
            disc_text.setStyleSheet(
                f"color:{_T.TEXT_2};font-size:8.5pt;border:none;"
            )
            disc_text.setTextFormat(Qt.RichText)
            disc_lay.addWidget(disc_text)
            root.addWidget(disc_frame)


# ══════════════════════════════════════════════════════════════════
# Monkey-patch: 将价格目标功能注入 ProbabilityDialog
# ══════════════════════════════════════════════════════════════════

def patch_probability_dialog(ProbabilityDialog_class, get_mpl_rc_fn, apply_font_fn,
                              make_mpf_style_fn, T_class, t_fn, mpl_style_fn,
                              fetch_stock_data_fn, get_realtime_price_fn):
    """
    将价格目标功能注入已有的 ProbabilityDialog 类
    在 quantpro_v9.py 末尾调用此函数即可

    用法（在 quantpro_v9.py 末尾，if __name__=="__main__": 之前）：

        from quantpro_price_target_patch import patch_probability_dialog
        patch_probability_dialog(
            ProbabilityDialog, _get_mpl_rc, _apply_font_to_figure,
            _make_mpf_style, T, t, _mpl_style,
            fetch_stock_data, get_realtime_price
        )
    """

    original_build_tab_kline = ProbabilityDialog_class._build_tab_kline
    original_kline_plot      = ProbabilityDialog_class._kline_plot
    original_build_tab_prob  = ProbabilityDialog_class._build_tab_prob
    original_run             = ProbabilityDialog_class._run

    # ── 替换 _build_tab_kline（添加叠加层开关）──────────────────
    def new_build_tab_kline(self):
        from PyQt5.QtWidgets import QCheckBox
        original_build_tab_kline(self)

        # 在按钮行末尾追加叠加层开关
        btn_row_layout = self.layout().itemAt(2).widget().layout()  # central_tab
        # 直接在 kline tab 内找按钮行更稳妥
        kline_tab_widget = self.tabs.widget(4)  # tab_kline 是第5个tab
        kline_lay = kline_tab_widget.layout() if kline_tab_widget else None
        if kline_lay and kline_lay.count() >= 3:
            btn_row = kline_lay.itemAt(kline_lay.count() - 1)

        # 叠加层开关
        self._show_mc_overlay  = True
        self._show_atr_overlay = True
        self._show_fib_overlay = True
        self._target_data      = None

    ProbabilityDialog_class._build_tab_kline = new_build_tab_kline

    # ── 替换 _kline_plot（核心：叠加价格目标层）────────────────
    def new_kline_plot(self):
        """扩展版 K 线绘制，添加价格目标叠加层"""
        sym = self.sym
        period = self._kline_period

        df = fetch_stock_data_fn(sym, period)
        if df.empty:
            self.kline_info.setText(t_fn("kline_no_data"))
            return

        ohlcv = df[['Open','High','Low','Close','Volume']].copy()
        for c in ohlcv.columns:
            ohlcv[c] = ohlcv[c].squeeze().astype(float)
        ohlcv.dropna(inplace=True)
        if len(ohlcv) > 1500:
            ohlcv = ohlcv.iloc[-1500:]

        cur_sym = _currency_fn(sym)
        rp = get_realtime_price_fn(sym)
        if rp:
            cp  = rp
            src = "实时"
            chg = (cp - float(ohlcv['Close'].iloc[-1])) / float(ohlcv['Close'].iloc[-1]) * 100
        else:
            cp  = float(ohlcv['Close'].iloc[-1])
            src = "历史"
            chg = (float(ohlcv['Close'].iloc[-1]) - float(ohlcv['Close'].iloc[-2])) / float(ohlcv['Close'].iloc[-2]) * 100 if len(ohlcv) > 1 else 0

        cc = T_class.GREEN if chg >= 0 else T_class.RED
        self.kline_info.setText(
            f"<b>{sym}</b>  {src}: <b>{cur_sym}{cp:.2f}</b>  "
            f"<b style='color:{cc}'>{chg:+.2f}%</b>  "
            f"{t_fn('kline_bars', n=len(ohlcv))} [{period.upper()}]"
        )

        try:
            # ── 基础 K 线 ──
            style = make_mpf_style_fn()
            mav_args = (5, 20) if len(ohlcv) < 60 else (5, 20, 60)

            with mpl.rc_context(get_mpl_rc_fn()):
                fig, axes = mpf.plot(
                    ohlcv, type='candle', volume=True, mav=mav_args,
                    returnfig=True, title=f"  {sym}",
                    style=style, figsize=(8, 5)
                )

            # ── 计算价格目标 ──
            if self._ind is not None:
                engine = PriceTargetEngine(
                    self._ind['close'],
                    self._ind,
                    cp
                )
                self._target_data = engine.summary()

                # ── 叠加层 ──
                main_ax = axes[0]  # 第一个轴是价格轴
                getattr(self, '_show_mc_overlay',  True)
                show_mc  = getattr(self, '_show_mc_overlay',  True)
                show_atr = getattr(self, '_show_atr_overlay', True)
                show_fib = getattr(self, '_show_fib_overlay', True)

                KlineOverlayPainter.overlay_price_targets(
                    ax          = main_ax,
                    df          = ohlcv,
                    target_data = self._target_data,
                    horizons    = [5, 20, 60],
                    show_mc     = show_mc,
                    show_atr    = show_atr,
                    show_fib    = show_fib,
                    currency    = cur_sym,
                )

            apply_font_fn(fig)
            self.kline_canvas.figure = fig
            fig.canvas = self.kline_canvas
            self.kline_canvas.draw()

            # ── 同步更新概率Tab价格目标卡片 ──
            if self._target_data:
                self._refresh_price_target_widget(cur_sym)

        except Exception as e:
            self.kline_info.setText(t_fn("kline_plot_fail", e=e))
            logger.error(f"KlinePlot error: {e}", exc_info=True)

    ProbabilityDialog_class._kline_plot = new_kline_plot

    # ── 注入 _refresh_price_target_widget ──────────────────────
    def _refresh_price_target_widget(self, currency: str):
        """在概率Tab底部更新/插入价格目标卡片"""
        if not hasattr(self, '_target_data') or not self._target_data:
            return
        if not HAS_QT:
            return

        tab_prob_widget = self.tabs.widget(0)  # tab_prob 是第1个tab
        if tab_prob_widget is None:
            return

        tab_lay = tab_prob_widget.layout()
        if tab_lay is None:
            return

        # 移除旧的 PriceTargetWidget（若有）
        if hasattr(self, '_price_target_widget') and self._price_target_widget:
            tab_lay.removeWidget(self._price_target_widget)
            self._price_target_widget.deleteLater()
            self._price_target_widget = None

        # 插入新的
        try:
            widget = PriceTargetWidget(self._target_data, currency, self)
            tab_lay.addWidget(widget)
            self._price_target_widget = widget
        except Exception as e:
            logger.warning(f"PriceTargetWidget error: {e}")

    ProbabilityDialog_class._refresh_price_target_widget = _refresh_price_target_widget

    return ProbabilityDialog_class


def _currency_fn(sym: str) -> str:
    """从 quantpro_v9 复制的货币符号函数"""
    s = sym.upper()
    if s.endswith(".HK"):   return "HK$"
    if s.endswith((".SS", ".SZ", ".T")): return "¥"
    if s.endswith(".L"):    return "£"
    if s.endswith((".PA", ".DE", ".MI")): return "€"
    return "$"


# ══════════════════════════════════════════════════════════════════
# 独立运行：演示 PriceTargetEngine 输出
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import yfinance as yf
    print("=" * 60)
    print("PriceTargetEngine 演示 — AAPL")
    print("=" * 60)

    df   = yf.download("AAPL", period="1y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze().astype(float)
    high  = df['High'].squeeze().astype(float)
    low   = df['Low'].squeeze().astype(float)
    vol   = df['Volume'].squeeze().astype(float)

    # 简单指标
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0.)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    tr    = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr   = tr.rolling(14).mean()

    ind = pd.DataFrame({'rsi': rsi, 'atr': atr, 'close': close})
    cp  = float(close.iloc[-1])

    engine  = PriceTargetEngine(close, ind, cp)
    summary = engine.summary()

    print(f"\n当前价: ${cp:.2f}\n")
    for h, band in summary['mc_bands'].items():
        print(f"[{h:2d}日] P5=${band['p5']:.2f}({band['p5_pct']:+.1f}%)  "
              f"P50=${band['p50']:.2f}({band['p50_pct']:+.1f}%)  "
              f"P95=${band['p95']:.2f}({band['p95_pct']:+.1f}%)  "
              f"上涨概率={band['up_prob']*100:.1f}%")

    print("\n[ATR止损]")
    atr_d = summary['atr_levels']
    print(f"  ATR值: ${atr_d.get('atr_value',0):.3f}")
    print(f"  -1ATR支撑: ${atr_d.get('support_1.0x',0):.2f}")
    print(f"  -2ATR止损: ${atr_d.get('stop_loss_atr2',0):.2f} ({atr_d.get('stop_loss_atr2_pct',0):+.1f}%)")

    print("\n[Fibonacci 回撤 近60日]")
    fib_d = summary['fib_levels']
    for k in ['fib_0000','fib_0236','fib_0382','fib_0500','fib_0618','fib_1000']:
        v = fib_d.get(k)
        if v: print(f"  {KlineOverlayPainter.FIB_LABELS[k]:8s} ${v:.2f}")
