"""
QuantPro — 大盘状态客观仪表盘（MarketRegimeDashboard）v1.0
══════════════════════════════════════════════════════════════════
把"主观盯大盘三五年"换成客观、可计算、规则透明的市场环境读数。

设计铁律：
  · 每个指标都用 OHLCV 机械计算，没有任何"我觉得/可能/预计"
  · 合成的 0–100「环境分」只是各客观指标的加权加总，**不是预测**
  · 面板里每一项都摊开原始值+得分，不是黑箱
  · 全程纯 yfinance 数据（指数/ETF），零额外数据源

指标（全部可回测）：
  趋势        SPX 相对 50/200 日均线位置；50 vs 200 金叉/死叉
  动量        SPX 20/60 日收益
  波动率      VIX 当前值 + 过去1年百分位
  期限结构    VIX vs VIX3M（倒挂=短期恐慌）
  回撤        SPX 距 52 周高点回撤
  风险偏好    进攻板块(QQQ/XLK) vs 防守板块(XLU/XLP) 20日相对强弱
  广度近似    11个行业ETF中价格在50日线上方的比例

集成（quantpro_v1_6.py 末尾、创建 QuantApp 之前，与其他注入并列）：

    from quantpro_market_dashboard import install_market_dashboard
    install_market_dashboard(globals())

主窗口按钮行新增「大盘仪表盘」按钮。

依赖：仅用主程序已有的 PyQt5 / matplotlib / pandas / numpy。
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
from typing import Optional, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QApplication, QGridLayout, QFrame, QScrollArea, QWidget
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.gridspec as gridspec

try:
    import quantpro_ticker_logos as _tlogos   # 可选：装了logo模块才显示板块ETF logo条
except ImportError:
    _tlogos = None

logger = logging.getLogger(__name__)

# 板块 ETF（纯 yfinance 可取）
ATTACK_ETFS = ["QQQ", "XLK", "XLY"]            # 进攻：科技/成长/可选消费
DEFENSE_ETFS = ["XLU", "XLP", "XLV"]           # 防守：公用/必需消费/医疗
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
               "XLP", "XLU", "XLB", "XLRE", "XLC"]   # 11 大行业（广度近似）


# ══════════════════════════════════════════════════════════════════
# 计算层（纯函数）
# ══════════════════════════════════════════════════════════════════
def _close(df) -> Optional[pd.Series]:
    if df is None or len(df) == 0:
        return None
    try:
        s = df["Close"]
        s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
        return s.astype(float).dropna()
    except Exception:
        return None


def _pct_rank(series: pd.Series, value: float, win: int = 252) -> float:
    """value 在 series 最近 win 个值里的百分位 (0~100)。"""
    tail = series.dropna().tail(win)
    if len(tail) < 20:
        return np.nan
    return float((tail < value).mean() * 100)


def compute_market_regime(fetch_fn: Callable) -> dict:
    """
    抓取指数/ETF 并计算全部客观指标 + 0–100 环境分。
    返回 dict：score, label, components[list], spx(用于画图), 各原始序列。
    fetch_fn: 主程序 fetch_stock_data(sym, period)
    """
    spx = _close(fetch_fn("^GSPC", "2y"))
    if spx is None or len(spx) < 210:
        return {"ok": False, "msg": "SPX(^GSPC) 数据不足，无法计算"}

    vix = _close(fetch_fn("^VIX", "2y"))
    vix3m = _close(fetch_fn("^VIX3M", "1y"))

    comps: List[dict] = []   # 每项: name, value(显示), score(0~100), note

    # ── 1) 趋势：相对 50/200 日均线 ──
    ma50 = spx.rolling(50).mean()
    ma200 = spx.rolling(200).mean()
    px = float(spx.iloc[-1]); m50 = float(ma50.iloc[-1]); m200 = float(ma200.iloc[-1])
    above50 = px > m50; above200 = px > m200
    trend_score = (50 if above200 else 0) + (50 if above50 else 0)
    comps.append({
        "name": "趋势结构", "value": f"价{px:.0f} / 50日{m50:.0f} / 200日{m200:.0f}",
        "score": trend_score,
        "note": ("200日上方·多头" if above200 else "200日下方·空头")
                + ("，50日上方" if above50 else "，50日下方")})

    # ── 2) 金叉/死叉（50 vs 200）──
    gc_now = m50 > m200
    # 最近一次状态翻转
    state = (ma50 > ma200).dropna()
    flip = state.ne(state.shift()).fillna(False)
    last_flip = state.index[flip][-1] if flip.any() else None
    days_since = int((spx.index[-1] - last_flip).days) if last_flip is not None else None
    cross_score = 100 if gc_now else 0
    comps.append({
        "name": "金叉/死叉", "value": ("金叉(50>200)" if gc_now else "死叉(50<200)")
                 + (f"，已{days_since}天" if days_since is not None else ""),
        "score": cross_score,
        "note": "多头排列" if gc_now else "空头排列"})

    # ── 3) 动量：20/60 日收益 ──
    r20 = float(spx.iloc[-1] / spx.iloc[-21] - 1) * 100 if len(spx) > 21 else np.nan
    r60 = float(spx.iloc[-1] / spx.iloc[-61] - 1) * 100 if len(spx) > 61 else np.nan
    # 打分：20日收益映射，-5%→0, +5%→100
    mom_score = float(np.clip((r20 + 5) / 10 * 100, 0, 100)) if np.isfinite(r20) else 50
    comps.append({
        "name": "动量", "value": f"20日 {r20:+.1f}% / 60日 {r60:+.1f}%",
        "score": mom_score, "note": "近月走势"})

    # ── 4) 波动率：VIX 水平 + 百分位 ──
    if vix is not None and len(vix) > 60:
        vix_now = float(vix.iloc[-1])
        vix_pct = _pct_rank(vix, vix_now, 252)
        # VIX 百分位越低越好：0分位→100分，100分位→0分
        vol_score = float(np.clip(100 - vix_pct, 0, 100)) if np.isfinite(vix_pct) else 50
        comps.append({
            "name": "波动率(VIX)", "value": f"{vix_now:.1f}（1年{vix_pct:.0f}分位）",
            "score": vol_score,
            "note": "市场紧张" if vix_pct > 75 else ("平静" if vix_pct < 25 else "中性")})
    else:
        vix_now = np.nan
        comps.append({"name": "波动率(VIX)", "value": "VIX无数据",
                      "score": 50, "note": "降级中性"})

    # ── 5) VIX 期限结构：VIX vs VIX3M ──
    if vix is not None and vix3m is not None and len(vix3m) > 5:
        v1 = float(vix.iloc[-1]); v3 = float(vix3m.iloc[-1])
        inverted = v1 > v3
        ts_score = 0 if inverted else 100
        comps.append({
            "name": "VIX期限结构", "value": f"VIX {v1:.1f} vs VIX3M {v3:.1f}",
            "score": ts_score,
            "note": "倒挂·短期恐慌" if inverted else "正常·无短期恐慌"})

    # ── 6) 回撤：距 52 周高点 ──
    hi52 = float(spx.tail(252).max())
    dd = float(spx.iloc[-1] / hi52 - 1) * 100
    # 0%回撤→100分, -20%→0分
    dd_score = float(np.clip((dd + 20) / 20 * 100, 0, 100))
    comps.append({
        "name": "距52周高点", "value": f"{dd:+.1f}%",
        "score": dd_score,
        "note": "接近高点" if dd > -3 else ("深度回撤" if dd < -15 else "正常回撤")})

    # ── 7) 风险偏好：进攻 vs 防守 ETF 20日相对强弱 ──
    def _avg_ret(etfs, win):
        rs = []
        for e in etfs:
            s = _close(fetch_fn(e, "6mo"))
            if s is not None and len(s) > win:
                rs.append(float(s.iloc[-1] / s.iloc[-win - 1] - 1))
        return np.mean(rs) if rs else np.nan
    atk = _avg_ret(ATTACK_ETFS, 20)
    dfn = _avg_ret(DEFENSE_ETFS, 20)
    if np.isfinite(atk) and np.isfinite(dfn):
        spread = (atk - dfn) * 100
        risk_score = float(np.clip((spread + 4) / 8 * 100, 0, 100))
        comps.append({
            "name": "风险偏好", "value": f"进攻−防守 20日 {spread:+.1f}%",
            "score": risk_score,
            "note": "追逐风险(Risk-On)" if spread > 1 else
                    ("避险(Risk-Off)" if spread < -1 else "中性")})
    else:
        comps.append({"name": "风险偏好", "value": "板块ETF数据不足",
                      "score": 50, "note": "降级中性"})

    # ── 8) 广度近似：行业ETF在50日线上方比例 ──
    above_cnt = 0; total = 0; sector_status = []
    for e in SECTOR_ETFS:
        s = _close(fetch_fn(e, "6mo"))
        if s is not None and len(s) > 55:
            m = s.rolling(50).mean()
            on = float(s.iloc[-1]) > float(m.iloc[-1])
            above_cnt += int(on); total += 1
            sector_status.append((e, on, float(s.iloc[-1] / s.iloc[-21] - 1) * 100))
    if total > 0:
        breadth_pct = above_cnt / total * 100
        comps.append({
            "name": "板块广度", "value": f"{above_cnt}/{total} 在50日线上方",
            "score": float(breadth_pct),
            "note": "普涨" if breadth_pct > 70 else
                    ("普跌" if breadth_pct < 30 else "分化")})
    else:
        breadth_pct = np.nan
        comps.append({"name": "板块广度", "value": "ETF数据不足",
                      "score": 50, "note": "降级中性"})

    # ── 合成环境分（透明加权）──
    weights = {
        "趋势结构": 0.18, "金叉/死叉": 0.12, "动量": 0.12,
        "波动率(VIX)": 0.15, "VIX期限结构": 0.08, "距52周高点": 0.10,
        "风险偏好": 0.13, "板块广度": 0.12,
    }
    num = den = 0.0
    for c in comps:
        w = weights.get(c["name"], 0.0)
        num += w * c["score"]; den += w
    score = num / den if den > 0 else 50.0

    if score >= 70:    label, lcolor = "偏进攻 (Risk-On)", "green"
    elif score >= 55:  label, lcolor = "中性偏多", "lime"
    elif score >= 45:  label, lcolor = "中性", "gold"
    elif score >= 30:  label, lcolor = "中性偏防守", "orange"
    else:              label, lcolor = "偏防守 (Risk-Off)", "red"

    return {
        "ok": True, "score": round(score, 1), "label": label, "lcolor": lcolor,
        "components": comps, "weights": weights,
        "spx": spx, "ma50": ma50, "ma200": ma200,
        "vix": vix, "sector_status": sector_status,
    }


# ══════════════════════════════════════════════════════════════════
# 仪表盘窗口
# ══════════════════════════════════════════════════════════════════
class MarketRegimeDashboard(QDialog):
    def __init__(self, env: dict, parent=None):
        super().__init__(parent)
        self.env = env
        self.T = env["T"]
        self.setWindowTitle("大盘状态仪表盘")
        try:
            self.setWindowIcon(env["_make_app_icon"]())
        except Exception:
            pass
        self.resize(1000, 820)
        # 补最小化/最大化按钮，去掉无用的帮助(?)按钮
        try:
            _f = self.windowFlags()
            _f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
            _f &= ~Qt.WindowContextHelpButtonHint
            self.setWindowFlags(_f)
        except Exception:
            pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self._data = None
        self._build_ui()
        QTimer.singleShot(150, self._refresh)

    def _build_ui(self):
        T = self.T
        lay = QVBoxLayout(self); lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(8)

        top = QHBoxLayout()
        hdr = QLabel("<b>大盘状态仪表盘</b>　客观指标机械加总 · 非预测")
        hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;border:none;")
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.setStyleSheet(self.refresh_btn.styleSheet() +
                                       f"color:{T.CYAN};border-color:{T.CYAN};")
        self.refresh_btn.clicked.connect(self._refresh)
        top.addWidget(hdr, 1); top.addWidget(self.refresh_btn, 0)
        lay.addLayout(top)

        # 环境分大字
        self.score_lbl = QLabel("计算中…")
        self.score_lbl.setAlignment(Qt.AlignCenter)
        self.score_lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:15pt;"
                                     f"background:{T.BG2};border-radius:8px;padding:10px;")
        lay.addWidget(self.score_lbl)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
        lay.addWidget(self.status)

        # 板块ETF速览条（logo + 代码 + 当日涨跌%，TradingView那种ticker strip风格）
        strip_scroll = QScrollArea()
        strip_scroll.setWidgetResizable(True)
        strip_scroll.setFixedHeight(58)
        strip_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        strip_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        strip_scroll.setStyleSheet(f"border:1px solid {T.BORDER};border-radius:6px;")
        self.sector_strip_host = QWidget()
        self.sector_strip_lay = QHBoxLayout(self.sector_strip_host)
        self.sector_strip_lay.setContentsMargins(8, 6, 8, 6)
        self.sector_strip_lay.setSpacing(10)
        strip_scroll.setWidget(self.sector_strip_host)
        lay.addWidget(strip_scroll)
        if _tlogos:
            strip_attr = QLabel(_tlogos.attribution_html())
            strip_attr.setOpenExternalLinks(True)
            lay.addWidget(strip_attr)

        # 图表
        self.canvas = FigureCanvas(Figure(figsize=(9.5, 3.6), facecolor=T.MPL_BG))
        lay.addWidget(self.canvas, 1)

        # 指标明细网格（可滚动）
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(6); self.grid.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        scroll.setStyleSheet(f"border:1px solid {T.BORDER};border-radius:6px;")
        scroll.setMinimumHeight(220)
        lay.addWidget(scroll)

        disc = QLabel("此仪表盘是上述客观指标的机械加权汇总，用于替代主观盯盘，"
                      "<b>不是买卖预测、不构成投资建议</b>。每项指标均可用历史数据自行回测验证。")
        disc.setWordWrap(True)
        disc.setStyleSheet(f"color:{T.TEXT_3};font-size:8pt;border:none;padding:2px;")
        lay.addWidget(disc)

    def _refresh(self):
        self.status.setText("拉取指数与板块ETF…（首次较慢）")
        self.refresh_btn.setEnabled(False)
        QApplication.processEvents()
        try:
            d = compute_market_regime(self.env["fetch_stock_data"])
        except Exception as e:
            self.status.setText(f"计算失败: {e}")
            self.refresh_btn.setEnabled(True)
            return
        self.refresh_btn.setEnabled(True)
        if not d.get("ok"):
            self.status.setText(d.get("msg", "计算失败")); return
        self._data = d
        self._render()
        self._refresh_sector_strip()

    def _refresh_sector_strip(self):
        """板块ETF当日涨跌% —— 独立小抓取，供logo速览条用，不影响主指标计算逻辑。"""
        fetch = self.env["fetch_stock_data"]
        chg = {}
        for sym in SECTOR_ETFS:
            try:
                df = fetch(sym, "5d")
                if df is not None and len(df) >= 2:
                    c = df["Close"].astype(float)
                    chg[sym] = float((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2] * 100.0)
            except Exception as e:
                logger.warning(f"sector strip fetch fail {sym}: {e}")
        self._render_sector_strip(chg)

    def _render_sector_strip(self, chg: dict):
        T = self.T
        while self.sector_strip_lay.count():
            it = self.sector_strip_lay.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self._strip_chip_by_symbol = {}
        for sym in SECTOR_ETFS:
            pct = chg.get(sym)
            color = T.GREEN if (pct is not None and pct >= 0) else (T.RED if pct is not None else T.TEXT_3)
            chip = QFrame()
            chip.setStyleSheet(f"QFrame{{background:{T.BG2};border:1px solid {T.BORDER};border-radius:7px;}}")
            cl = QHBoxLayout(chip); cl.setContentsMargins(6, 3, 8, 3); cl.setSpacing(5)
            icon_lbl = QLabel()
            icon_lbl.setFixedSize(20, 20)
            if _tlogos:
                icon_lbl.setPixmap(_tlogos.get_or_fallback_icon(sym, 20).pixmap(20, 20))
            cl.addWidget(icon_lbl)
            txt = f"<b style='color:{T.TEXT_H};'>{sym}</b><br>"
            txt += (f"<span style='color:{color};font-size:8pt;'>{pct:+.2f}%</span>" if pct is not None
                    else f"<span style='color:{T.TEXT_3};font-size:8pt;'>--</span>")
            lb = QLabel(txt); lb.setStyleSheet("border:none;")
            cl.addWidget(lb)
            self.sector_strip_lay.addWidget(chip)
            self._strip_chip_by_symbol[sym] = icon_lbl
        self.sector_strip_lay.addStretch()
        self._start_strip_logo_loading()

    def _start_strip_logo_loading(self):
        """启动后台线程加载板块ETF的logo，并在线程结束时清理引用。"""
        if not _tlogos:
            return
        old = getattr(self, "_strip_logo_thread", None)
        if old is not None and old.isRunning():
            old.quit()
            old.wait(200)
        symbols = list(self._strip_chip_by_symbol.keys())
        if not symbols:
            return
        th = _tlogos.LogoLoaderThread(symbols, parent=self)
        th.loaded.connect(self._on_strip_logo_loaded)
        th.missing.connect(self._on_strip_logo_missing)
        # 关键修复：线程结束后清空引用，防止 closeEvent 访问已删除对象
        th.finished_all.connect(lambda: setattr(self, '_strip_logo_thread', None))
        self._strip_logo_thread = th
        th.start()

    def _on_strip_logo_loaded(self, symbol, data):
        pm = _tlogos.pixmap_from_bytes(data, 20)
        if pm is None:
            return
        _tlogos.cache_pixmap(symbol, 20, pm)
        lbl = self._strip_chip_by_symbol.get(symbol)
        if lbl is not None:
            lbl.setPixmap(QIcon(pm).pixmap(20, 20))

    def _on_strip_logo_missing(self, symbol):
        pm = _tlogos.make_fallback_icon(symbol, 20)
        _tlogos.cache_pixmap(symbol, 20, pm)
        lbl = self._strip_chip_by_symbol.get(symbol)
        if lbl is not None:
            lbl.setPixmap(QIcon(pm).pixmap(20, 20))

    def closeEvent(self, event):
        """安全关闭线程，防止访问已删除的 LogoLoaderThread 对象。"""
        th = getattr(self, "_strip_logo_thread", None)
        if th is not None:
            try:
                if th.isRunning():
                    th.quit()
                    th.wait(1000)
            except RuntimeError:
                # 对象已被删除，忽略
                pass
            self._strip_logo_thread = None
        super().closeEvent(event)

    def _render(self):
        T = self.T; d = self._data
        cmap = {"green": T.GREEN, "lime": "#84cc16", "gold": T.GOLD,
                "orange": T.ORANGE, "red": T.RED}
        col = cmap.get(d["lcolor"], T.GOLD)
        self.score_lbl.setText(
            f"<span style='color:{col};'>环境分 {d['score']:.0f}/100　·　{d['label']}</span>")
        self.status.setText("数据已更新（指数/ETF，会话内缓存）")

        # 清空网格
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        # 表头
        for j, h in enumerate(["指标", "数值", "得分", "说明"]):
            lb = QLabel(f"<b>{h}</b>")
            lb.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
            self.grid.addWidget(lb, 0, j)
        for i, c in enumerate(d["components"], start=1):
            sc = c["score"]
            scol = (T.GREEN if sc >= 60 else T.RED if sc <= 40 else T.GOLD)
            w = d["weights"].get(c["name"], 0)
            cells = [
                (f"{c['name']}　<span style='color:{T.TEXT_3};font-size:8pt;'>"
                 f"权重{w*100:.0f}%</span>", T.TEXT_H),
                (c["value"], T.TEXT_1),
                (f"<b style='color:{scol};'>{sc:.0f}</b>", T.TEXT_H),
                (c["note"], T.TEXT_2),
            ]
            for j, (txt, color) in enumerate(cells):
                lb = QLabel(txt); lb.setWordWrap(True)
                lb.setStyleSheet(f"color:{color};font-size:9pt;border:none;"
                                 f"border-bottom:1px solid {T.BORDER};padding:3px;")
                self.grid.addWidget(lb, i, j)

        self._draw_chart()

    def _draw_chart(self):
        T = self.T; d = self._data
        get_rc = self.env["_get_mpl_rc"]; mpl_style = self.env["_mpl_style"]
        apply_font = self.env["_apply_font_to_figure"]
        spx, ma50, ma200, vix = d["spx"], d["ma50"], d["ma200"], d["vix"]
        with mpl.rc_context(get_rc()):
            fig = self.canvas.figure; fig.clear()
            fig.patch.set_facecolor(T.MPL_BG)
            gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0],
                                   wspace=0.28, figure=fig)
            ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

            tail = 252
            s = spx.tail(tail); m50 = ma50.tail(tail); m200 = ma200.tail(tail)
            ax1.plot(s.index, s.values, color=T.ACCENT, lw=1.3, label="SPX")
            ax1.plot(m50.index, m50.values, color=T.GOLD, lw=1.0, label="50日")
            ax1.plot(m200.index, m200.values, color=T.PURPLE, lw=1.0, label="200日")
            mpl_style(ax1, title="SPX 与均线（近1年）", ylabel="点位")
            ax1.legend(fontsize=7.5, facecolor=T.BG2, labelcolor=T.TEXT_H,
                       edgecolor=T.BORDER, framealpha=0.85)

            # 板块20日涨幅条形
            ss = d.get("sector_status", [])
            if ss:
                ss_sorted = sorted(ss, key=lambda x: x[2])
                names = [e for e, _, _ in ss_sorted]
                vals = [r for _, _, r in ss_sorted]
                colors = [T.GREEN if v >= 0 else T.RED for v in vals]
                ax2.barh(range(len(names)), vals, color=colors, alpha=0.8)
                ax2.set_yticks(range(len(names)))
                ax2.set_yticklabels(names, fontsize=7, color=T.TEXT_2)
                ax2.axvline(0, color=T.TEXT_3, lw=0.8)
                mpl_style(ax2, title="板块20日涨幅%", xlabel="%")
            apply_font(fig)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
# 一键安装入口
# ══════════════════════════════════════════════════════════════════
def install_market_dashboard(env: dict):
    """env = 主程序 globals()。主窗口按钮行新增「大盘仪表盘」。"""
    required = ["QuantApp", "fetch_stock_data", "T", "_make_app_icon",
                "_get_mpl_rc", "_apply_font_to_figure", "_mpl_style"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_market_dashboard: 主程序缺少 {missing}")

    QuantApp_cls = env["QuantApp"]; T = env["T"]
    _orig_init = QuantApp_cls.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            from PyQt5.QtWidgets import QPushButton, QHBoxLayout
            btn = QPushButton("大盘仪表盘")
            btn.setStyleSheet(btn.styleSheet() +
                              f"color:{T.YELLOW};border-color:{T.YELLOW};")
            anchor = getattr(self, "pair_btn", None) or getattr(self, "coint_btn", None)
            inserted = False
            # 首选：主程序暴露的 self.btn_row
            row = getattr(self, "btn_row", None)
            if isinstance(row, QHBoxLayout):
                idx = row.indexOf(anchor) if anchor is not None else -1
                if idx != -1:
                    row.insertWidget(idx + 1, btn)
                else:
                    row.insertWidget(max(0, row.count() - 1), btn)
                inserted = True
            if not inserted and anchor is not None:
                root = self.layout()
                for i in range(root.count()):
                    r = root.itemAt(i).layout()
                    if isinstance(r, QHBoxLayout) and r.indexOf(anchor) != -1:
                        r.insertWidget(r.indexOf(anchor) + 1, btn)
                        inserted = True
                        break
            if not inserted:
                self.layout().addWidget(btn)
            btn.clicked.connect(lambda: MarketRegimeDashboard(env, parent=self).show())
            self.regime_btn = btn
        except Exception as e:
            logger.warning(f"market dashboard button install: {e}")

    QuantApp_cls.__init__ = _new_init
    logger.info("[market_dashboard] 已安装：主窗口「大盘仪表盘」按钮")
    return MarketRegimeDashboard


# ══════════════════════════════════════════════════════════════════
# 独立自检（offscreen，合成数据，无需联网）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2023-06-01", periods=520)

    def _mk(drift, vol, base):
        return pd.DataFrame(
            {"Close": pd.Series(base * np.cumprod(1 + rng.normal(drift, vol, len(idx))),
                                index=idx)})

    bank = {
        "^GSPC": _mk(0.0005, 0.009, 4200),
        "^VIX": pd.DataFrame({"Close": pd.Series(
            np.clip(15 + rng.normal(0, 1, len(idx)).cumsum() * 0.1, 10, 45), index=idx)}),
        "^VIX3M": pd.DataFrame({"Close": pd.Series(
            np.clip(17 + rng.normal(0, 1, len(idx)).cumsum() * 0.08, 12, 45), index=idx)}),
    }
    for e in set(ATTACK_ETFS + DEFENSE_ETFS + SECTOR_ETFS):
        bank[e] = _mk(rng.normal(0.0004, 0.0003), 0.012, 100)

    def _fetch(sym, period="6mo"):
        return bank.get(sym)

    d = compute_market_regime(_fetch)
    print("=" * 60)
    print(f"环境分 = {d['score']}/100  →  {d['label']}")
    print("=" * 60)
    for c in d["components"]:
        print(f"  {c['name']:<12} 得分{c['score']:>5.0f}  {c['value']}  [{c['note']}]")
    assert d["ok"] and 0 <= d["score"] <= 100
    assert len(d["components"]) >= 7

    # 极端场景：熊市（应低分）
    bank["^GSPC"] = _mk(-0.0015, 0.018, 4200)
    bank["^VIX"] = pd.DataFrame({"Close": pd.Series(
        np.clip(30 + rng.normal(0, 1, len(idx)).cumsum() * 0.15, 20, 70), index=idx)})
    d2 = compute_market_regime(_fetch)
    print(f"\n熊市场景环境分 = {d2['score']}/100 → {d2['label']}（应明显更低）")
    assert d2["score"] < d["score"], "熊市评分未低于基准"

    # 窗口冒烟
    class _T:
        MPL_BG="#ffffff"; MPL_AXES="#fbfdf9"; BG2="#ffffff"; BORDER="#dde5db"
        GOLD="#a9822f"; YELLOW="#d97706"; ACCENT="#1b7a43"; CYAN="#0f9c9c"
        PURPLE="#7c5cb0"; GREEN="#16a34a"; RED="#dc2626"; ORANGE="#d97706"
        TEXT_H="#16241a"; TEXT_1="#33413a"; TEXT_2="#6d7d70"; TEXT_3="#9aab9c"
    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(_T.MPL_AXES); ax.set_title(title, color=_T.TEXT_H, fontsize=9)
    env = {"QuantApp": None, "fetch_stock_data": _fetch, "T": _T, "GLOBAL_STYLE": "",
           "_make_app_icon": lambda: None, "_get_mpl_rc": lambda: {},
           "_apply_font_to_figure": lambda f: None, "_mpl_style": _style}
    app = QApplication(sys.argv)
    dlg = MarketRegimeDashboard(env)
    dlg._refresh()
    print(f"\n窗口冒烟: 子图数={len(dlg.canvas.figure.axes)}(应=2) "
          f"明细行数={dlg.grid.rowCount()} 环境分标签非空={bool(dlg.score_lbl.text())}")
    print("全部自检通过 ✓")
    dlg.close()  # 触发 closeEvent，等后台logo线程收尾，避免退出时 QThread 未结束报错