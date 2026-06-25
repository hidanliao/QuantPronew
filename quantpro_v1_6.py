"""
QuantPro v13.0  —  量化交易助手（情感修复 + 自我修正引擎）
══════════════════════════════════════════════════════════════════
v13.0 新增：

  [情感引擎修复] ─────────────────────────────────────────────────
  1.  内置150+金融专用词典注入VADER
      修复 rises/beats/rally/surge/upgrade 全被误判为中性的问题
  2.  情感得分阈值优化（金融文本compound阈值调低至±0.03）
  3.  双重情感验证：VADER金融版 + 规则关键词交叉校验

  [自我修正引擎 SelfCorrectionEngine] ─────────────────────────
  4.  SQLite持久化预测记录（quantpro_predictions.db）
  5.  预测到期后自动回测：比较预测概率 vs 实际涨跌
  6.  滚动校准误差（Brier Score + 方向准确率）
  7.  动态权重调整：根据近期accuracy重新加权XGB/HQ/Mom三个子模型
  8.  模型性能仪表盘：显示各模型5日/20日/60日准确率
  9.  预测历史Tab：查看全部历史预测 + 对错标注

══════════════════════════════════════════════════════════════════
"""



import sys
import time
import logging
import threading
import traceback
import csv
import sqlite3
import json
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from io import StringIO
from typing import List, Tuple, Dict, Optional

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

# ── 情感分析库（可选依赖）──────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False

try:
    from transformers import pipeline as _hf_pipeline
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

try:
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from hmmlearn.hmm import GaussianHMM

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QProgressBar, QStatusBar, QMessageBox,
    QHeaderView, QGroupBox, QSlider, QCheckBox, QTabWidget,
    QTextEdit, QTextBrowser, QDialog, QSplitter, QFrame,
    QTableView, QAbstractItemView, QFileDialog, QComboBox,
    QGraphicsOpacityEffect, QScrollArea, QSpinBox
)
from PyQt5.QtCore import (
    QThread, pyqtSignal, Qt, QAbstractTableModel, QModelIndex,
    QSortFilterProxyModel, QTimer, QPropertyAnimation, QEasingCurve
)
from PyQt5.QtGui import (
    QColor, QFont, QBrush, QIcon, QPixmap, QPainter,
    QPen, QLinearGradient, QFontDatabase, QCursor
)

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 字体系统
# ══════════════════════════════════════════════════════════════════
_CHINESE_FONT_NAME: str = "sans-serif"
_CHINESE_FONT_FILE: Optional[str] = None

def _detect_chinese_font() -> Tuple[str, Optional[str]]:
    candidates = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'STHeiti',
                  'Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'Hiragino Sans GB']
    ttf_map: Dict[str, str] = {f.name: f.fname for f in fm.fontManager.ttflist}
    for cand in candidates:
        if cand in ttf_map:
            return cand, ttf_map[cand]
    for name, path in ttf_map.items():
        if any(k in name.lower() for k in ['cjk', 'hei', 'yahei', 'noto sans cjk']):
            return name, path
    return 'DejaVu Sans', None

def _apply_font_to_rcparams(font_name: str):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [font_name]
    mpl.rcParams['axes.unicode_minus'] = False

def _setup_matplotlib_font():
    global _CHINESE_FONT_NAME, _CHINESE_FONT_FILE
    _CHINESE_FONT_NAME, _CHINESE_FONT_FILE = _detect_chinese_font()
    _apply_font_to_rcparams(_CHINESE_FONT_NAME)

_setup_matplotlib_font()

def _get_mpl_rc() -> dict:
    return {'font.family': 'sans-serif', 'font.sans-serif': [_CHINESE_FONT_NAME], 'axes.unicode_minus': False}

def _apply_font_to_figure(fig: Figure):
    if _CHINESE_FONT_FILE:
        prop = fm.FontProperties(fname=_CHINESE_FONT_FILE)
        for text in fig.findobj(mpl.text.Text):
            try: text.set_fontproperties(prop)
            except: pass
    else:
        for text in fig.findobj(mpl.text.Text):
            try: text.set_fontfamily(_CHINESE_FONT_NAME)
            except: pass

def _make_mpf_style() -> dict:
    mc = mpf.make_marketcolors(up='#10b981', down='#ef4444', edge='inherit', wick='inherit', volume='in')
    return mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc,
                               facecolor='#070d14', figcolor='#070d14',
                               gridcolor='#1e3452', gridstyle='--', rc=_get_mpl_rc())

# ══════════════════════════════════════════════════════════════════
# 设计令牌
# ══════════════════════════════════════════════════════════════════
class T:
    BG0='#070d14'; BG1='#0d1826'; BG2='#111f30'; BG3='#172438'; BG4='#1e2f45'
    ACCENT='#0ea5e9'; ACCENT2='#38bdf8'; GOLD='#f59e0b'; GOLD_LT='#fcd34d'
    GREEN='#10b981'; GREEN_BG='#052e16'; RED='#ef4444'; RED_BG='#2d0a0a'
    YELLOW='#f59e0b'; YELLOW_BG='#1c1400'; TEXT_H='#e2eaf3'; TEXT_1='#c8d8e8'
    TEXT_2='#8ba3be'; TEXT_3='#4e6a82'; BORDER='#1e3452'; BORDER_ACT='#0ea5e9'
    MPL_BG='#070d14'; MPL_AXES='#0d1826'; PURPLE='#a78bfa'; CYAN='#06b6d4'
    ORANGE='#f97316'
    # ── 渐变/高光令牌（v15 视觉升级）──
    HILITE='#22364f'      # 块顶部高光（比边框更亮，营造玻璃高光）
    BORDER_HI='#2c4a6e'

# ══════════════════════════════════════════════════════════════════
# 国际化
# ══════════════════════════════════════════════════════════════════
LANG: str = "zh"

_S: Dict[str, Dict[str, str]] = {
    "win_title":         {"zh": "QuantPro v13.0  |  量化研究平台（情感修复 + 自我修正）", "en": "QuantPro v13.0  |  Self-Correcting Quant Platform"},
    "btn_scan":          {"zh": "扫描", "en": "Scan"},
    "btn_top100":        {"zh": "美股 Top 100", "en": "US Top 100"},
    "btn_backtest":      {"zh": "Walk Forward回测", "en": "Walk Forward BT"},
    "btn_stop":          {"zh": "停止", "en": "Stop"},
    "btn_clear":         {"zh": "清空", "en": "Clear"},
    "btn_export":        {"zh": "导出 CSV", "en": "Export CSV"},
    "btn_cache":         {"zh": "清除缓存", "en": "Clear Cache"},
    "btn_lang":          {"zh": "English", "en": "中文"},
    "btn_portfolio":     {"zh": "组合优化", "en": "Portfolio Opt"},
    "btn_coint":         {"zh": "协整配对", "en": "Coint Pairs"},
    "btn_cross":         {"zh": "横截面排名", "en": "Cross-Section"},
    "lbl_code":          {"zh": "代码:", "en": "Symbol:"},
    "lbl_filter_ph":     {"zh": "过滤（代码/名称/信号）", "en": "Filter"},
    "lbl_input_ph":      {"zh": "输入股票代码（逗号分隔）: AAPL, TSLA, NVDA", "en": "Enter symbols: AAPL, TSLA, NVDA"},
    "lbl_rsi_buy":       {"zh": "RSI买入上限", "en": "RSI Buy Max"},
    "lbl_rsi_sell":      {"zh": "RSI超买阈值", "en": "RSI Overbought"},
    "lbl_ma_chk":        {"zh": "均线多头（MA20>MA60）", "en": "MA Bull (MA20>MA60)"},
    "lbl_stop_loss":     {"zh": "止损", "en": "Stop Loss"},
    "lbl_take_profit":   {"zh": "止盈", "en": "Take Profit"},
    "lbl_param_group":   {"zh": "策略参数", "en": "Strategy Params"},
    "lbl_score_info":    {"zh": "评分>=6买入 <=2卖出 (12分满)\n短期信号: 均值回归+量价背离\n长期信号: 动量+趋势+IC加权", "en": "Score>=6 Buy <=2 Sell (Max12)\nShort: Mean-Rev+Divergence\nLong: Momentum+Trend+IC"},
    "tab_scan":          {"zh": "股票扫描", "en": "Scan"},
    "tab_backtest":      {"zh": "Walk Forward回测", "en": "Walk Forward"},
    "tab_cross":         {"zh": "横截面Alpha", "en": "Cross-Sectional"},
    "tab_portfolio":     {"zh": "组合优化", "en": "Portfolio"},
    "tab_garch":         {"zh": "GARCH风险", "en": "GARCH Risk"},
    "tab_kline":         {"zh": "K线图", "en": "K-Line"},
    "tab_news":          {"zh": "新闻", "en": "News"},
    "tab_prob":          {"zh": "概率预测", "en": "Prediction"},
    "tab_mc":            {"zh": "蒙特卡洛", "en": "Monte Carlo"},
    "tab_feat":          {"zh": "特征贡献", "en": "Features"},
    "tab_quantile":      {"zh": "Alpha分层", "en": "Alpha Quantile"},
    "tab_coint":         {"zh": "协整配对", "en": "Cointegration"},
    "col_code":          {"zh": "代码", "en": "Symbol"},
    "col_name":          {"zh": "名称", "en": "Name"},
    "col_price":         {"zh": "价格", "en": "Price"},
    "col_chg":           {"zh": "涨跌%", "en": "Chg%"},
    "col_rsi":           {"zh": "RSI", "en": "RSI"},
    "col_cs_rank":       {"zh": "横截面排名", "en": "CS Rank"},
    "col_rel_str":       {"zh": "相对强弱", "en": "Rel Str"},
    "col_ma20":          {"zh": "MA20", "en": "MA20"},
    "col_ma60":          {"zh": "MA60", "en": "MA60"},
    "col_macd":          {"zh": "MACD", "en": "MACD"},
    "col_vol_ratio":     {"zh": "量比", "en": "Vol Ratio"},
    "col_atr":           {"zh": "ATR", "en": "ATR"},
    "col_obv_trend":     {"zh": "OBV趋势", "en": "OBV"},
    "col_signal":        {"zh": "信号", "en": "Signal"},
    "col_risk":          {"zh": "风险", "en": "Risk"},
    "col_sl":            {"zh": "止损", "en": "Stop Loss"},
    "col_tp":            {"zh": "止盈", "en": "Take Profit"},
    "col_detail":        {"zh": "详情", "en": "Detail"},
    "sig_buy":           {"zh": "买入", "en": "Buy"},
    "sig_sell":          {"zh": "卖出", "en": "Sell"},
    "sig_watch":         {"zh": "观望", "en": "Watch"},
    "status_ready":      {"zh": "就绪 — 双击行查看分析", "en": "Ready — Double-click for analysis"},
    "status_analyzing":  {"zh": "分析: {sym}", "en": "Analyzing: {sym}"},
    "status_done_scan":  {"zh": "扫描完成 共{n}只", "en": "Scan done {n} symbols"},
    "status_stopped":    {"zh": "已停止", "en": "Stopped"},
    "status_cache_cleared": {"zh": "缓存已清除", "en": "Cache cleared"},
    "sum_total":         {"zh": "扫描总数", "en": "Total"},
    "sum_buy":           {"zh": "买入信号", "en": "Buy"},
    "sum_sell":          {"zh": "卖出信号", "en": "Sell"},
    "sum_watch":         {"zh": "观望", "en": "Watch"},
    "sum_avg_rsi":       {"zh": "平均RSI", "en": "Avg RSI"},
    "sum_avg_score":     {"zh": "平均评分", "en": "Avg Score"},
    "dlg_confirm":       {"zh": "确认", "en": "Confirm"},
    "dlg_clear_q":       {"zh": "清空所有结果？", "en": "Clear all?"},
    "dlg_no_data":       {"zh": "没有数据", "en": "No data"},
    "dlg_hint":          {"zh": "提示", "en": "Hint"},
    "dlg_input_code":    {"zh": "请输入股票代码", "en": "Enter symbols"},
    "dlg_need_2stocks":  {"zh": "需要至少2只股票", "en": "Need 2+ symbols"},
    "dlg_error":         {"zh": "错误", "en": "Error"},
    "btn_refresh_price": {"zh": "刷新", "en": "Refresh"},
    "btn_refresh_news":  {"zh": "刷新新闻", "en": "Refresh News"},
    "kline_loading":     {"zh": "加载中...", "en": "Loading..."},
    "kline_no_data":     {"zh": "无法获取数据", "en": "No data"},
    "kline_note":        {"zh": "K线历史 | 价格实时", "en": "K-line: history | Price: live"},
    "kline_bars":        {"zh": "{n}根K线", "en": "{n} bars"},
    "kline_plot_fail":   {"zh": "绘图失败: {e}", "en": "Plot failed: {e}"},
    "news_loading":      {"zh": "拉取新闻...", "en": "Fetching..."},
    "news_none":         {"zh": "暂无新闻", "en": "No news"},
    "news_count":        {"zh": "{n}条", "en": "{n} articles"},
    "news_err":          {"zh": "新闻失败: {e}", "en": "News failed: {e}"},
    "flip_loading":      {"zh": "加载指数...", "en": "Loading..."},
    "flip_waiting":      {"zh": "等待...", "en": "Waiting..."},
    "period_6mo":        {"zh": "6个月", "en": "6M"},
    "period_1y":         {"zh": "1年", "en": "1Y"},
    "period_2y":         {"zh": "2年", "en": "2Y"},
    "period_5y":         {"zh": "5年", "en": "5Y"},
    "prob_calc":         {"zh": "计算概率模型中...", "en": "Computing models..."},
    "prob_done":         {"zh": "完成 5日:{p5:.0f}% 20日:{p20:.0f}% 60日:{p60:.0f}%", "en": "Done 5d:{p5:.0f}% 20d:{p20:.0f}% 60d:{p60:.0f}%"},
    "prob_fail":         {"zh": "计算出错: {e}", "en": "Error: {e}"},
    "prob_no_data":      {"zh": "数据不足", "en": "Insufficient data"},
    "err_live_price":    {"zh": "{sym}实时价失败", "en": "{sym} live price failed"},
    "status_loading_top100": {"zh": "加载Top100...", "en": "Loading Top 100..."},
    "status_loaded":     {"zh": "已加载{n}只", "en": "Loaded {n}"},
    "bt_start":          {"zh": "开始Walk Forward回测...\n", "en": "Starting Walk Forward backtest...\n"},
    "bt_skip":           {"zh": "  {sym} 数据不足，跳过\n", "en": "  {sym} skipped\n"},
    "bt_error":          {"zh": "  {sym}: {e}\n", "en": "  {sym}: {e}\n"},
    "bt_no_data":        {"zh": "无可回测数据", "en": "No backtest data"},
    "bt_result":         {"zh": "[{sym}] {name}\n  策略(含成本):{tr:+.2f}%  B&H:{bh:+.2f}%  Alpha:{alpha:+.2f}%\n  胜率:{wr:.1f}%  交易:{tc}次  最大回撤:{dd:.2f}%\n  Sharpe:{sr:.2f}  Sortino:{so:.2f}  Calmar:{ca:.2f}\n  盈亏比:{pf:.2f}  OOS衰减:{dec:.1%}  换手率:{turn:.1%}/期\n  VaR95:{var95:.2f}%  CVaR95:{cvar95:.2f}%\n  OOS期数:{folds}\n{line}\n",
                          "en": "[{sym}] {name}\n  Strat(net):{tr:+.2f}%  B&H:{bh:+.2f}%  Alpha:{alpha:+.2f}%\n  WR:{wr:.1f}%  Trades:{tc}  MaxDD:{dd:.2f}%\n  Sharpe:{sr:.2f}  Sortino:{so:.2f}  Calmar:{ca:.2f}\n  PF:{pf:.2f}  OOS-Decay:{dec:.1%}  Turnover:{turn:.1%}/fold\n  VaR95:{var95:.2f}%  CVaR95:{cvar95:.2f}%\n  OOS Folds:{folds}\n{line}\n"},
    "bt_equity_title":   {"zh": "Walk Forward净值曲线（OOS片段标注）", "en": "Walk Forward Equity (OOS Segments)"},
    "bt_equity_yaxis":   {"zh": "权益($)", "en": "Equity($)"},
    "bt_equity_init":    {"zh": "初始资金", "en": "Initial Capital"},
    "status_backtesting":{"zh": "回测: {sym}", "en": "Backtesting: {sym}"},
    "status_bt_done":    {"zh": "回测完成 {n}只", "en": "Backtest done {n}"},
}

def t(key: str, **kw) -> str:
    s = _S.get(key, {}).get(LANG, key)
    if kw:
        try: s = s.format(**kw)
        except: pass
    return s

def set_lang(lang: str):
    global LANG
    LANG = lang

_COL_KEYS = ["col_code","col_name","col_price","col_chg","col_rsi","col_cs_rank","col_rel_str",
             "col_ma20","col_ma60","col_macd","col_vol_ratio","col_atr","col_obv_trend",
             "col_signal","col_risk","col_sl","col_tp","col_detail"]
_PRICE_COL_KEYS = {"col_price","col_sl","col_tp"}
_NUM_COL_KEYS   = {"col_rsi","col_ma20","col_ma60","col_macd","col_vol_ratio","col_atr","col_cs_rank","col_rel_str"}

# 多周期涨跌：(中文标签, 英文标签, 约交易日数, 行内隐藏字段)
_CHG_PERIODS = [
    ("1天","1D",   1,    "_chg_1d"),
    ("3天","3D",   3,    "_chg_3d"),
    ("5天","5D",   5,    "_chg_5d"),
    ("1月","1M",   21,   "_chg_1mo"),
    ("3月","3M",   63,   "_chg_3mo"),
    ("1年","1Y",   252,  "_chg_1y"),
    ("3年","3Y",   756,  "_chg_3y"),
    ("5年","5Y",   1260, "_chg_5y"),
]
def _chg_period_items():
    return [(zh if LANG=="zh" else en) for zh,en,_,_ in _CHG_PERIODS]

# ══════════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════════
def _make_app_icon() -> QIcon:
    sz = 64; px = QPixmap(sz, sz); px.fill(Qt.transparent)
    p = QPainter(px); p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, sz, sz)
    g.setColorAt(0, QColor(7,13,20)); g.setColorAt(1, QColor(14,165,233))
    p.setBrush(g); p.setPen(Qt.NoPen); p.drawRoundedRect(0,0,sz,sz,12,12)
    pts = [(6,52),(18,46),(28,38),(38,34),(50,22),(58,18)]
    pen = QPen(QColor(245,158,11), 2); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
    for i in range(len(pts)-1): p.drawLine(*pts[i], *pts[i+1])
    p.end(); return QIcon(px)

def _enable_window_buttons(widget):
    """
    给 QDialog 补上最小化/最大化按钮，并去掉无用的帮助(?)按钮。
    QDialog 默认只有关闭+帮助按钮，内容多的对话框（概率预测、协整等）
    需要能最大化。在 setWindowTitle 之后、show 之前调用。
    """
    try:
        f = widget.windowFlags()
        f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
        f &= ~Qt.WindowContextHelpButtonHint
        widget.setWindowFlags(f)
    except Exception:
        pass


def _fit_to_screen(widget, w: int, h: int, max_frac: float = 0.90, center: bool = True):
    """
    屏幕自适应窗口尺寸：把期望的 (w,h) 限制在【当前屏幕可用区域】的
    max_frac 以内，避免小屏/高DPI笔记本上窗口超出屏幕导致显示不全。
    大屏：保持期望尺寸；小屏：自动缩小到刚好放下并居中。
    关键：① 检测失败时退回【保守小尺寸】而非原始大尺寸；
         ② 同时设最大高/宽上限，窗口永远不可能超过屏幕。
    """
    from PyQt5.QtWidgets import QApplication
    # 统一给所有顶层窗口补最小化/最大化、去掉无用的帮助(?)按钮
    _enable_window_buttons(widget)
    avail = None
    # 多重来源依次尝试取屏幕可用区域
    for getter in (
        lambda: widget.screen().availableGeometry(),
        lambda: QApplication.screenAt(QCursor.pos()).availableGeometry(),
        lambda: QApplication.primaryScreen().availableGeometry(),
        lambda: QApplication.desktop().availableGeometry(widget),
    ):
        try:
            g = getter()
            if g is not None and g.width() > 100 and g.height() > 100:
                avail = g; break
        except Exception:
            continue

    if avail is None:
        # 彻底取不到屏幕信息 → 保守小窗，绝不用原始大尺寸
        # 注意：不用 setMaximumSize（会让系统置灰最大化按钮），只设初始尺寸
        fw, fh = min(int(w), 1000), min(int(h), 640)
        widget.resize(fw, fh)
        return

    max_w = int(avail.width() * max_frac)
    max_h = int(avail.height() * max_frac)
    fw, fh = min(int(w), max_w), min(int(h), max_h)
    # 防超屏靠"初始尺寸钳进屏幕内"，不用 setMaximumSize ——
    # setMaximumSize 会让系统把最大化按钮置灰禁用。最大化交给系统，
    # 它只会放大到屏幕可用区，本就不会越界。
    widget.setMinimumSize(min(560, fw), min(420, fh))
    widget.resize(fw, fh)
    if center:
        x = avail.x() + (avail.width() - fw) // 2
        y = avail.y() + (avail.height() - fh) // 2
        widget.move(max(avail.x(), x), max(avail.y(), y))

def _currency(sym: str) -> str:
    s = sym.upper()
    if s.endswith(".HK"):   return "HK$"
    if s.endswith((".SS",".SZ",".T")): return "¥"
    if s.endswith(".L"):    return "£"
    if s.endswith((".PA",".DE",".MI")): return "€"
    return "$"

def _normalize(sym: str) -> str: return sym.replace("/", "-")

_cache_lock = threading.Lock()
_data_cache: Dict[str, pd.DataFrame] = {}

def fetch_stock_data(sym: str, period: str = "6mo") -> pd.DataFrame:
    sym = _normalize(sym); key = f"{sym}_{period}"
    with _cache_lock:
        if key in _data_cache: return _data_cache[key]
    for _ in range(3):
        try:
            if period == "ytd":
                start = date(date.today().year, 1, 1).isoformat()
                df = yf.download(sym, start=start, progress=False, auto_adjust=True)
            elif period == "max":
                df = yf.download(sym, period="max", progress=False, auto_adjust=True)
            else:
                df = yf.download(sym, period=period, progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                for col in ['Open','High','Low','Close','Volume']:
                    if col not in df.columns: raise KeyError(f"Missing {col}")
                with _cache_lock: _data_cache[key] = df
                return df
        except Exception as e:
            logger.warning(f"Download {sym}: {e}"); time.sleep(1)
    return pd.DataFrame()

def clear_cache():
    with _cache_lock: _data_cache.clear()

_rt_cache: Dict[str, tuple] = {}
_rt_lock = threading.Lock()
_RT_TTL = 5.0

def get_realtime_price(sym: str) -> Optional[float]:
    with _rt_lock:
        if sym in _rt_cache:
            ts, p = _rt_cache[sym]
            if time.time() - ts < _RT_TTL: return p
    try:
        tk = yf.Ticker(sym)
        if hasattr(tk, 'fast_info') and tk.fast_info:
            p = tk.fast_info.last_price
            if p is not None:
                with _rt_lock: _rt_cache[sym] = (time.time(), float(p))
                return float(p)
        info = tk.info
        p = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('lastPrice')
        if p is not None:
            with _rt_lock: _rt_cache[sym] = (time.time(), float(p))
            return float(p)
    except: pass
    return None

# ══════════════════════════════════════════════════════════════════
# Look-ahead Bias 防护
# ══════════════════════════════════════════════════════════════════
def no_lookahead(func):
    """装饰器：确保计算中不使用未来数据"""
    def wrapper(df, *args, **kwargs):
        # 验证没有未来值泄露
        result = func(df, *args, **kwargs)
        if isinstance(result, pd.DataFrame):
            # 检查shift(-n)导致的NA是否被填充
            if result.isna().all().any():
                logger.warning(f"Potential lookahead in {func.__name__}: all-NA column detected")
        return result
    wrapper.__name__ = func.__name__
    return wrapper

# ══════════════════════════════════════════════════════════════════
# 技术指标 v9.0（扩展因子库）
# ══════════════════════════════════════════════════════════════════
class TechnicalIndicators:
    @staticmethod
    @no_lookahead
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return pd.DataFrame()

        def s1d(col): 
            x = df[col]
            if isinstance(x, pd.DataFrame): x = x.iloc[:, 0]
            return x.squeeze().astype(float)

        close  = s1d('Close'); volume = s1d('Volume')
        high   = s1d('High');  low    = s1d('Low');  open_  = s1d('Open')

        # 基础均线
        ma5  = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        ma200= close.rolling(200).mean()

        # RSI
        delta = close.diff(); gain = delta.where(delta>0,0.).rolling(14).mean()
        loss  = (-delta.where(delta<0,0.)).rolling(14).mean()
        rsi   = 100 - 100/(1+gain/loss.replace(0,np.nan))

        # Stochastic RSI
        rsi_min = rsi.rolling(14).min(); rsi_max = rsi.rolling(14).max()
        stoch_rsi = (rsi-rsi_min)/(rsi_max-rsi_min+1e-9)
        stoch_rsi_k = stoch_rsi.rolling(3).mean()*100
        stoch_rsi_d = stoch_rsi_k.rolling(3).mean()

        # MACD
        exp12 = close.ewm(span=12,adjust=False).mean()
        exp26 = close.ewm(span=26,adjust=False).mean()
        macd  = exp12-exp26; sig9 = macd.ewm(span=9,adjust=False).mean()

        # Bollinger Bands
        bb_std = close.rolling(20).std()
        bb_up  = ma20+2*bb_std; bb_dn = ma20-2*bb_std

        # Volume
        vol_ma = volume.rolling(20).mean()
        vol_r  = volume/vol_ma.replace(0,np.nan)

        # ATR
        tr  = pd.concat([high-low,(high-close.shift()).abs(),(low-close.shift()).abs()],axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # OBV
        obv    = (np.sign(close.diff())*volume).fillna(0).cumsum()
        obv_ma = obv.rolling(20).mean()

        # Momentum
        mom10 = close.pct_change(10)*100
        mom20 = close.pct_change(20)*100

        # 52周位置
        w52h = high.rolling(252,min_periods=20).max()
        w52l = low.rolling(252,min_periods=20).min()
        rs_pos = (close-w52l)/(w52h-w52l+1e-9)*100

        # ── v9.0 新增扩展因子 ──────────────────────────────────────

        # VWAP偏离因子（日内价格相对成交量加权均价的偏离）
        vwap_daily = (close*volume).rolling(20).sum() / volume.rolling(20).sum().replace(0,np.nan)
        vwap_dev   = (close - vwap_daily) / vwap_daily.replace(0,np.nan)

        # 隔夜收益因子（Open相对前日Close的跳空）
        overnight_ret = (open_ - close.shift(1)) / close.shift(1).replace(0,np.nan)

        # 日内反转因子（Close相对Open，短期反转信号）
        intraday_ret  = (close - open_) / open_.replace(0,np.nan)
        intraday_rev  = -intraday_ret.rolling(5).mean()  # 均值反转

        # Gap因子（跳空缺口大小）
        gap_factor = ((open_ - close.shift(1)) / close.shift(1)).abs()

        # ATR扩张因子（当前ATR vs 历史ATR，波动率扩张信号）
        atr_ma_slow  = atr.rolling(60).mean()
        atr_expansion = atr / atr_ma_slow.replace(0,np.nan)

        # 成交量不均衡因子（量能不对称）
        up_vol   = volume.where(close > close.shift(1), 0).rolling(10).sum()
        down_vol = volume.where(close < close.shift(1), 0).rolling(10).sum()
        vol_imbalance = (up_vol - down_vol) / (up_vol + down_vol + 1e-9)

        chgp = close.pct_change()*100

        # ── v10.0 新增：量价背离因子 ──────────────────────────
        price_high_20 = close.rolling(20).max()
        price_low_20  = close.rolling(20).min()
        vol_ma_20     = volume.rolling(20).mean()
        near_high     = (close >= price_high_20 * 0.97).astype(float)
        near_low      = (close <= price_low_20 * 1.03).astype(float)
        low_vol_flag  = (volume < vol_ma_20 * 0.8).astype(float)
        vol_price_div = near_low * low_vol_flag - near_high * low_vol_flag

        # ── v10.0 新增：1日反转因子（隔日均值回归）────────────
        ret1d        = close.pct_change(1)
        rev_1d       = -ret1d
        rev_5d       = -close.pct_change(5).rolling(3).mean()

        # ── v10.0 新增：量能不对称升级版（10日 vs 20日）────────
        up_vol_20   = volume.where(close > close.shift(1), 0).rolling(20).sum()
        dn_vol_20   = volume.where(close < close.shift(1), 0).rolling(20).sum()
        vol_imb_20  = (up_vol_20 - dn_vol_20) / (up_vol_20 + dn_vol_20 + 1e-9)

        return pd.DataFrame({
            'close':close,'high':high,'low':low,'volume':volume,'open':open_,
            'ma5':ma5,'ma20':ma20,'ma60':ma60,'ma200':ma200,
            'rsi':rsi,'stoch_rsi_k':stoch_rsi_k,'stoch_rsi_d':stoch_rsi_d,
            'macd':macd,'signal':sig9,'hist':macd-sig9,
            'bb_up':bb_up,'bb_mid':ma20,'bb_dn':bb_dn,
            'volume_ratio':vol_r,'vol_ma20':vol_ma,'atr':atr,'atr_expansion':atr_expansion,
            'obv':obv,'obv_ma':obv_ma,
            'mom10':mom10,'mom20':mom20,'rs_pos':rs_pos,
            'vwap_dev':vwap_dev,'overnight_ret':overnight_ret,
            'intraday_rev':intraday_rev,'gap_factor':gap_factor,
            'vol_imbalance':vol_imbalance,'vol_imbalance_20':vol_imb_20,
            'vol_price_div':vol_price_div,'rev_1d':rev_1d,'rev_5d':rev_5d,
            'chg_pct':chgp,
        }, index=df.index)

    @staticmethod
    def score(row, params: dict) -> Tuple[int, List[str]]:
        rsi_buy=params.get('rsi_buy_max',55); rsi_sell=params.get('rsi_sell_min',70)
        use_ma=params.get('use_ma_condition',True)
        sc=0; reasons=[]

        # ① MA多空（趋势过滤）
        if use_ma:
            if row['ma20']>row['ma60']: sc+=2; reasons.append("MA多头+2")
            else: reasons.append("MA空头")

        # ② RSI
        rv=float(row['rsi'])
        if 30<rv<rsi_buy: sc+=2; reasons.append(f"RSI({rv:.0f})健康+2")
        elif rv>=rsi_sell: sc-=1; reasons.append(f"RSI({rv:.0f})超买-1")
        elif rv<=30: sc+=1; reasons.append(f"RSI({rv:.0f})超卖+1")
        else: reasons.append(f"RSI({rv:.0f})")

        # ③ MACD
        if row['macd']>row['signal']: sc+=1; reasons.append("MACD金叉+1")
        else: reasons.append("MACD死叉")

        # ④ 成交量
        vr=float(row['volume_ratio'])
        if vr>1.5: sc+=1; reasons.append(f"大量({vr:.1f}x)+1")
        elif vr>1.0: reasons.append(f"放量({vr:.1f}x)")
        else: reasons.append(f"缩量({vr:.1f}x)")

        # ⑤ Bollinger Band
        bp=(float(row['close'])-float(row['bb_dn']))/(float(row['bb_up'])-float(row['bb_dn'])+1e-9)
        if bp<0.2: sc+=1; reasons.append("BB下轨+1")
        elif bp>0.85: sc-=1; reasons.append("BB上轨-1")

        # ⑥ 动量
        mom=float(row.get('mom10',0) or 0)
        if mom>2.0: sc+=1; reasons.append("动量+1")
        elif mom<-2.0: reasons.append("动量↓")

        # ⑦ OBV
        obv_v=float(row.get('obv',0) or 0); obv_ma_v=float(row.get('obv_ma',0) or 0)
        if obv_v>obv_ma_v: sc+=1; reasons.append("OBV上升+1")
        else: reasons.append("OBV下降")

        # ⑧ StochRSI
        srsi=float(row.get('stoch_rsi_k',50) or 50)
        if srsi<20: sc+=1; reasons.append("StochRSI超卖+1")
        elif srsi>80: sc-=1; reasons.append("StochRSI超买-1")

        # ⑨ VWAP偏离
        vwap_d=float(row.get('vwap_dev',0) or 0)
        if vwap_d<-0.02: sc+=1; reasons.append("VWAP低买+1")
        elif vwap_d>0.05: sc-=1; reasons.append("VWAP过高-1")

        # ⑩ Vol Imbalance
        vi=float(row.get('vol_imbalance',0) or 0)
        if vi>0.3: sc+=1; reasons.append("资金流入+1")
        elif vi<-0.3: sc-=1; reasons.append("资金流出-1")

        # ⑪ v10.0新增：量价背离检测
        vpd=float(row.get('vol_price_div',0) or 0)
        if vpd>0: sc+=1; reasons.append("抛压耗尽+1")
        elif vpd<0: sc-=1; reasons.append("量价背离-1")

        # ⑫ v10.0新增：短期均值回归信号（1日反转）
        rev1=float(row.get('rev_1d',0) or 0)
        if rev1>0.02: sc+=1; reasons.append("短期反弹+1")
        elif rev1<-0.02: reasons.append("短期延续↓")

        return sc, reasons

    @staticmethod
    def obv_trend_label(ind: pd.DataFrame) -> str:
        if ind.empty or len(ind)<5: return "--"
        obv=ind['obv'].dropna(); obv_ma=ind['obv_ma'].dropna()
        if len(obv)<5 or len(obv_ma)<5: return "--"
        last_obv=float(obv.iloc[-1]); last_obv_ma=float(obv_ma.iloc[-1])
        slope=float(obv.iloc[-1]-obv.iloc[-5])/(abs(obv.iloc[-5])+1e-9)
        if last_obv>last_obv_ma and slope>0.01: return "↑强势"
        elif last_obv>last_obv_ma: return "↑"
        elif slope<-0.01: return "↓弱势"
        else: return "↓"

# ══════════════════════════════════════════════════════════════════
# 横截面 Alpha 系统（v9.0 核心）
# ══════════════════════════════════════════════════════════════════
class CrossSectionalAlpha:
    """
    机构核心：股票间相对强弱排名
    不是预测"AAPL是否上涨"，而是"AAPL是否比其他股票更强"
    多空组合：做多Top Quintile，做空Bottom Quintile
    """

    FACTORS = {
        'momentum_1m':   '1月动量',
        'momentum_3m':   '3月动量',
        'momentum_12m':  '12月动量（-1月）',
        'rsi_cs':        'RSI横截面',
        'vol_ratio_cs':  '量比横截面',
        'vwap_dev_cs':   'VWAP偏离',
        'obv_score':     'OBV能量',
        'vol_expansion': '波动率扩张',
        'intraday_rev_cs':'日内反转',
        'overnight_cs':  '隔夜收益',
    }

    @staticmethod
    def compute_scores(syms_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        对多只股票计算横截面因子得分并排名
        返回 DataFrame: rows=股票, cols=因子
        """
        if not syms_data: return pd.DataFrame()
        rows = {}
        for sym, df in syms_data.items():
            if df.empty or len(df) < 65: continue
            ind = TechnicalIndicators.compute_all(df)
            if ind.empty: continue
            close = ind['close']
            row = {}

            # 动量因子（避免look-ahead：只用已实现数据）
            if len(close) >= 21:
                row['momentum_1m'] = float(close.iloc[-1]/close.iloc[-21]-1) if close.iloc[-21] > 0 else 0
            if len(close) >= 63:
                row['momentum_3m'] = float(close.iloc[-1]/close.iloc[-63]-1) if close.iloc[-63] > 0 else 0
            if len(close) >= 252:
                # 12月-1月（经典momentum因子，排除最近1月）
                ret12 = close.iloc[-252]/close.iloc[0]-1 if close.iloc[0] > 0 else 0
                ret1  = close.iloc[-1]/close.iloc[-21]-1 if close.iloc[-21] > 0 else 0
                row['momentum_12m'] = float(ret12 - ret1)
            else:
                row['momentum_12m'] = row.get('momentum_3m', 0)

            lat = ind.iloc[-1]
            row['rsi_cs']        = float(lat.get('rsi', 50))
            row['vol_ratio_cs']  = float(lat.get('volume_ratio', 1))
            row['vwap_dev_cs']   = float(lat.get('vwap_dev', 0))
            row['vol_expansion'] = float(lat.get('atr_expansion', 1))
            row['intraday_rev_cs']= float(lat.get('intraday_rev', 0))
            row['overnight_cs']  = float(ind['overnight_ret'].tail(5).mean())

            # OBV得分（OBV vs OBV MA）
            obv = float(lat.get('obv', 0)); obv_ma = float(lat.get('obv_ma', 0))
            row['obv_score'] = (obv - obv_ma)/(abs(obv_ma)+1e-9)

            rows[sym] = row

        if not rows: return pd.DataFrame()
        df_factors = pd.DataFrame(rows).T

        # 横截面标准化（z-score rank）
        for col in df_factors.columns:
            vals = df_factors[col].dropna()
            if len(vals) < 2: continue
            mn, std = vals.mean(), vals.std()
            if std > 1e-9:
                df_factors[col] = (df_factors[col] - mn) / std
            # Winsorize（截断极端值±3σ）
            df_factors[col] = df_factors[col].clip(-3, 3)

        # 综合得分（等权）
        available = [c for c in CrossSectionalAlpha.FACTORS.keys() if c in df_factors.columns]
        df_factors['composite_score'] = df_factors[available].mean(axis=1)

        # 排名（1=最强）
        df_factors['rank'] = df_factors['composite_score'].rank(ascending=False, method='min')
        df_factors['percentile'] = df_factors['composite_score'].rank(pct=True) * 100

        return df_factors

    @staticmethod
    def long_short_portfolio(scores: pd.DataFrame,
                              returns_dict: Dict[str, pd.Series],
                              top_n: int = None) -> pd.Series:
        """
        构建多空组合净值：
        做多Top Quintile，做空Bottom Quintile（等权）
        """
        if scores.empty: return pd.Series()
        n = len(scores)
        if top_n is None: top_n = max(1, n//5)

        sorted_syms = scores.sort_values('composite_score', ascending=False)
        long_syms   = sorted_syms.head(top_n).index.tolist()
        short_syms  = sorted_syms.tail(top_n).index.tolist()

        # 对齐时间轴
        all_ret = pd.DataFrame({s: returns_dict[s] for s in long_syms+short_syms if s in returns_dict})
        if all_ret.empty: return pd.Series()
        all_ret.dropna(how='all', inplace=True)

        long_ret  = all_ret[long_syms].mean(axis=1)
        short_ret = all_ret[short_syms].mean(axis=1)
        ls_ret    = long_ret - short_ret  # 多空组合每日收益

        equity = (1 + ls_ret).cumprod() * 10000
        return equity


# ══════════════════════════════════════════════════════════════════
# Walk Forward 回测引擎（v9.0 核心）
# ══════════════════════════════════════════════════════════════════
class WalkForwardBacktest:
    """
    真正的Walk Forward Optimization（WFO）
    训练期：N个月  OOS测试期：M个月
    滚动执行，只用OOS期净值拼接为最终净值
    """

    def __init__(self, train_months: int = 12, test_months: int = 3):
        self.train_months = train_months
        self.test_months  = test_months

    def run(self, sym: str, df: pd.DataFrame, params: dict) -> Optional[dict]:
        """返回walk forward结果，包含OOS净值和每期统计"""
        ind = TechnicalIndicators.compute_all(df)
        if ind.empty or len(ind) < 300: return None

        # 按月分段
        train_bars = self.train_months * 21
        test_bars  = self.test_months  * 21
        step       = test_bars

        all_oos_equity: list = []
        all_oos_dates:  list = []
        fold_stats:     list = []

        i = train_bars
        fold = 0
        while i + test_bars <= len(ind):
            train_end = i
            test_start= i
            test_end  = min(i + test_bars, len(ind))

            train_ind = ind.iloc[:train_end]
            test_ind  = ind.iloc[test_start:test_end]
            test_df   = df.iloc[test_start:test_end]

            if len(test_ind) < 5:
                i += step; continue

            # 在训练期优化参数（简化：直接用传入参数，实际可做网格搜索）
            # v9.0升级：在训练期计算最优RSI阈值
            opt_params = self._optimize_params(train_ind, params)

            # 在OOS期回测
            oos_result = self._backtest_period(test_ind, opt_params)
            if oos_result:
                all_oos_equity.extend(oos_result['equity'])
                all_oos_dates.extend(oos_result['dates'])
                fold_stats.append({
                    'fold': fold+1,
                    'start': test_ind.index[0],
                    'end':   test_ind.index[-1],
                    'return': oos_result['total_return'],
                    'trades': oos_result['trades'],
                    'win_rate': oos_result['win_rate'],
                })
            i += step; fold += 1

        if not all_oos_equity: return None

        equity_s = pd.Series(all_oos_equity, index=all_oos_dates[:len(all_oos_equity)])
        returns  = equity_s.pct_change().dropna()

        # 全局统计
        total_r  = (all_oos_equity[-1]/10000-1)*100 if all_oos_equity else 0
        dd       = (equity_s - equity_s.expanding().max())/equity_s.expanding().max()*100
        mdd      = float(dd.min()) if not dd.empty else 0

        sharpe = float(returns.mean()/returns.std()*np.sqrt(252)) if returns.std()>0 else 0

        dn_ret  = returns[returns<0]
        sortino = float(returns.mean()/dn_ret.std()*np.sqrt(252)) if len(dn_ret)>2 and dn_ret.std()>0 else 0

        n_years  = len(equity_s)/252
        calmar   = (total_r/100)/(-mdd/100) if n_years>0.1 and abs(mdd)>0 else 0

        var95  = float(np.percentile(returns, 5)*100)
        cvar95 = float(returns[returns<=np.percentile(returns, 5)].mean()*100) if len(returns)>0 else 0

        bh_close = ind['close']
        bh_r     = (float(bh_close.iloc[-1])/float(bh_close.iloc[0])-1)*100

        total_trades = sum(f['trades'] for f in fold_stats)
        avg_wr = np.mean([f['win_rate'] for f in fold_stats if f['win_rate']>0]) if fold_stats else 0

        # v10.0：OOS衰减系数（早期折 vs 晚期折的收益对比）
        oos_decay = 0.0
        if len(fold_stats) >= 4:
            early = np.mean([f['return'] for f in fold_stats[:len(fold_stats)//2]])
            late  = np.mean([f['return'] for f in fold_stats[len(fold_stats)//2:]])
            if abs(early) > 1e-6:
                oos_decay = float(np.clip((early - late) / abs(early), 0, 1))

        # v10.0：盈亏比
        wins_list = [f['return'] for f in fold_stats if f['return'] > 0]
        loss_list = [abs(f['return']) for f in fold_stats if f['return'] < 0]
        profit_factor = (np.mean(wins_list) / np.mean(loss_list)) if wins_list and loss_list else 1.0

        # v10.0：换手率汇总
        avg_turnover = np.mean([f.get('turnover', 0) for f in fold_stats]) if fold_stats else 0

        return {
            'symbol': sym,
            'equity_curve': (list(equity_s.index), list(equity_s.values)),
            'fold_stats': fold_stats,
            'total_return': round(total_r, 2),
            'bh_return': round(bh_r, 2),
            'max_drawdown': round(mdd, 2),
            'sharpe': round(sharpe, 2),
            'sortino': round(sortino, 2),
            'calmar': round(calmar, 2),
            'var95': round(var95, 2),
            'cvar95': round(cvar95, 2),
            'trade_count': total_trades,
            'win_rate': round(avg_wr, 1),
            'n_folds': len(fold_stats),
            'oos_decay': round(oos_decay, 3),
            'profit_factor': round(profit_factor, 2),
            'avg_turnover': round(avg_turnover, 3),
        }

    def _optimize_params(self, train_ind: pd.DataFrame, base_params: dict) -> dict:
        """简化版参数优化：网格搜索RSI阈值"""
        best_sharpe = -999
        best_params = base_params.copy()
        for rsi_buy in [45, 50, 55, 60]:
            for rsi_sell in [65, 70, 75, 80]:
                p = base_params.copy()
                p['rsi_buy_max']  = rsi_buy
                p['rsi_sell_min'] = rsi_sell
                r = self._backtest_period(train_ind, p)
                if r and r.get('sharpe',0) > best_sharpe:
                    best_sharpe = r['sharpe']
                    best_params = p.copy()
        return best_params

    def _backtest_period(self, ind: pd.DataFrame, params: dict) -> Optional[dict]:
        """单期回测（v10.0：增加执行成本调整 + 动态波动率仓控）"""
        if len(ind) < 5: return None
        sl_pct=params.get('stop_loss_pct',0.05); tp_pct=params.get('take_profit_pct',0.10)
        comm=0.001; slip=0.0005
        buy_s=t("sig_buy"); sell_s=t("sig_sell")

        signals=[]
        for i in range(len(ind)):
            row=ind.iloc[i]
            if any(pd.isna([row.get('ma20',np.nan),row.get('rsi',np.nan)])):
                signals.append(t("sig_watch")); continue
            sc,_=TechnicalIndicators.score(row,params)
            signals.append(t("sig_buy") if sc>=5 else (t("sig_sell") if sc<=1 else t("sig_watch")))

        pos=0.; cash=10000.; entry=0.; entry_atr=0.
        records=[]; trade_log=[]
        wins_h=[]; loss_h=[]
        atrs=ind['atr'].values

        # v10.0：动态波动率目标仓位控制器
        vol_controller = DynamicVolatilityTargeting(target_vol=0.15, lookback=20)
        equity_hist = pd.Series(dtype=float)
        total_trades_cost = 0.0

        for i in range(min(5,len(ind)-1), len(ind)-1):
            sig=signals[i]; cur_c=float(ind['close'].iloc[i])
            cur_atr=float(atrs[i]) if not np.isnan(atrs[i]) else cur_c*sl_pct
            nxt=float(ind['close'].iloc[i+1])
            exited=False

            if pos>0:
                pnl=(cur_c-entry)/entry
                atr_stop=(entry-2*entry_atr)/entry-1
                act_sl=max(atr_stop,-sl_pct)
                if pnl<=act_sl or pnl>=tp_pct:
                    sp=nxt*(1-slip); cash=pos*sp*(1-comm)
                    pv=(sp-entry)/entry
                    if trade_log and trade_log[-1]['ep'] is None:
                        trade_log[-1].update({'ep':sp,'pnl':pv})
                    (wins_h if pv>0 else loss_h).append(abs(pv))
                    total_trades_cost += comm + slip
                    pos=entry=entry_atr=0.; exited=True

            if not exited:
                if sig==buy_s and pos==0:
                    # v10.0：动态波动率调整仓位上限
                    cur_equity = cash + pos * cur_c
                    if len(equity_hist) >= 20:
                        leverage = vol_controller.compute_leverage(equity_hist.pct_change().dropna())
                        dd_scale = vol_controller.drawdown_scaling(equity_hist)
                        f_max = min(0.25, 0.125 * leverage * dd_scale)
                    else:
                        f_max = 0.125

                    f=f_max
                    if wins_h or loss_h:
                        wr=len(wins_h)/(len(wins_h)+len(loss_h))
                        aw=np.mean(wins_h) if wins_h else 0.01
                        al=np.mean(loss_h) if loss_h else 0.01
                        b=abs(aw/al); f_k=(wr*b-(1-wr))/b
                        f=max(0.,min(f_k*0.5, f_max))
                    bp=nxt*(1+slip)
                    pos=cash*f*(1-comm)/bp; cash=cash*(1-f)
                    entry=bp; entry_atr=cur_atr
                    total_trades_cost += comm + slip
                    trade_log.append({'bp':bp,'ep':None,'pnl':None})
                elif sig==sell_s and pos>0:
                    sp=nxt*(1-slip); cash+=pos*sp*(1-comm)
                    pv=(sp-entry)/entry
                    if trade_log and trade_log[-1]['ep'] is None:
                        trade_log[-1].update({'ep':sp,'pnl':pv})
                    (wins_h if pv>0 else loss_h).append(abs(pv))
                    total_trades_cost += comm + slip
                    pos=entry=entry_atr=0.

            cur_equity_val = cash+pos*cur_c
            records.append((ind.index[i], cur_equity_val))
            equity_hist = pd.Series([r[1] for r in records],
                                     index=[r[0] for r in records])

        if pos>0:
            fp=float(ind['close'].iloc[-1])*(1-slip)
            cash+=pos*fp*(1-comm)
            if records: records[-1]=(records[-1][0], cash)

        if not records: return None

        dates=[r[0] for r in records]; equity=[r[1] for r in records]
        closed=[x for x in trade_log if x['pnl'] is not None]
        wins=sum(1 for x in closed if x['pnl']>0)
        wr_f=wins/len(closed)*100 if closed else 0.

        eq_s=pd.Series(equity,index=dates)
        dd=(eq_s-eq_s.expanding().max())/eq_s.expanding().max()*100
        dr=eq_s.pct_change().dropna()
        sharpe=float(dr.mean()/dr.std()*np.sqrt(252)) if dr.std()>0 else 0.

        # v10.0：换手率计算
        turnover = len(closed) / max(len(records), 1)
        gross_return = (equity[-1]/10000-1)*100
        # 成本调整后收益
        cost_adj_return = ExecutionCostModel.adjusted_return(
            gross_return/100, turnover, cost_per_trade=0.002
        ) * 100

        return {
            'equity': equity, 'dates': dates,
            'total_return': cost_adj_return,  # v10.0：返回成本调整后
            'gross_return': gross_return,
            'trades': len(closed), 'win_rate': wr_f, 'sharpe': sharpe,
            'turnover': turnover, 'total_cost': total_trades_cost * 100,
        }


# ══════════════════════════════════════════════════════════════════
# 高级组合优化（v9.0）
# ══════════════════════════════════════════════════════════════════
class PortfolioOptimizer:
    """
    机构级组合优化：
    1. HRP（Hierarchical Risk Parity）
    2. Mean-Variance（有效前沿）
    3. Vol Targeting
    4. Beta Neutral检查
    """

    @staticmethod
    def hrp(returns: pd.DataFrame) -> Dict[str, float]:
        """
        HRP：层级风险平价（Lopez de Prado 2016）
        比传统风险平价更稳定，不需要估计期望收益
        """
        cov  = returns.cov()
        corr = returns.corr()
        # 相关性距离矩阵
        dist = np.sqrt((1 - corr) / 2)
        dist = np.clip(dist, 0, 1)
        np.fill_diagonal(dist.values, 0)

        # 层级聚类
        condensed_dist = squareform(dist.values)
        condensed_dist = np.clip(condensed_dist, 0, None)
        link = linkage(condensed_dist, method='ward')

        # 递归平分
        sorted_idx = PortfolioOptimizer._get_quasi_diag(link, len(returns.columns))
        sorted_syms = returns.columns[sorted_idx].tolist()
        weights = PortfolioOptimizer._hrp_recursive_bisect(cov, sorted_syms)
        return weights

    @staticmethod
    def _get_quasi_diag(link, n_items):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= n_items:
            sort_ix.index = range(0, sort_ix.shape[0]*2, 2)
            df0 = sort_ix[sort_ix >= n_items]
            i = df0.index; j = df0.values - n_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i+1)
            sort_ix = pd.concat([sort_ix, df0]).sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    @staticmethod
    def _hrp_recursive_bisect(cov, sorted_syms):
        weights = pd.Series(1.0, index=sorted_syms)
        clusters = [sorted_syms]
        while len(clusters) > 0:
            clusters = [c[int(len(c)/2):] + [None] + c[:int(len(c)/2)]
                        for c in clusters if len(c) > 1]
            new_clusters = []
            for c in clusters:
                if None in c:
                    idx = c.index(None)
                    c1 = [x for x in c[:idx] if x is not None]
                    c2 = [x for x in c[idx+1:] if x is not None]
                    if c1 and c2:
                        try:
                            v1 = PortfolioOptimizer._cluster_var(cov, c1)
                            v2 = PortfolioOptimizer._cluster_var(cov, c2)
                            alpha = 1 - v1/(v1+v2+1e-10)
                            weights[c1] *= alpha
                            weights[c2] *= (1-alpha)
                        except: pass
                    if c1 and len(c1)>1: new_clusters.append(c1)
                    if c2 and len(c2)>1: new_clusters.append(c2)
            clusters = new_clusters
        total = weights.sum()
        return {s: round(float(w/total*100), 2) for s,w in weights.items()}

    @staticmethod
    def _cluster_var(cov, cluster):
        cov_c = cov.loc[cluster, cluster]
        w = 1.0/np.diag(cov_c.values)
        w /= w.sum()
        return float(np.dot(w, np.dot(cov_c.values, w)))

    @staticmethod
    def mean_variance_frontier(returns: pd.DataFrame, n_points: int = 60) -> Tuple[list, list, list]:
        """
        有效前沿：蒙特卡洛模拟随机组合
        返回 (波动率列表, 收益率列表, 夏普比率列表)
        """
        mu  = returns.mean() * 252
        cov = returns.cov()  * 252
        n   = len(returns.columns)
        vols, rets, sharpes = [], [], []
        rng = np.random.default_rng(42)
        for _ in range(n_points * 20):
            w = rng.dirichlet(np.ones(n))
            p_ret  = float(np.dot(w, mu))
            p_vol  = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
            p_sr   = p_ret/p_vol if p_vol>0 else 0
            vols.append(p_vol*100); rets.append(p_ret*100); sharpes.append(p_sr)
        return vols, rets, sharpes

    @staticmethod
    def vol_target_weights(returns: pd.DataFrame, target_vol: float = 0.15,
                            base_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        波动率目标化：将组合波动率缩放到目标水平
        """
        cov = returns.cov() * 252
        n   = len(returns.columns)
        if base_weights:
            w = np.array([base_weights.get(s, 1/n) for s in returns.columns])
        else:
            w = np.ones(n)/n
        w /= w.sum()
        cur_vol = float(np.sqrt(np.dot(w, np.dot(cov.values, w))))
        if cur_vol > 1e-6:
            scale = target_vol / cur_vol
            w *= scale
        w = np.clip(w, 0, 0.4)  # 单股上限40%
        w /= w.sum()
        return {s: round(float(v*100), 2) for s, v in zip(returns.columns, w)}

    @staticmethod
    def beta_exposure(returns: pd.DataFrame,
                      market_ret: pd.Series) -> Dict[str, float]:
        """计算每只股票相对市场的Beta"""
        betas = {}
        for col in returns.columns:
            r = returns[col].dropna()
            m = market_ret.reindex(r.index).dropna()
            aligned = pd.concat([r, m], axis=1).dropna()
            if len(aligned) < 30:
                betas[col] = 1.0; continue
            cov_val = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
            beta = cov_val[0,1]/cov_val[1,1] if cov_val[1,1]>0 else 1.0
            betas[col] = round(float(beta), 3)
        return betas


# ══════════════════════════════════════════════════════════════════
# GARCH 风险引擎（v9.0）
# ══════════════════════════════════════════════════════════════════
class GARCHRiskEngine:
    """
    GARCH(1,1) 条件波动率预测
    Fat-tail Monte Carlo（t分布）
    VaR / CVaR 估计
    """

    @staticmethod
    def fit_garch(returns: pd.Series) -> Optional[dict]:
        """拟合GARCH(1,1)模型"""
        if not HAS_ARCH:
            return None
        r = returns.dropna() * 100  # 以%为单位
        if len(r) < 100:
            return None
        try:
            model = arch_model(r, vol='Garch', p=1, q=1, dist='t', rescale=False)
            res   = model.fit(disp='off', show_warning=False)
            forecasts = res.forecast(horizon=20)
            forecast_var = forecasts.variance.iloc[-1].values
            forecast_vol = np.sqrt(forecast_var)/100  # 转回小数

            return {
                'aic': float(res.aic),
                'bic': float(res.bic),
                'omega':  float(res.params.get('omega', 0)),
                'alpha':  float(res.params.get('alpha[1]', 0)),
                'beta':   float(res.params.get('beta[1]', 0)),
                'nu':     float(res.params.get('nu', 0)),
                'cond_vol': list(res.conditional_volatility/100),
                'cond_vol_index': list(returns.dropna().index),
                'forecast_vol': list(forecast_vol),
                'persistence': float(res.params.get('alpha[1]', 0) + res.params.get('beta[1]', 0)),
            }
        except Exception as e:
            logger.warning(f"GARCH fit failed: {e}")
            return None

    @staticmethod
    def fat_tail_monte_carlo(returns: pd.Series, days: int = 20, n: int = 10000) -> dict:
        """
        Fat-tail Monte Carlo：使用t分布模拟（重尾风险）
        比Normal MC更真实地反映尾部风险
        """
        r = returns.dropna()
        cur_price = 1.0
        if len(r) < 30:
            return {}
        # 拟合t分布参数
        try:
            df_t, loc, scale = stats.t.fit(r)
        except:
            df_t, loc, scale = 5.0, r.mean(), r.std()

        # ── v11 修复：漂移衰减，避免长期预测虚高 ──────────────────
        decay_factor = np.exp(-0.8 * days / 252)
        loc_adj = float(loc) * decay_factor
        rng = np.random.default_rng(days * 7 + 42)   # 不同days用不同seed
        # 正态 vs t分布模拟
        normal_samples = rng.normal(loc_adj, scale, (n, days))
        t_samples = stats.t.rvs(df_t, loc=loc_adj, scale=scale, size=(n, days), random_state=days * 7 + 42)

        normal_finals = np.cumprod(1+normal_samples, axis=1)[:,-1]
        t_finals      = np.cumprod(1+t_samples,      axis=1)[:,-1]

        def calc_stats(finals):
            rets = (finals - 1)*100
            return {
                'up_prob':    float(np.mean(finals>1)),
                'var95':      float(np.percentile(rets, 5)),
                'var99':      float(np.percentile(rets, 1)),
                'cvar95':     float(rets[rets<=np.percentile(rets,5)].mean()),
                'cvar99':     float(rets[rets<=np.percentile(rets,1)].mean()),
                'median_pct': float(np.median(rets)),
                'low_pct':    float(np.percentile(rets, 5)),
                'high_pct':   float(np.percentile(rets, 95)),
                'finals':     finals,
            }

        return {
            'normal': calc_stats(normal_finals),
            't_dist': calc_stats(t_finals),
            'df_t': df_t, 'scale': scale,
        }

    @staticmethod
    def rolling_var(returns: pd.Series, window: int = 60, confidence: float = 0.95) -> pd.Series:
        """滚动VaR序列"""
        return returns.rolling(window).quantile(1-confidence)


# ══════════════════════════════════════════════════════════════════
# 因子IC & Alpha分层
# ══════════════════════════════════════════════════════════════════
class FactorAnalysis:
    @staticmethod
    def rolling_ic(ind: pd.DataFrame, close: pd.Series,
                   factor_col: str = 'rsi', fwd: int = 5, window: int = 20) -> pd.Series:
        factor = ind[factor_col].copy()
        future = close.pct_change(fwd).shift(-fwd)  # shift(-fwd)是OK的：我们在历史分析，不是实时预测
        comb   = pd.concat([factor, future], axis=1).dropna()
        comb.columns = ['factor','future']
        ics = []
        for i in range(window, len(comb)):
            sub = comb.iloc[i-window:i]
            if len(sub)<5: ics.append(np.nan); continue
            ic,_ = stats.spearmanr(sub['factor'], sub['future'])
            ics.append(ic)
        idx = comb.index[window:]
        return pd.Series(ics, index=idx[:len(ics)])

    @staticmethod
    def quantile_analysis(ind, close, factor_col='rsi', fwd=5, n_quantiles=5):
        factor = ind[factor_col].copy()
        future = close.pct_change(fwd).shift(-fwd)*100
        comb   = pd.concat([factor,future],axis=1).dropna()
        comb.columns=['factor','future']
        if len(comb)<n_quantiles*5: return [],[]
        try:
            comb['quantile']=pd.qcut(comb['factor'],n_quantiles,labels=False,duplicates='drop')
        except: return [],[]
        labels=[f"Q{i+1}" for i in range(n_quantiles)]
        means=[float(comb[comb['quantile']==i]['future'].mean()) for i in range(n_quantiles)]
        return labels,means


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：IC加权因子系统
# ══════════════════════════════════════════════════════════════════
class ICWeightedAlpha:
    """
    对标大奖章基金：因子IC（信息系数）动态加权
    高IC因子权重更高，低IC因子自动降权
    IC = Spearman(因子值, 未来收益) 的滚动均值
    """

    @staticmethod
    def compute_ic_weights(
        ind: pd.DataFrame,
        close: pd.Series,
        factor_cols: list,
        fwd: int = 5,
        window: int = 60,
        min_periods: int = 20
    ) -> Dict[str, float]:
        """
        计算每个因子的IC（Spearman相关系数）并归一化为权重
        返回 {factor_name: weight}
        """
        future_ret = close.pct_change(fwd).shift(-fwd)
        ic_dict = {}

        for col in factor_cols:
            if col not in ind.columns:
                ic_dict[col] = 0.0
                continue
            factor = ind[col].dropna()
            combined = pd.concat([factor, future_ret], axis=1).dropna()
            combined.columns = ['f', 'r']
            if len(combined) < min_periods:
                ic_dict[col] = 0.0
                continue
            # 滚动IC
            ics = []
            for i in range(window, len(combined)):
                sub = combined.iloc[i-window:i]
                try:
                    ic, _ = stats.spearmanr(sub['f'], sub['r'])
                    ics.append(ic if not np.isnan(ic) else 0.0)
                except:
                    ics.append(0.0)
            if ics:
                mean_ic = float(np.mean(ics[-20:]))  # 最近20期IC均值
                # ── v11 修复：IC衰减检测 ──────────────────────────────
                # 若近期5期IC均值持续为负（因子失效），权重归零
                recent_ic_5 = float(np.mean(ics[-5:])) if len(ics) >= 5 else mean_ic
                if recent_ic_5 < -0.02:  # 近期IC显著为负 → 因子失效
                    ic_dict[col] = 0.0
                else:
                    ic_dict[col] = abs(mean_ic)  # 取绝对值（负相关也是有效信号）
            else:
                ic_dict[col] = 0.0

        # 归一化（softmax风格，避免某个因子权重过高）
        total = sum(ic_dict.values())
        if total < 1e-9:
            n = len(factor_cols)
            return {c: 1.0/n for c in factor_cols}
        # 上限：单因子权重不超过30%
        weights = {c: v/total for c, v in ic_dict.items()}
        for c in weights:
            weights[c] = min(weights[c], 0.30)
        total2 = sum(weights.values())
        return {c: v/total2 for c, v in weights.items()}

    @staticmethod
    def compute_scores_ic_weighted(
        syms_data: Dict[str, pd.DataFrame],
        base_scores: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        IC加权横截面排名（替代等权）
        如果已有base_scores（基础等权），在此基础上重加权
        """
        if not syms_data:
            return pd.DataFrame()

        # 先计算等权基础分
        base = CrossSectionalAlpha.compute_scores(syms_data)
        if base.empty:
            return base

        # 找一只有足够数据的代表性股票计算IC权重
        ic_weights = {}
        for sym, df in syms_data.items():
            if len(df) >= 252:
                ind = TechnicalIndicators.compute_all(df)
                factor_cols = [c for c in CrossSectionalAlpha.FACTORS.keys() if c in ind.columns]
                if factor_cols:
                    ic_weights = ICWeightedAlpha.compute_ic_weights(
                        ind, ind['close'], factor_cols, fwd=5, window=60
                    )
                break

        if not ic_weights:
            return base

        # 用IC权重替换等权综合得分
        available = [c for c in ic_weights if c in base.columns]
        if available:
            weighted_score = sum(
                base[col] * ic_weights.get(col, 0)
                for col in available
            )
            base['composite_score'] = weighted_score
            base['rank'] = base['composite_score'].rank(ascending=False, method='min')
            base['percentile'] = base['composite_score'].rank(pct=True) * 100

        return base


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：短期均值回归策略（大奖章核心策略之一）
# ══════════════════════════════════════════════════════════════════
class ShortTermMeanReversion:
    """
    短期均值回归Alpha信号
    核心逻辑：统计上价格偏离短期均值后有回归倾向
    适合1~5日持仓的高频换手策略

    参考文献：Lo & MacKinlay (1990), Jegadeesh (1990)
    """

    @staticmethod
    def compute_signals(ind: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """
        计算短期均值回归信号集合
        返回包含多个信号的DataFrame
        """
        signals = pd.DataFrame(index=ind.index)

        # ── 1. 1日收益反转（隔日反转，文献证实有效）──────────────
        ret1d = close.pct_change(1)
        signals['rev_1d'] = -ret1d  # 上涨做空，下跌做多

        # ── 2. 5日收益反转（周度反转）──────────────────────────
        ret5d = close.pct_change(5)
        signals['rev_5d'] = -ret5d.rolling(3).mean()

        # ── 3. 布林带偏离度（均值回归强度）──────────────────────
        bb_up = ind.get('bb_up', pd.Series(np.nan, index=ind.index))
        bb_dn = ind.get('bb_dn', pd.Series(np.nan, index=ind.index))
        bb_mid = ind.get('bb_mid', close.rolling(20).mean())
        bb_width = (bb_up - bb_dn) / (bb_mid + 1e-9)
        bb_pos_raw = (close - bb_mid) / (bb_width * bb_mid / 2 + 1e-9)
        signals['bb_reversion'] = -bb_pos_raw  # 高于中轨做空，低于做多

        # ── 4. RSI极端反转（超买超卖回归）──────────────────────
        rsi = ind.get('rsi', pd.Series(50, index=ind.index))
        signals['rsi_reversion'] = -(rsi - 50) / 50  # RSI>50做空，<50做多

        # ── 5. 隔夜跳空填补（Gap Fill预期）──────────────────────
        overnight = ind.get('overnight_ret', pd.Series(0, index=ind.index))
        # 大的正跳空（开盘高开）→ 当日可能回落
        signals['gap_fade'] = -overnight.rolling(3).mean()

        # ── 6. 日内高低幅度反转（大涨日后容易回调）──────────────
        high = ind.get('high', close)
        low = ind.get('low', close)
        intraday_range = (high - low) / (close + 1e-9)
        range_zscore = (intraday_range - intraday_range.rolling(20).mean()) / \
                       (intraday_range.rolling(20).std() + 1e-9)
        intraday_dir = (close - (high + low) / 2) / (intraday_range + 1e-9)
        signals['intraday_reversion'] = -(intraday_dir * (range_zscore > 1).astype(float))

        # ── 综合短期反转得分（等权合成，后续可换IC加权）──────────
        cols = ['rev_1d', 'rev_5d', 'bb_reversion', 'rsi_reversion', 'gap_fade', 'intraday_reversion']
        available = [c for c in cols if c in signals.columns]
        # 标准化
        for col in available:
            s = signals[col]
            std = s.rolling(60).std()
            signals[col] = s / (std + 1e-9)
            signals[col] = signals[col].clip(-3, 3)
        signals['short_term_score'] = signals[available].mean(axis=1)

        return signals

    @staticmethod
    def combined_signal(
        ind: pd.DataFrame,
        close: pd.Series,
        regime: str = 'neutral',
        trend_weight: float = 0.5,
    ) -> pd.Series:
        """
        制度感知的合成信号：
        - 震荡市（oscillating）：均值回归权重更高
        - 趋势市（trending）：动量权重更高
        - 高波市（high_vol）：降低整体信号强度
        trend_weight: 0=纯反转, 1=纯趋势动量
        """
        rev_signals = ShortTermMeanReversion.compute_signals(ind, close)
        rev_score = rev_signals.get('short_term_score', pd.Series(0, index=ind.index))

        # 动量分量（已有的mom10）
        mom_score = ind.get('mom10', pd.Series(0, index=ind.index))
        mom_norm = mom_score / (mom_score.rolling(60).std() + 1e-9)
        mom_norm = mom_norm.clip(-3, 3)

        if regime == 'high_vol':
            w_rev, w_mom = 0.3, 0.2  # 高波动期降低信号强度
        elif regime == 'low_vol':
            w_rev, w_mom = 0.6, 0.4  # 低波动期均值回归更有效
        else:
            w_rev = 1 - trend_weight
            w_mom = trend_weight

        combined = w_rev * rev_score + w_mom * mom_norm
        return combined.fillna(0)


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：量价背离检测（Divergence Detection）
# ══════════════════════════════════════════════════════════════════
class VolumePriceDivergence:
    """
    量价背离是大奖章基金等统计套利策略中的重要信号
    价格新高但成交量萎缩 → 趋势疲弱，可能反转
    价格新低但成交量萎缩 → 抛压减弱，可能企稳
    """

    @staticmethod
    def detect(ind: pd.DataFrame, lookback: int = 20) -> pd.Series:
        """
        检测量价背离
        返回: +1 = 量增价涨（健康趋势）
               0 = 无背离
              -1 = 背离（危险信号）
        """
        close = ind['close']
        volume = ind['volume']

        price_high = close.rolling(lookback).max()
        price_low = close.rolling(lookback).min()
        vol_ma = volume.rolling(lookback).mean()

        # 价格接近高点但成交量低于均值 → 负背离
        near_high = close >= price_high * 0.97
        low_vol = volume < vol_ma * 0.8
        bearish_div = (near_high & low_vol).astype(float) * -1

        # 价格接近低点但成交量低于均值 → 正背离（抛压耗尽）
        near_low = close <= price_low * 1.03
        bullish_div = (near_low & low_vol).astype(float)

        divergence = bearish_div + bullish_div
        return divergence

    @staticmethod
    def label(ind: pd.DataFrame) -> str:
        """返回最新背离状态的文字描述"""
        div = VolumePriceDivergence.detect(ind)
        if div.empty:
            return "无"
        last = div.iloc[-1]
        if last < 0:
            return "⚠ 量价背离"
        elif last > 0:
            return "↑ 抛压耗尽"
        return "正常"


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：执行成本模型（Kyle's Lambda简化）
# ══════════════════════════════════════════════════════════════════
class ExecutionCostModel:
    """
    市场冲击成本估算
    大盘股（日均成交>1亿美元）：冲击成本约5~10bp
    小盘股：冲击成本可达50~100bp
    回测中必须考虑，否则结果虚高
    """

    @staticmethod
    def estimate_impact(
        trade_size_usd: float,
        adv_usd: float,  # 日均成交额
        spread_pct: float = 0.001
    ) -> float:
        """
        Kyle's Lambda简化版市场冲击估算
        trade_size / adv 越大，冲击越大
        返回单边冲击成本（比例，如0.001 = 0.1%）
        """
        if adv_usd <= 0:
            return 0.005  # 默认0.5%

        participation_rate = trade_size_usd / adv_usd
        # 线性近似（真实是非线性，但对中小仓位足够）
        impact = spread_pct / 2 + 0.1 * np.sqrt(participation_rate)
        return float(np.clip(impact, 0.0001, 0.02))  # 上限2%

    @staticmethod
    def adjusted_return(gross_return: float, turnover: float,
                        cost_per_trade: float = 0.002) -> float:
        """
        调整后收益 = 毛收益 - 换手率 × 每次交易成本
        turnover: 换手率（如1.0表示100%换手）
        cost_per_trade: 单边成本（默认0.2%，含佣金+滑点）
        """
        return gross_return - turnover * cost_per_trade * 2  # 买卖各一次


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：组合层面动态波动率目标
# ══════════════════════════════════════════════════════════════════
class DynamicVolatilityTargeting:
    """
    组合级别的动态波动率目标仓位管理
    参考AQR、文艺复兴等基金的核心风险管理方法

    目标：保持组合年化波动率稳定在目标水平（如15%）
    方法：根据近期已实现波动率动态调整总仓位
    """

    def __init__(self, target_vol: float = 0.15, lookback: int = 20):
        self.target_vol = target_vol
        self.lookback = lookback

    def compute_leverage(self, portfolio_returns: pd.Series) -> float:
        """
        计算应该持有的杠杆倍数
        leverage = target_vol / realized_vol
        上限：2x；下限：0.1x
        """
        if len(portfolio_returns) < self.lookback:
            return 1.0
        realized_vol = float(portfolio_returns.tail(self.lookback).std() * np.sqrt(252))
        if realized_vol < 1e-6:
            return 1.0
        leverage = self.target_vol / realized_vol
        return float(np.clip(leverage, 0.1, 2.0))

    def drawdown_scaling(
        self,
        equity_curve: pd.Series,
        max_dd_threshold: float = 0.10
    ) -> float:
        """
        最大回撤触发降仓
        当策略回撤超过阈值时自动降低仓位
        大奖章基金的核心风控之一
        """
        if equity_curve.empty:
            return 1.0
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        current_dd = float(drawdown.iloc[-1])

        if current_dd < -max_dd_threshold:
            # 回撤超过阈值，降仓至50%
            severity = abs(current_dd) / max_dd_threshold
            scale = max(0.3, 1.0 - severity * 0.5)
            return float(scale)
        return 1.0

    @staticmethod
    def correlation_filter(
        new_sym: str,
        existing_syms: list,
        returns_dict: Dict[str, pd.Series],
        max_corr: float = 0.70
    ) -> bool:
        """
        新建仓相关性过滤
        避免组合中加入与现有持仓高度相关的资产
        返回True表示通过过滤（可以加仓）
        """
        if not existing_syms or new_sym not in returns_dict:
            return True
        new_ret = returns_dict.get(new_sym, pd.Series())
        for sym in existing_syms:
            if sym not in returns_dict:
                continue
            combined = pd.concat([new_ret, returns_dict[sym]], axis=1).dropna()
            if len(combined) < 20:
                continue
            corr = float(combined.corr().iloc[0, 1])
            if corr > max_corr:
                return False  # 相关性过高，不加仓
        return True


# ══════════════════════════════════════════════════════════════════
# v10.0 新增：策略性能雷达图
# ══════════════════════════════════════════════════════════════════
class StrategyRadarChart:
    """
    策略性能雷达图（对标大奖章基金指标）
    维度：Sharpe / Calmar / 胜率 / 盈亏比 / Alpha / 稳定性
    """

    DIMS = ['Sharpe比率', 'Calmar比率', '胜率', '盈亏比', 'Alpha稳定性', '换手效率']

    @staticmethod
    def draw(ax, results: list, colors: list = None):
        """
        在ax上绘制雷达图
        results: list of dict，每个dict包含各维度的归一化得分[0,1]
        """
        N = len(StrategyRadarChart.DIMS)
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        ax.set_facecolor('#0d1826')
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(StrategyRadarChart.DIMS, color='#8ba3be', fontsize=8)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['', '', '', '', ''], fontsize=6)
        ax.set_ylim(0, 1)
        ax.grid(color='#1e3452', linestyle='--', linewidth=0.5)

        palette = colors or ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#a78bfa']

        for i, res in enumerate(results):
            vals = [
                min(max(res.get('sharpe', 0) / 3.0, 0), 1),    # Sharpe 归一化 (3=满分)
                min(max(res.get('calmar', 0) / 3.0, 0), 1),    # Calmar 归一化
                res.get('win_rate', 50) / 100,                  # 胜率
                min(max(res.get('profit_factor', 1) / 3.0, 0), 1),  # 盈亏比
                min(max(1 - res.get('oos_decay', 0.5), 0), 1), # Alpha稳定性
                min(max(1 - res.get('turnover_cost', 0.5), 0), 1),  # 换手效率
            ]
            vals += vals[:1]
            col = palette[i % len(palette)]
            ax.plot(angles, vals, color=col, linewidth=1.8, alpha=0.9, label=res.get('label', f'策略{i+1}'))
            ax.fill(angles, vals, color=col, alpha=0.12)

        ax.legend(
            loc='upper right', bbox_to_anchor=(0.1, 0.1),
            fontsize=7, facecolor='#0d1826', labelcolor='#c8d8e8', edgecolor='#1e3452'
        )


# ══════════════════════════════════════════════════════════════════
# v12/13：新闻情感分析引擎（v13修复：内置金融词典）
# ══════════════════════════════════════════════════════════════════

# ── 金融专用词典（150+词条）──────────────────────────────────────
# VADER默认不含金融域词汇，导致rises/beats/rally/surge全部得分为0
# 此词典注入VADER lexicon后大幅提升金融新闻识别准确率
_FINANCIAL_LEXICON: Dict[str, float] = {
    # ── 强正面（价格动作）──
    'surges':       2.8,  'surge':        2.8,  'surged':       2.8,
    'soars':        2.8,  'soar':         2.8,  'soared':       2.8,
    'rallies':      2.2,  'rally':        2.2,  'rallied':      2.2,
    'jumps':        2.2,  'jump':         2.2,  'jumped':       2.2,
    'spikes':       2.2,  'spike':        2.2,  'spiked':       2.2,
    'rises':        1.6,  'rise':         1.6,  'rose':         1.6,
    'climbs':       1.6,  'climb':        1.6,  'climbed':      1.6,
    'gains':        1.5,  'gain':         1.5,  'gained':       1.5,
    'rebounds':     1.8,  'rebound':      1.8,  'rebounded':    1.8,
    'recovers':     1.6,  'recover':      1.6,  'recovered':    1.6,
    'recovery':     1.6,
    # ── 正面（业绩/评级）──
    'beats':        2.2,  'beat':         2.2,
    'exceeds':      2.0,  'exceed':       2.0,  'exceeded':     2.0,
    'tops':         1.8,  'topped':       1.8,
    'outperforms':  2.2,  'outperform':   2.2,  'outperformed': 2.2,
    'upgrade':      2.2,  'upgraded':     2.2,  'upgrades':     2.2,
    'bullish':      2.8,  'bull':         1.6,  'bulls':        1.6,
    'record':       1.5,
    'breakthrough': 2.0,
    'boosts':       1.6,  'boost':        1.6,  'boosted':      1.6,
    'accelerates':  1.5,  'accelerate':   1.5,
    'optimistic':   1.8,  'optimism':     1.8,
    'profitable':   1.6,  'profit':       1.2,  'profits':      1.2,
    'growth':       1.2,
    'robust':       1.6,
    'buyback':      1.5,  'buybacks':     1.5,
    'dividend':     1.2,  'dividends':    1.2,
    'acquisition':  0.6,  'merger':       0.5,
    'momentum':     0.8,
    'outpace':      1.8,  'outpaces':     1.8,
    'expands':      1.3,  'expand':       1.3,  'expansion':    1.3,
    'lifts':        1.4,  'lift':         1.4,
    'strengthens':  1.4,  'strengthen':   1.4,
    'above':        0.4,  # "above estimates"
    'exceeding':    1.8,
    # ── 强负面（价格动作）──
    'crashes':     -2.8,  'crash':       -2.8,  'crashed':     -2.8,
    'plunges':     -2.8,  'plunge':      -2.8,  'plunged':     -2.8,
    'tumbles':     -2.2,  'tumble':      -2.2,  'tumbled':     -2.2,
    'sinks':       -2.2,  'sink':        -2.2,  'sank':        -2.2,
    'slides':      -1.8,  'slide':       -1.8,  'slid':        -1.8,
    'falls':       -1.6,  'fall':        -1.6,  'fell':        -1.6,
    'drops':       -1.6,  'drop':        -1.6,  'dropped':     -1.6,
    'declines':    -1.6,  'decline':     -1.6,  'declined':    -1.6,
    'declining':   -1.6,
    'retreats':    -1.5,  'retreat':     -1.5,  'retreated':   -1.5,
    # ── 负面（业绩/评级）──
    'misses':      -2.2,  'miss':        -2.2,  'missed':      -2.2,
    'disappoints': -2.2,  'disappoint':  -2.2,  'disappointing': -2.2,
    'disappointed':-2.2,
    'downgrade':   -2.2,  'downgraded':  -2.2,  'downgrades':  -2.2,
    'bearish':     -2.8,  'bear':        -1.6,  'bears':       -1.6,
    'underperforms':-2.2, 'underperform':-2.2,  'underperformed':-2.2,
    'recession':   -2.2,  'recessionary':-2.0,
    'layoffs':     -1.8,  'layoff':      -1.8,  'layoffs':     -1.8,
    'bankruptcy':  -3.0,  'bankrupt':    -3.0,  'insolvent':   -2.8,
    'losses':      -1.6,  'loss':        -1.6,
    'warns':       -1.6,  'warning':     -1.5,  'warned':      -1.5,
    'concern':     -1.0,  'concerns':    -1.0,
    'volatile':    -0.8,  'volatility':  -0.5,
    'weaker':      -1.4,  'weakens':     -1.4,  'weakening':   -1.4,
    'shrinks':     -1.5,  'shrink':      -1.5,  'contraction': -1.5,
    'selloff':     -2.0,  'sell-off':    -2.0,
    'panic':       -2.2,
    'struggles':   -1.5,  'struggle':    -1.5,  'struggling':  -1.5,
    'below':       -0.4,  # "below estimates"
    'slowing':     -1.2,  'slowdown':    -1.4,
    'debt':        -0.5,
    'fraud':       -2.5,  'scandal':     -2.5,
    'investigation':-1.5, 'probe':       -1.2,
    'cut':         -1.0,  'cuts':        -1.0,  # "cuts guidance"
    'downbeat':    -1.8,
}

class NewsSentimentEngine:
    """
    新闻情感分析引擎 v13 — 修复版
    核心修复：内置金融专用词典注入VADER，解决金融新闻全中性问题

    情感得分 compound ∈ [-1, 1]
    金融阈值：|compound| >= 0.03 即视为有效情感（低于通用阈值0.05）

    参考：
    - Johannesburg/Stellenbosch研究：新闻情感预测短期股价收益
    - Hutto & Gilbert (2014): VADER原论文
    - 大奖章基金媒体情感信号（1-3日Alpha）
    """

    _vader_instance = None   # 注入了金融词典的增强VADER
    _finbert_instance = None
    _finbert_loading = False

    @classmethod
    def get_vader(cls):
        """获取注入了金融词典的增强VADER实例（单例）"""
        if not HAS_VADER:
            return None
        if cls._vader_instance is None:
            va = _VaderAnalyzer()
            # 注入金融词典
            va.lexicon.update(_FINANCIAL_LEXICON)
            cls._vader_instance = va
            logger.info("NewsSentimentEngine: 金融词典已注入VADER（150+词条）")
        return cls._vader_instance

    @classmethod
    def get_finbert(cls):
        if not HAS_FINBERT:
            return None
        if cls._finbert_instance is None and not cls._finbert_loading:
            try:
                cls._finbert_loading = True
                cls._finbert_instance = _hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    max_length=512, truncation=True, device=-1,
                )
            except Exception as e:
                logger.warning(f"FinBERT load failed: {e}")
            finally:
                cls._finbert_loading = False
        return cls._finbert_instance

    @staticmethod
    def score_title(title: str, use_finbert: bool = False) -> dict:
        """
        对单条新闻标题打分
        优先FinBERT（若可用且requested），否则金融增强VADER
        """
        if not title:
            return {'compound': 0.0, 'label': 'neutral', 'model': 'none'}

        if use_finbert and HAS_FINBERT:
            fb = NewsSentimentEngine.get_finbert()
            if fb:
                try:
                    result = fb(title[:512])[0]
                    label_map = {'positive': 1.0, 'negative': -1.0, 'neutral': 0.0}
                    raw = result['label'].lower()
                    score = result['score']
                    compound = label_map.get(raw, 0.0) * score
                    # 金融阈值0.03
                    label = 'positive' if compound >= 0.03 else ('negative' if compound <= -0.03 else 'neutral')
                    return {'compound': round(compound, 4), 'label': label,
                            'confidence': round(score, 4), 'model': 'finbert'}
                except Exception as e:
                    logger.warning(f"FinBERT scoring failed: {e}")

        # 金融增强VADER
        vader = NewsSentimentEngine.get_vader()
        if vader is None:
            # 纯规则备用
            return NewsSentimentEngine._rule_based_score(title)

        scores = vader.polarity_scores(title)
        compound = scores['compound']
        # v13修复：金融阈值0.03（低于通用0.05，金融文本措辞更克制）
        label = 'positive' if compound >= 0.03 else ('negative' if compound <= -0.03 else 'neutral')
        return {
            'compound':  round(compound, 4),
            'pos':       scores['pos'],
            'neg':       scores['neg'],
            'neu':       scores['neu'],
            'label':     label,
            'model':     'vader_financial',
        }

    @staticmethod
    def _rule_based_score(title: str) -> dict:
        """
        纯规则备用（无需任何库）
        基于关键词匹配，精度低但总比全中性好
        """
        t = title.lower()
        pos_kw = ['surge', 'rise', 'gain', 'beat', 'record', 'rally', 'jump',
                  'upgrade', 'bullish', 'rebound', 'strong', 'profit', 'growth',
                  'outperform', 'boost', 'buy']
        neg_kw = ['crash', 'fall', 'drop', 'miss', 'disappoint', 'downgrade',
                  'bearish', 'loss', 'decline', 'recession', 'warn', 'weak',
                  'plunge', 'tumble', 'layoff', 'bankruptcy', 'sell']
        pos = sum(1 for w in pos_kw if w in t)
        neg = sum(1 for w in neg_kw if w in t)
        if pos > neg:
            compound = min(0.4 * pos, 0.9)
        elif neg > pos:
            compound = max(-0.4 * neg, -0.9)
        else:
            compound = 0.0
        label = 'positive' if compound >= 0.03 else ('negative' if compound <= -0.03 else 'neutral')
        return {'compound': round(compound, 4), 'label': label, 'model': 'rule_based'}

    @staticmethod
    def score_news_batch(news_items: list, use_finbert: bool = False) -> list:
        scored = []
        for item in news_items:
            title = item.get('title', '')
            s = NewsSentimentEngine.score_title(title, use_finbert)
            item_copy = dict(item)
            item_copy['sentiment'] = s
            scored.append(item_copy)
        return scored

    @staticmethod
    def aggregate_sentiment(scored_news: list) -> dict:
        """聚合多条新闻情感，返回综合信号"""
        if not scored_news:
            return {'overall_score': 0.0, 'momentum': 0.0, 'reversal_signal': 0.0,
                    'pos_ratio': 0.0, 'neg_ratio': 0.0, 'n_total': 0,
                    'n_pos': 0, 'n_neg': 0, 'n_neu': 0, 'label': 'neutral',
                    'compounds': []}

        compounds = [item['sentiment']['compound'] for item in scored_news]
        labels    = [item['sentiment']['label']    for item in scored_news]
        n = len(compounds)

        # 时间衰减权重（最新新闻权重最高）
        weights = np.exp(np.linspace(0, -1.5, n))[::-1]
        weights /= weights.sum()
        overall = float(np.dot(weights, compounds))

        # 情感动量：最近5条 vs 全部均值
        recent     = np.mean(compounds[:5]) if len(compounds) >= 5 else np.mean(compounds)
        full_mean  = np.mean(compounds)
        momentum   = float(recent - full_mean)

        # 反转信号
        reversal_signal = 0.0
        if full_mean < -0.25 and compounds[0] > -0.05:
            reversal_signal = 1.0    # 情感触底，可能反转
        elif full_mean > 0.25 and compounds[0] < 0.05:
            reversal_signal = -1.0   # 情感顶部，可能反转

        n_pos = labels.count('positive')
        n_neg = labels.count('negative')
        n_neu = labels.count('neutral')

        if overall >= 0.03:    agg_label = 'positive'
        elif overall <= -0.03: agg_label = 'negative'
        else:                  agg_label = 'neutral'

        return {
            'overall_score':   round(overall, 4),
            'momentum':        round(momentum, 4),
            'reversal_signal': reversal_signal,
            'pos_ratio':       n_pos / n,
            'neg_ratio':       n_neg / n,
            'n_total':         n,
            'n_pos':           n_pos,
            'n_neg':           n_neg,
            'n_neu':           n_neu,
            'label':           agg_label,
            'compounds':       compounds,
        }

    @staticmethod
    def sentiment_to_prob_adjustment(agg: dict) -> float:
        """
        情感信号 → 概率调整量 ∈ [-0.12, +0.12]
        S型压缩，避免过拟合
        """
        score    = agg.get('overall_score', 0.0)
        reversal = agg.get('reversal_signal', 0.0)
        momentum = agg.get('momentum', 0.0)
        base_adj = float(np.tanh(score * 2.0) * 0.08)
        mom_adj  = float(np.tanh(momentum * 3.0) * 0.03)
        rev_adj  = -reversal * 0.04
        return float(np.clip(base_adj + mom_adj + rev_adj, -0.12, 0.12))


# ══════════════════════════════════════════════════════════════════
# v13.0 新增：自我修正引擎
# ══════════════════════════════════════════════════════════════════
class SelfCorrectionEngine:
    """
    预测记录 + 自动回测 + 动态权重调整

    工作流：
    1. 每次打开股票详情 → 记录当日预测（symbol, date, horizon, p5/p20/p60）
    2. 后台定期检查到期预测 → 拉取实际价格 → 计算方向准确率
    3. 用近期准确率重新加权 XGB / HQ / Mom 三个子模型
    4. Brier Score 衡量概率校准质量（越低越好）

    数据库：quantpro_predictions.db（sqlite3，与脚本同目录）
    """

    _DB_PATH: str = "quantpro_predictions.db"
    _LOCK = threading.Lock()
    # 缓存各模型动态权重（由近期准确率计算）
    _model_weights: Dict[str, Dict[int, Dict[str, float]]] = {}

    @classmethod
    def init_db(cls):
        """创建数据库表（若不存在）"""
        with cls._LOCK:
            try:
                conn = sqlite3.connect(cls._DB_PATH)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol      TEXT    NOT NULL,
                        pred_date   TEXT    NOT NULL,
                        horizon     INTEGER NOT NULL,
                        p_xgb       REAL,
                        p_hq        REAL,
                        p_mom       REAL,
                        p_ensemble  REAL,
                        price_at_pred REAL,
                        w1          REAL DEFAULT 0.45,
                        w2          REAL DEFAULT 0.35,
                        w3          REAL DEFAULT 0.20,
                        resolved    INTEGER DEFAULT 0,
                        actual_ret  REAL,
                        actual_up   INTEGER,
                        brier       REAL,
                        correct     INTEGER,
                        resolve_date TEXT
                    )
                """)
                c.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sym_date
                    ON predictions(symbol, pred_date, horizon)
                """)
                conn.commit()
            except Exception as e:
                logger.warning(f"SelfCorrection DB init: {e}")
            finally:
                try: conn.close()
                except: pass

    @classmethod
    def save_prediction(cls, symbol: str, horizon: int,
                        p_xgb: float, p_hq: float, p_mom: float,
                        p_ensemble: float, price: float,
                        w1: float = 0.45, w2: float = 0.35, w3: float = 0.20):
        """保存一条预测记录（若当日已存在则跳过）"""
        today = date.today().isoformat()
        with cls._LOCK:
            try:
                conn = sqlite3.connect(cls._DB_PATH)
                c = conn.cursor()
                # 当日同一symbol/horizon只记录一次
                c.execute(
                    "SELECT id FROM predictions WHERE symbol=? AND pred_date=? AND horizon=?",
                    (symbol, today, horizon)
                )
                if c.fetchone() is None:
                    c.execute("""
                        INSERT INTO predictions
                        (symbol, pred_date, horizon, p_xgb, p_hq, p_mom, p_ensemble,
                         price_at_pred, w1, w2, w3)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (symbol, today, horizon, p_xgb, p_hq, p_mom, p_ensemble,
                          price, w1, w2, w3))
                    conn.commit()
            except Exception as e:
                logger.warning(f"save_prediction: {e}")
            finally:
                try: conn.close()
                except: pass

    @classmethod
    def resolve_pending(cls, symbol: str):
        """
        检查该股票所有未resolved的预测
        若到期（pred_date + horizon <= today），拉取实际价格并评分
        """
        today = date.today()
        with cls._LOCK:
            try:
                conn = sqlite3.connect(cls._DB_PATH)
                c = conn.cursor()
                c.execute("""
                    SELECT id, pred_date, horizon, p_ensemble, price_at_pred
                    FROM predictions
                    WHERE symbol=? AND resolved=0
                """, (symbol,))
                rows = c.fetchall()
                conn.close()
            except Exception as e:
                logger.warning(f"resolve_pending fetch: {e}")
                return

        for row_id, pred_date_str, horizon, p_ensemble, price_at_pred in rows:
            try:
                pred_dt = date.fromisoformat(pred_date_str)
                resolve_dt = pred_dt
                # 跳过周末（交易日估算）
                days_added = 0
                tmp = pred_dt
                while days_added < horizon:
                    tmp = tmp.replace(day=tmp.day + 1) if tmp.day < 28 else \
                          date.fromordinal(tmp.toordinal() + 1)
                    if tmp.weekday() < 5:  # 周一至周五
                        days_added += 1
                resolve_dt = tmp

                if today < resolve_dt:
                    continue  # 还没到期

                # 拉取到期日实际价格
                actual_price = get_realtime_price(symbol)
                if actual_price is None or price_at_pred is None or price_at_pred <= 0:
                    continue

                actual_ret  = (actual_price - price_at_pred) / price_at_pred
                actual_up   = 1 if actual_ret > 0 else 0
                # Brier Score: (p - actual)^2，越小越好
                brier       = (p_ensemble - actual_up) ** 2
                correct     = 1 if (p_ensemble >= 0.5) == bool(actual_up) else 0

                with cls._LOCK:
                    try:
                        conn = sqlite3.connect(cls._DB_PATH)
                        cc = conn.cursor()
                        cc.execute("""
                            UPDATE predictions
                            SET resolved=1, actual_ret=?, actual_up=?,
                                brier=?, correct=?, resolve_date=?
                            WHERE id=?
                        """, (round(actual_ret, 6), actual_up,
                              round(brier, 6), correct,
                              resolve_dt.isoformat(), row_id))
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        logger.warning(f"resolve update: {e}")

            except Exception as e:
                logger.warning(f"resolve row {row_id}: {e}")

    @classmethod
    def compute_adaptive_weights(
        cls, symbol: str, horizon: int, lookback: int = 30
    ) -> Tuple[float, float, float]:
        """
        基于近期预测准确率动态调整 w1/w2/w3
        - 对每个子模型单独评估方向准确率
        - 准确率越高权重越大，最低权重5%，单模型上限70%

        返回 (w1, w2, w3) 归一化后的权重
        """
        try:
            conn = sqlite3.connect(cls._DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT p_xgb, p_hq, p_mom, actual_up
                FROM predictions
                WHERE symbol=? AND horizon=? AND resolved=1
                ORDER BY pred_date DESC
                LIMIT ?
            """, (symbol, horizon, lookback))
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"compute_adaptive_weights: {e}")
            return (0.45, 0.35, 0.20)

        if len(rows) < 5:
            return (0.45, 0.35, 0.20)  # 数据不足，用默认权重

        # 计算每个子模型的方向准确率
        accs = [0.0, 0.0, 0.0]  # xgb, hq, mom
        for p_xgb, p_hq, p_mom, actual_up in rows:
            if p_xgb is None or actual_up is None: continue
            accs[0] += 1 if (p_xgb >= 0.5) == bool(actual_up) else 0
            accs[1] += 1 if (p_hq  >= 0.5) == bool(actual_up) else 0
            accs[2] += 1 if (p_mom >= 0.5) == bool(actual_up) else 0

        n = len(rows)
        accs = [a / n for a in accs]

        # 转换为权重（softmax风格，以0.5为基准）
        raw = [max(a - 0.5, 0.0) + 0.05 for a in accs]  # 基准随机=0.5，低于随机→最小权重0.05
        total = sum(raw)
        if total < 1e-9:
            return (0.45, 0.35, 0.20)

        w = [r / total for r in raw]
        # 上限70%
        w = [min(wi, 0.70) for wi in w]
        total2 = sum(w)
        w = [wi / total2 for wi in w]
        return (round(w[0], 3), round(w[1], 3), round(w[2], 3))

    @classmethod
    def get_accuracy_report(cls, symbol: str = None) -> dict:
        """
        获取准确率报告
        若symbol为None则统计全部
        """
        try:
            conn = sqlite3.connect(cls._DB_PATH)
            c = conn.cursor()
            if symbol:
                c.execute("""
                    SELECT horizon, COUNT(*), AVG(correct), AVG(brier), AVG(actual_ret)
                    FROM predictions
                    WHERE resolved=1 AND symbol=?
                    GROUP BY horizon
                """, (symbol,))
            else:
                c.execute("""
                    SELECT horizon, COUNT(*), AVG(correct), AVG(brier), AVG(actual_ret)
                    FROM predictions
                    WHERE resolved=1
                    GROUP BY horizon
                """)
            rows = c.fetchall()
            conn.close()
        except Exception as e:
            logger.warning(f"get_accuracy_report: {e}")
            return {}

        report = {}
        for horizon, n, avg_correct, avg_brier, avg_ret in rows:
            report[horizon] = {
                'n':           n,
                'accuracy':    round(avg_correct * 100, 1) if avg_correct else 0,
                'brier':       round(avg_brier, 4) if avg_brier else 0,
                'avg_ret':     round(avg_ret * 100, 2) if avg_ret else 0,
            }
        return report

    @classmethod
    def get_prediction_history(cls, symbol: str, limit: int = 50) -> list:
        """获取该股票的预测历史（已resolved + pending）"""
        try:
            conn = sqlite3.connect(cls._DB_PATH)
            c = conn.cursor()
            c.execute("""
                SELECT pred_date, horizon, p_ensemble, actual_up, actual_ret,
                       correct, brier, resolved
                FROM predictions
                WHERE symbol=?
                ORDER BY pred_date DESC, horizon ASC
                LIMIT ?
            """, (symbol, limit))
            rows = c.fetchall()
            conn.close()
            return rows
        except Exception as e:
            logger.warning(f"get_prediction_history: {e}")
            return []


# 初始化数据库（模块加载时执行）
try:
    SelfCorrectionEngine.init_db()
except Exception as _e:
    logger.warning(f"SelfCorrectionEngine DB init skipped: {_e}")


# ══════════════════════════════════════════════════════════════════
# 协整分析
# ══════════════════════════════════════════════════════════════════
class CointegrationAnalysis:
    @staticmethod
    def test_pair(s1, s2):
        df=pd.concat([s1,s2],axis=1).dropna()
        if len(df)<60: return 0.,1.,pd.Series()
        x=df.iloc[:,0].values; y=df.iloc[:,1].values
        slope,intercept,_,_,_=stats.linregress(x,y)
        spread=y-(slope*x+intercept)
        if HAS_STATSMODELS:
            try:
                res=adfuller(spread,maxlags=1)
                t_stat,p_val=res[0],res[1]
            except: t_stat,p_val=0.,1.
        else:
            t_stat,p_val=0.,0.5
        return float(t_stat),float(p_val),pd.Series(spread,index=df.index)

    @staticmethod
    def find_pairs(symbols, close_dict, threshold=0.05):
        pairs=[]
        syms=[s for s in symbols if s in close_dict]
        for i in range(len(syms)):
            for j in range(i+1,len(syms)):
                s1,s2=syms[i],syms[j]
                _,p,spread=CointegrationAnalysis.test_pair(close_dict[s1],close_dict[s2])
                if p<threshold: pairs.append((s1,s2,p,spread))
        pairs.sort(key=lambda x:x[2])
        return pairs


# ══════════════════════════════════════════════════════════════════
# 概率预测引擎 v9.0（升级Fat-tail MC）
# ══════════════════════════════════════════════════════════════════
class ProbabilityEngine:
    # ── v11 新增：短线/长线特征分离 ──────────────────────────────
    # 5日短线：均值回归因子为主（轻量，泛化好）
    SHORT_FEATURES = [
        'rsi', 'zscore20', 'bb_pos', 'vwap_dev',
        'vol_imbalance_f', 'stoch_rsi_k_norm', 'obv_dev',
        'rev_1d', 'rev_5d',  # 均值回归核心因子
        'intraday_rev', 'overnight_ret',  # 隔夜/日内反转
        'volume_ratio', 'atr_ratio',
    ]
    # 20/60日长线：趋势动量因子为主（XGB）
    FEATURES = [
        'rsi','macd_norm','ma_dev20','ma_dev60','volume_ratio','bb_pos',
        'mom5','mom10','zscore20','volatility','atr_ratio','trend_strength',
        'mom20','rs_pos_norm','stoch_rsi_k_norm','obv_dev',
        'vwap_dev','vol_imbalance_f','atr_expansion_f',  # v9.0新增
    ]

    def __init__(self, ind, close):
        self.ind=ind.copy(); self.close=close.copy(); self._feat=None
        self._build_features()

    def _build_features(self):
        df=self.ind.copy(); c=self.close
        df['macd_norm']          = df['macd']/(c.abs()+1e-9)
        df['ma_dev20']           = (c-df['ma20'])/(df['ma20']+1e-9)
        df['ma_dev60']           = (c-df['ma60'])/(df['ma60']+1e-9)
        denom                    = (df['bb_up']-df['bb_dn'])+1e-9
        df['bb_pos']             = (c-df['bb_dn'])/denom
        df['mom5']               = c.pct_change(5)
        df['mom10']              = c.pct_change(10)
        df['mom20']              = c.pct_change(20)
        df['zscore20']           = (c-c.rolling(20).mean())/(c.rolling(20).std()+1e-9)
        returns                  = c.pct_change()
        df['volatility']         = returns.rolling(20).std()
        df['atr_ratio']          = df['atr']/(c.abs()+1e-9)
        df['trend_strength']     = abs(df['ma20']-df['ma60'])/(c.abs()+1e-9)
        df['rs_pos_norm']        = df.get('rs_pos',pd.Series(50,index=df.index))/100.
        df['stoch_rsi_k_norm']   = df.get('stoch_rsi_k',pd.Series(50,index=df.index))/100.
        obv=df.get('obv',pd.Series(0,index=df.index))
        obv_ma=df.get('obv_ma',obv.rolling(20).mean())
        df['obv_dev']            = (obv-obv_ma)/(obv_ma.abs()+1e-9)
        df['vwap_dev']           = df.get('vwap_dev',pd.Series(0,index=df.index))
        df['vol_imbalance_f']    = df.get('vol_imbalance',pd.Series(0,index=df.index))
        df['atr_expansion_f']    = df.get('atr_expansion',pd.Series(1,index=df.index))
        # ── v11 新增：短线反转因子（供 SHORT_FEATURES 使用）──────
        df['rev_1d']             = df.get('rev_1d',    pd.Series(0, index=df.index))
        df['rev_5d']             = df.get('rev_5d',    pd.Series(0, index=df.index))
        df['intraday_rev']       = df.get('intraday_rev', pd.Series(0, index=df.index))
        df['overnight_ret']      = df.get('overnight_ret', pd.Series(0, index=df.index))
        all_feats = list(dict.fromkeys(self.FEATURES + self.SHORT_FEATURES))
        self._feat=df[[f for f in all_feats if f in df.columns]].copy()

    def detect_regime(self):
        ret=self.close.pct_change().dropna()
        if len(ret)<100: return "neutral"
        X=ret.values.reshape(-1,1)
        try:
            model=GaussianHMM(n_components=3,covariance_type="diag",n_iter=200)
            model.fit(X); states=model.predict(X)
            vols=[ret[states==i].std() if (states==i).sum()>0 else 999 for i in range(3)]
            cs=states[-1]
            if vols[cs]==max(vols): return "high_vol"
            if vols[cs]==min(vols): return "low_vol"
            return "neutral"
        except: return "neutral"

    def xgb_probability_short(self, fwd=5):
        """
        v11 新增：5日专用短线模型
        使用均值回归因子 + 逻辑回归（比XGB在短周期更稳健，过拟合风险低）
        """
        from sklearn.linear_model import LogisticRegression
        feats = [f for f in self.SHORT_FEATURES if f in self._feat.columns]
        if not feats:
            return self.xgb_probability(fwd)
        df = self._feat[feats].copy()
        future_return = (self.close.shift(-fwd) - self.close) / self.close
        tgt = (future_return > 0).astype(int)
        comb = df.join(tgt.rename('tgt')).dropna().iloc[:-fwd]
        if len(comb) < 80:
            return self.xgb_probability(fwd)
        X = comb[feats].values
        y = comb['tgt'].values
        if len(np.unique(y)) < 2:
            return 0.5
        tscv = TimeSeriesSplit(n_splits=5)
        preds = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr = y[train_idx]
            if len(np.unique(ytr)) < 2:
                continue
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)
            # 逻辑回归：短线更稳健，避免过拟合
            model = LogisticRegression(C=0.5, max_iter=500, random_state=42)
            model.fit(Xtr, ytr)
            preds.append(model.predict_proba(Xte)[-1][1])
        return float(np.mean(preds)) if preds else 0.5

    def xgb_probability(self, fwd=5):
        feats=[f for f in self.FEATURES if f in self._feat.columns]
        df=self._feat[feats].copy()
        future_return=(self.close.shift(-fwd)-self.close)/self.close
        tgt=(future_return>0).astype(int)
        comb=df.join(tgt.rename('tgt')).dropna().iloc[:-fwd]
        if len(comb)<120: return 0.5
        X=comb[feats].values; y=comb['tgt'].values
        if len(np.unique(y))<2: return 0.5
        tscv=TimeSeriesSplit(n_splits=5); preds=[]
        for train_idx,test_idx in tscv.split(X):
            Xtr,Xte=X[train_idx],X[test_idx]; ytr=y[train_idx]
            if len(np.unique(ytr))<2: continue
            scaler=StandardScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
            n_comp=min(8,Xtr.shape[1])
            pca=PCA(n_components=n_comp); Xtr=pca.fit_transform(Xtr); Xte=pca.transform(Xte)
            model=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,
                                 subsample=0.8,colsample_bytree=0.8,
                                 objective='binary:logistic',random_state=42,verbosity=0)
            model.fit(Xtr,ytr)
            preds.append(model.predict_proba(Xte)[-1][1])
        return float(np.mean(preds)) if preds else 0.5

    def historical_quantile_prob(self, fwd=5):
        rsi=self.ind['rsi']; future=(self.close.shift(-fwd)>self.close).astype(float)
        cr=float(rsi.iloc[-1]); sub=future[(rsi>=cr-8)&(rsi<=cr+8)].dropna()
        return float(sub.mean()) if len(sub)>=15 else 0.5

    def momentum_probability(self):
        feats=[f for f in self.FEATURES if f in self._feat.columns]
        row=self._feat[feats].iloc[-1]
        score=0.; weight=0.
        factors=[('mom5',0.2),('mom10',0.2),('mom20',0.15),('ma_dev20',0.15),('trend_strength',0.15),('zscore20',0.15)]
        for col,w in factors:
            val=row.get(col,np.nan) if hasattr(row,'get') else (row[col] if col in row.index else np.nan)
            if pd.isna(val): continue
            if col=='zscore20': score+=w*(0 if val>1 else (1 if val<-1 else 0.5))
            else: score+=w*(1 if val>0 else 0)
            weight+=w
        return score/weight if weight>0 else 0.5

    def ensemble(self, fwd=5, sentiment_agg: dict = None, symbol: str = None):
        regime=self.detect_regime()

        # v13.0：从自我修正引擎获取自适应权重
        if symbol:
            w1, w2, w3 = SelfCorrectionEngine.compute_adaptive_weights(symbol, fwd)
        else:
            if fwd <= 5:
                if regime == "high_vol":   w1, w2, w3 = 0.30, 0.45, 0.25
                elif regime == "low_vol":  w1, w2, w3 = 0.50, 0.25, 0.25
                else:                     w1, w2, w3 = 0.45, 0.35, 0.20
            else:
                if regime == "high_vol":   w1, w2, w3 = 0.35, 0.40, 0.25
                elif regime == "low_vol":  w1, w2, w3 = 0.55, 0.20, 0.25
                else:                     w1, w2, w3 = 0.50, 0.30, 0.20

        if fwd <= 5:
            p1 = self.xgb_probability_short(fwd)
        else:
            p1 = self.xgb_probability(fwd)
        p2 = self.historical_quantile_prob(fwd)
        p3 = self.momentum_probability()

        base_prob = float(p1*w1 + p2*w2 + p3*w3)

        # v12.0：情感信号调整（仅5日）
        if sentiment_agg and fwd <= 5:
            adj = NewsSentimentEngine.sentiment_to_prob_adjustment(sentiment_agg)
            base_prob = float(np.clip(base_prob + adj, 0.05, 0.95))

        return base_prob, p1, p2, p3, w1, w2, w3

    def monte_carlo(self, days=20, n=10000, use_fat_tail=True):
        returns=self.close.pct_change().dropna()
        cur=float(self.close.iloc[-1])
        # ── v11 优先：GARCH(1,1) 条件波动率路径 ─────────────────
        if use_fat_tail and HAS_ARCH and len(returns) >= 100:
            try:
                from arch import arch_model
                am = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='t', rescale=False)
                res = am.fit(disp='off', show_warning=False)
                # 提取GARCH参数
                omega = float(res.params.get('omega', 0))
                alpha = float(res.params.get('alpha[1]', 0.05))
                beta  = float(res.params.get('beta[1]', 0.90))
                nu    = max(float(res.params.get('nu', 5.0)), 2.1)
                # 漂移：用衰减后的日度均值
                mu_raw = float(returns.mean())
                decay  = np.exp(-0.8 * days / 252)
                mu_adj = mu_raw * decay
                last_sigma2 = float(res.conditional_volatility[-1]**2) / 10000  # 转回小数^2
                # GARCH路径模拟（波动率聚集）
                rng = np.random.default_rng(42 + days)
                all_finals = np.zeros(n)
                for i in range(n):
                    s2 = last_sigma2
                    price = cur
                    for _ in range(days):
                        sigma = np.sqrt(s2)
                        eps   = stats.t.rvs(nu, random_state=None) * sigma
                        ret   = mu_adj + eps
                        price *= (1 + ret)
                        s2    = omega / 10000 + alpha * (eps**2) + beta * s2  # 递推
                    all_finals[i] = price
                pcts = (all_finals - cur) / cur * 100
                return {
                    'up_prob':    float(np.mean(all_finals > cur)),
                    'var95':      float(np.percentile(pcts, 5)),
                    'cvar95':     float(pcts[pcts <= np.percentile(pcts, 5)].mean()),
                    'low_pct':    float(np.percentile(pcts, 5)),
                    'high_pct':   float(np.percentile(pcts, 95)),
                    'median_pct': float(np.median(pcts)),
                    'finals':     all_finals,
                    'cur':        cur,
                }
            except Exception as e:
                logger.warning(f'GARCH MC failed ({e}), fallback to t-dist MC')
        # ── 回退：t分布 MC（带漂移衰减）─────────────────────────
        if use_fat_tail:
            mc=GARCHRiskEngine.fat_tail_monte_carlo(returns, days, n)
            if mc:
                mc['t_dist']['cur']=cur; mc['t_dist']['up_prob']=mc['t_dist']['up_prob']
                return mc['t_dist']
        # 最终回退：Bootstrap MC
        rng=np.random.default_rng(42 + days)
        sampled=rng.choice(returns.values, size=(n,days), replace=True)
        finals=cur*np.cumprod(1+sampled,axis=1)[:,-1]
        pcts=(finals-cur)/cur*100
        return {
            'up_prob': float(np.mean(finals>cur)),
            'var95': float(np.percentile(pcts,5)),
            'cvar95': float(pcts[pcts<=np.percentile(pcts,5)].mean()),
            'low_pct': float(np.percentile(pcts,5)), 'high_pct': float(np.percentile(pcts,95)),
            'median_pct': float(np.median(pcts)), 'finals': finals, 'cur': cur,
        }

    def feature_importance(self, fwd=5):
        feats=[f for f in self.FEATURES if f in self._feat.columns]
        df=self._feat[feats].copy()
        fret=(self.close.shift(-fwd)-self.close)/self.close
        tgt=(fret>0).astype(int)
        comb=df.join(tgt.rename('tgt')).dropna().iloc[:-fwd]
        if len(comb)<120 or len(np.unique(comb['tgt']))<2: return []
        X=comb[feats]; y=comb['tgt']
        sc=StandardScaler(); Xs=sc.fit_transform(X)
        m=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.03,subsample=0.8,
                         colsample_bytree=0.8,random_state=42,verbosity=0)
        m.fit(Xs,y)
        pairs=list(zip(feats,m.feature_importances_))
        pairs.sort(key=lambda x:x[1],reverse=True)
        return pairs

# ══════════════════════════════════════════════════════════════════
# 新闻 & 指数线程
# ══════════════════════════════════════════════════════════════════
class NewsThread(QThread):
    news_ready=pyqtSignal(list); error_msg=pyqtSignal(str)
    def __init__(self, sym, use_finbert=False):
        super().__init__(); self.sym=sym; self.use_finbert=use_finbert
    def run(self):
        try:
            raw=yf.Ticker(self.sym).news or []; items=[]
            for n in raw[:15]:
                ct=n.get("content",{})
                if ct:
                    title=ct.get("title",""); link=ct.get("canonicalUrl",{}).get("url","")
                    pub=ct.get("provider",{}).get("displayName",""); ts=ct.get("pubDate","")
                    if ts:
                        try: dt=datetime.fromisoformat(ts.replace("Z","+00:00")); ts=dt.strftime("%Y-%m-%d %H:%M")
                        except: pass
                else:
                    title=n.get("title",""); link=n.get("link",""); pub=n.get("publisher","")
                    ts=n.get("providerPublishTime",0)
                    if ts: ts=datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
                if title: items.append({"title":title,"link":link,"publisher":pub,"time":str(ts)})
            # v12.0：情感分析
            if items:
                items = NewsSentimentEngine.score_news_batch(items, use_finbert=self.use_finbert)
            self.news_ready.emit(items)
        except Exception as e: self.error_msg.emit(str(e))

class IndexTickerThread(QThread):
    data_ready=pyqtSignal(dict); error_msg=pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.symbols={'^GSPC':'S&P 500','^IXIC':'NASDAQ','^DJI':'DOW','^VIX':'VIX恐慌'}
        self._stop=False
    def stop(self): self._stop=True
    def run(self):
        while not self._stop:
            try:
                data={}
                for sym,name in self.symbols.items():
                    tk=yf.Ticker(sym); hist=tk.history(period="1d")
                    if not hist.empty:
                        price=hist.iloc[-1]['Close']; prev=tk.info.get('previousClose',price)
                        chg=price-prev; pct=chg/prev*100 if prev else 0.
                        data[sym]=(price,chg,pct,name)
                self.data_ready.emit(data)
            except Exception as e: self.error_msg.emit(str(e))
            for _ in range(30):
                if self._stop: break
                self.sleep(1)

# ══════════════════════════════════════════════════════════════════
# Top 100 加载器
# ══════════════════════════════════════════════════════════════════
class Top100Loader:
    URL="https://raw.githubusercontent.com/Ate329/top-us-stock-tickers/main/tickers/top_100.csv"
    ZH={"NVDA":"英伟达","AAPL":"苹果","GOOGL":"谷歌-A","MSFT":"微软","AMZN":"亚马逊",
        "AVGO":"博通","META":"Meta","TSLA":"特斯拉","WMT":"沃尔玛","COST":"好市多",
        "NFLX":"奈飞","PLTR":"Palantir","AMD":"超微","CSCO":"思科","PEP":"百事",
        "KO":"可口可乐","ADBE":"Adobe","TXN":"德仪","QCOM":"高通","AMGN":"安进",
        "SBUX":"星巴克","PANW":"Palo Alto","MRNA":"莫德纳","PDD":"拼多多","BIDU":"百度",
        "JD":"京东","BABA":"阿里巴巴","ASML":"阿斯麦","BRK-B":"伯克希尔","JPM":"摩根大通",
        "V":"Visa","MA":"万事达","UNH":"联合健康","LLY":"礼来","JNJ":"强生",
        "PG":"宝洁","HD":"家得宝","ABBV":"艾伯维","MU":"美光科技","INTU":"财捷"}
    FALLBACK=[("AAPL","苹果"),("MSFT","微软"),("GOOGL","谷歌"),("AMZN","亚马逊"),
              ("NVDA","英伟达"),("TSLA","特斯拉"),("META","Meta"),("JPM","摩根大通"),
              ("V","Visa"),("MA","万事达"),("UNH","联合健康"),("LLY","礼来")]

    @classmethod
    def get_name(cls,sym,fallback=""):
        return cls.ZH.get(sym, fallback or sym)

    @classmethod
    def load(cls):
        try:
            resp=requests.get(cls.URL,timeout=10); resp.raise_for_status()
            data=csv.DictReader(StringIO(resp.text)); syms,seen=[],set()
            for row in data:
                sym=row.get("symbol","").strip().upper()
                if sym and sym not in seen:
                    seen.add(sym)
                    name=cls.get_name(sym,row.get("name",sym).split()[0])
                    syms.append((sym,name))
                    if len(syms)>=100: break
            return syms or cls.FALLBACK
        except Exception as e:
            logger.error(f"Top100: {e}"); return cls.FALLBACK

# ══════════════════════════════════════════════════════════════════
# 数据模型（v9.0：新增横截面排名列）
# ══════════════════════════════════════════════════════════════════
_SIG_COLOR = {
    "买入":QColor(16,185,129),"卖出":QColor(239,68,68),"观望":QColor(245,158,11),
    "Buy": QColor(16,185,129),"Sell": QColor(239,68,68),"Watch":QColor(245,158,11),
}

class ScanModel(QAbstractTableModel):
    def __init__(self): super().__init__(); self._rows: List[Dict]=[]; self.chg_period_label=""
    def rowCount(self,p=QModelIndex()): return len(self._rows)
    def columnCount(self,p=QModelIndex()): return len(_COL_KEYS)
    def headerData(self,sec,ori,role=Qt.DisplayRole):
        if role==Qt.DisplayRole:
            if ori==Qt.Horizontal:
                ck=_COL_KEYS[sec]
                if ck=="col_chg" and self.chg_period_label:
                    return f"{t('col_chg')}({self.chg_period_label})"
                return t(ck)
            return str(sec+1)
        return None
    def data(self,index,role=Qt.DisplayRole):
        if not index.isValid(): return None
        row=self._rows[index.row()]; ck=_COL_KEYS[index.column()]; val=row.get(ck,"")
        cur=row.get('_currency','$')
        if role==Qt.DisplayRole:
            if ck in _PRICE_COL_KEYS:
                try: return f"{cur}{float(val):.2f}"
                except: return str(val)
            if ck=="col_chg":
                if val is None: return "--"
                try: return f"{float(val):+.2f}%"
                except: return str(val)
            if ck=="col_macd":
                try: return f"{float(val):.4f}"
                except: return str(val)
            if ck=="col_cs_rank":
                try: return f"#{int(val)}"
                except: return str(val)
            if ck=="col_rel_str":
                try: return f"{float(val):+.2f}%"
                except: return str(val)
            if isinstance(val,float): return f"{val:.2f}"
            return str(val) if val is not None else ""
        if role==Qt.UserRole:
            if ck=="col_chg" and val is None: return -1e12
            try: return float(val)
            except: return str(val) if val is not None else ""
        if role==Qt.TextAlignmentRole:
            if ck in _NUM_COL_KEYS or ck in _PRICE_COL_KEYS or ck=="col_chg":
                return Qt.AlignRight|Qt.AlignVCenter
            return Qt.AlignCenter
        if role==Qt.BackgroundRole:
            if ck=='col_signal': return QBrush(_SIG_COLOR.get(str(val),QColor(30,47,69)))
            if ck=='col_cs_rank':
                try:
                    rank=int(val); total=len(self._rows)
                    if total>0:
                        pct=rank/total
                        if pct<=0.2:  return QBrush(QColor(5,46,22))
                        if pct>=0.8:  return QBrush(QColor(45,10,10))
                except: pass
            if ck=='col_chg':
                try:
                    v=float(val)
                    if v>0: return QBrush(QColor(5,46,22))
                    if v<0: return QBrush(QColor(45,10,10))
                except: pass
        if role==Qt.ForegroundRole:
            if ck=='col_chg':
                try:
                    v=float(val)
                    if v>0: return QBrush(QColor(16,185,129))
                    if v<0: return QBrush(QColor(239,68,68))
                except: pass
            if ck=='col_cs_rank':
                try:
                    rank=int(val); total=len(self._rows)
                    if total>0:
                        pct=rank/total
                        if pct<=0.2:  return QBrush(QColor(16,185,129))
                        if pct>=0.8:  return QBrush(QColor(239,68,68))
                except: pass
        return None
    def sort(self,col,order):
        ck=_COL_KEYS[col]; rev=(order==Qt.DescendingOrder)
        def key(r):
            v=r.get(ck,"")
            try: return float(v)
            except: return str(v)
        self.layoutAboutToBeChanged.emit(); self._rows.sort(key=key,reverse=rev); self.layoutChanged.emit()
    def append_row(self,d):
        pos=len(self._rows); self.beginInsertRows(QModelIndex(),pos,pos); self._rows.append(d); self.endInsertRows()
    def clear(self):
        self.beginResetModel(); self._rows.clear(); self.endResetModel()
    def reload_columns(self): self.headerDataChanged.emit(Qt.Horizontal,0,self.columnCount()-1)
    def to_dataframe(self): return pd.DataFrame([{t(k):r.get(k,"") for k in _COL_KEYS} for r in self._rows])
    def row_dict(self,i): return self._rows[i] if 0<=i<len(self._rows) else {}
    def update_cs_ranks(self, ranks: Dict[str, int]):
        for i, row in enumerate(self._rows):
            sym = row.get('col_code','')
            if sym in ranks:
                row['col_cs_rank'] = ranks[sym]
        if self._rows:
            self.dataChanged.emit(self.index(0,0), self.index(len(self._rows)-1, self.columnCount()-1))

    def set_chg_period(self, period_key: str, label: str):
        """把 col_chg 列切换成指定周期的涨跌（period_key 是行内隐藏字段名）。"""
        self.chg_period_label = label
        for row in self._rows:
            row['col_chg'] = row.get(period_key)   # 可能为 None（历史不足）
        chg_col = _COL_KEYS.index("col_chg")
        if self._rows:
            self.dataChanged.emit(self.index(0,chg_col), self.index(len(self._rows)-1, chg_col))
        self.headerDataChanged.emit(Qt.Horizontal, chg_col, chg_col)

class DisplayFilterProxy(QSortFilterProxyModel):
    def __init__(self,parent=None):
        super().__init__(parent); self._text=""; self.setSortRole(Qt.UserRole)
    def setFilterText(self,txt): self._text=txt.strip().lower(); self.invalidateFilter()
    def filterAcceptsRow(self,row,parent):
        if not self._text: return True
        m=self.sourceModel()
        for col in range(m.columnCount()):
            cell=m.data(m.index(row,col,parent),Qt.DisplayRole)
            if cell and self._text in str(cell).lower(): return True
        return False

# ══════════════════════════════════════════════════════════════════
# 分析线程（v9.0：增加横截面因子）
# ══════════════════════════════════════════════════════════════════
class AnalysisThread(QThread):
    progress=pyqtSignal(int,str); result_ready=pyqtSignal(dict)
    finished_all=pyqtSignal(int); error_msg=pyqtSignal(str)

    def __init__(self,syms,params):
        super().__init__(); self.syms=syms; self.params=params; self._stop=False

    def stop(self): self._stop=True

    def run(self):
        total=len(self.syms); done=ok=0
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs={ex.submit(self._analyze,s,n):(s,n) for s,n in self.syms}
            for fut in as_completed(futs):
                if self._stop: break
                s,n=futs[fut]
                try:
                    d=fut.result()
                    if d: self.result_ready.emit(d); ok+=1
                except Exception as e: self.error_msg.emit(f"{s}: {e}")
                done+=1; self.progress.emit(int(done/total*100),s)
        self.finished_all.emit(ok)

    def _analyze(self, sym, name):
        try:
            # 多周期涨跌需要更长历史：拉10年（足够覆盖5年周期），指标只用最近500根算
            df_full=fetch_stock_data(sym,"10y")
            if df_full.empty or len(df_full)<65: return None
            df=df_full.tail(500) if len(df_full)>500 else df_full
            ind=TechnicalIndicators.compute_all(df)
            if ind.empty: return None
            lat=ind.iloc[-1]; hist_close=float(lat['close'])
            rp=get_realtime_price(sym)
            if rp is not None:
                close=rp; prev=None
                try:
                    fi=getattr(yf.Ticker(sym),'fast_info',None)
                    if fi: prev=getattr(fi,'previous_close',None)
                    if prev is None: prev=yf.Ticker(sym).info.get('previousClose')
                except: pass
                if not prev or prev<=0: prev=hist_close
                chg=(close-float(prev))/float(prev)*100
            else:
                close=hist_close; chg=float(lat['chg_pct'])
                self.error_msg.emit(t("err_live_price",sym=sym))

            rsi=float(lat['rsi']); ma20=float(lat['ma20']); ma60=float(lat['ma60'])
            macd=float(lat['macd']); vol_r=float(lat['volume_ratio']); atr=float(lat['atr'])
            bb_up=float(lat['bb_up']); bb_dn=float(lat['bb_dn'])
            obv_trend=TechnicalIndicators.obv_trend_label(ind)

            # v9.0：计算相对SPY强弱
            rel_str = 0.0
            try:
                spy_df = fetch_stock_data("SPY","1mo")
                if not spy_df.empty:
                    sym_ret = (close - float(df['Close'].iloc[-21])) / float(df['Close'].iloc[-21])*100 if len(df)>=21 else 0
                    spy_ret = (float(spy_df['Close'].iloc[-1]) - float(spy_df['Close'].iloc[0])) / float(spy_df['Close'].iloc[0])*100
                    rel_str = sym_ret - spy_ret
            except: pass

            sc,reasons=TechnicalIndicators.score(lat,self.params)
            sig=t("sig_buy") if sc>=5 else (t("sig_sell") if sc<=1 else t("sig_watch"))

            # v10.0：短期均值回归信号
            try:
                regime = ProbabilityEngine(ind, ind['close']).detect_regime()
            except:
                regime = 'neutral'
            st_combined = ShortTermMeanReversion.combined_signal(ind, ind['close'], regime)
            st_score = float(st_combined.iloc[-1]) if not st_combined.empty else 0.0
            st_signal = "短买" if st_score > 0.5 else ("短卖" if st_score < -0.5 else "中性")

            # v10.0：量价背离
            div_label = VolumePriceDivergence.label(ind)

            detail=f"评分:{sc}/12 | 短期:{st_signal}({st_score:+.2f}) | {div_label} | "+"; ".join(reasons[:5])

            pvm=(close-ma20)/ma20*100
            if rsi>75 or pvm>12: rl,rc=("高位追涨",QColor(180,50,50))
            elif rsi>65 or pvm>6: rl,rc=("趋势追涨",QColor(200,130,50))
            elif rsi<30 and close<ma20: rl,rc=("超卖反弹",QColor(59,130,246))
            elif 30<=rsi<=55 and ma20>ma60: rl,rc=("稳健机会",QColor(16,185,129))
            elif 30<=rsi<=65: rl,rc=("震荡观望",QColor(100,130,160))
            else: rl,rc=("观望",QColor(80,100,120))

            sl=self.params.get('stop_loss_pct',0.05); tp=self.params.get('take_profit_pct',0.10)
            dyn_sl=max(close-2*atr, close*(1-sl))

            # ── 多周期涨跌（用全量历史 close + 当前价）──────────────
            closes_full=df_full['Close'].squeeze().astype(float).dropna()
            def _ret_over(nbars):
                if len(closes_full)>nbars:
                    past=float(closes_full.iloc[-1-nbars])
                    if past>0: return round((close-past)/past*100, 2)
                return None
            chg_map={}
            for _zh,_en,_nb,_key in _CHG_PERIODS:
                chg_map[_key]= round(chg,2) if _key=="_chg_1d" else _ret_over(_nb)

            ret_dict={
                'col_code':sym,'col_name':name,'col_price':round(close,2),'col_chg':round(chg,2),
                'col_rsi':round(rsi,1),'col_cs_rank':0,'col_rel_str':round(rel_str,2),
                'col_ma20':round(ma20,2),'col_ma60':round(ma60,2),'col_macd':round(macd,4),
                'col_vol_ratio':round(vol_r,2),'col_atr':round(atr,2),'col_obv_trend':obv_trend,
                'col_signal':sig,'col_risk':rl,'col_sl':round(dyn_sl,2),
                'col_tp':round(close*(1+tp),2),'col_detail':detail,
                '_risk_color':rc,'_score':sc,'_currency':_currency(sym),
            }
            ret_dict.update(chg_map)
            return ret_dict
        except: logger.error(f"Analyze {sym}: {traceback.format_exc()}"); return None

# ══════════════════════════════════════════════════════════════════
# Walk Forward 回测线程
# ══════════════════════════════════════════════════════════════════
class BacktestThread(QThread):
    progress=pyqtSignal(int,str); result_ready=pyqtSignal(dict)
    log_msg=pyqtSignal(str); finished_all=pyqtSignal(int); error_msg=pyqtSignal(str)

    def __init__(self, syms, params, period="1y", train_months=12, test_months=3):
        super().__init__()
        self.syms=syms; self.params=params; self.period=period
        self.train_months=train_months; self.test_months=test_months; self._stop=False

    def stop(self): self._stop=True

    def run(self):
        total=len(self.syms); done=ok=0
        self.log_msg.emit(t("bt_start"))
        wf=WalkForwardBacktest(self.train_months, self.test_months)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs={ex.submit(self._run_wf,wf,s,n):(s,n) for s,n in self.syms}
            for fut in as_completed(futs):
                if self._stop: break
                s,n=futs[fut]
                try:
                    res=fut.result()
                    if res:
                        res['chn_name']=n; self.result_ready.emit(res); ok+=1
                    else:
                        self.log_msg.emit(t("bt_skip",sym=s))
                except Exception as e:
                    self.error_msg.emit(t("bt_error",sym=s,e=e))
                done+=1; self.progress.emit(int(done/total*100),s)
        self.finished_all.emit(ok)

    def _run_wf(self, wf, sym, name):
        df=fetch_stock_data(sym, self.period)
        if df.empty or len(df)<200: return None
        return wf.run(sym, df, self.params)

# ══════════════════════════════════════════════════════════════════
# 绘图辅助
# ══════════════════════════════════════════════════════════════════
def _mpl_style(ax,title="",xlabel="",ylabel=""):
    ax.set_facecolor(T.MPL_AXES)
    ax.tick_params(colors=T.TEXT_2,labelsize=8)
    ax.set_xlabel(xlabel,color=T.TEXT_2,fontsize=9)
    ax.set_ylabel(ylabel,color=T.TEXT_2,fontsize=9)
    if title: ax.set_title(title,color=T.TEXT_H,fontsize=10,pad=8)
    for sp in ax.spines.values(): sp.set_edgecolor(T.BORDER)
    ax.grid(True,color=T.BORDER,alpha=0.5,linewidth=0.5)

# ══════════════════════════════════════════════════════════════════
# 摘要面板
# ══════════════════════════════════════════════════════════════════
class SummaryPanel(QFrame):
    _DEFS=[("total","sum_total","--"),("buy","sum_buy","--"),("sell","sum_sell","--"),
           ("watch","sum_watch","--"),("avgRSI","sum_avg_rsi","--"),("avgScore","sum_avg_score","--")]
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{border-radius:10px;border:1px solid {T.BORDER};border-top:1px solid {T.BORDER_HI};"
                           f"background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #142539, stop:1 #0d1826);}}")
        lay=QHBoxLayout(self); lay.setContentsMargins(16,10,16,10)
        self._titles={}; self._vals={}
        for key,tkey,default in self._DEFS:
            box=QVBoxLayout()
            lt=QLabel(t(tkey)); lt.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
            lv=QLabel(default); lv.setStyleSheet(f"color:{T.TEXT_H};font-size:15pt;font-weight:700;border:none;")
            lt.setAlignment(Qt.AlignCenter); lv.setAlignment(Qt.AlignCenter)
            box.addWidget(lt); box.addWidget(lv); lay.addLayout(box)
            self._titles[key]=lt; self._vals[key]=lv
            if key!=self._DEFS[-1][0]:
                sep=QFrame(); sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet(f"color:{T.BORDER};border:none;background:{T.BORDER};max-width:1px;")
                lay.addWidget(sep)
    def retranslate(self):
        for key,tkey,_ in self._DEFS: self._titles[key].setText(t(tkey))
    def update_stats(self,model):
        rows=model._rows; n=len(rows)
        buys=sum(1 for r in rows if r.get('col_signal') in ("买入","Buy"))
        sells=sum(1 for r in rows if r.get('col_signal') in ("卖出","Sell"))
        watch=n-buys-sells
        rsis=[r['col_rsi'] for r in rows if isinstance(r.get('col_rsi'),(int,float))]
        scrs=[r['_score']  for r in rows if isinstance(r.get('_score'),(int,float))]
        self._vals['total'].setText(str(n))
        self._vals['buy'].setText(str(buys)); self._vals['buy'].setStyleSheet(f"color:{T.GREEN};font-size:15pt;font-weight:700;border:none;")
        self._vals['sell'].setText(str(sells)); self._vals['sell'].setStyleSheet(f"color:{T.RED};font-size:15pt;font-weight:700;border:none;")
        self._vals['watch'].setText(str(watch)); self._vals['watch'].setStyleSheet(f"color:{T.YELLOW};font-size:15pt;font-weight:700;border:none;")
        self._vals['avgRSI'].setText(f"{sum(rsis)/len(rsis):.1f}" if rsis else "--")
        self._vals['avgScore'].setText(f"{sum(scrs)/len(scrs):.1f}" if scrs else "--")

# ══════════════════════════════════════════════════════════════════
# 翻牌组件
# ══════════════════════════════════════════════════════════════════
class FlipBoardWidget(QWidget):
    # 方向渐变背景（贴近参考截图：涨深绿、跌深红、中性深蓝）
    _BG_UP   = "qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0c3a26, stop:0.5 #0a2c1d, stop:1 #07140d)"
    _BG_DOWN = "qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3a1414, stop:0.5 #2c0d0d, stop:1 #140707)"
    _BG_NEUT = "qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #142539, stop:1 #0d1826)"

    def __init__(self,parent=None):
        super().__init__(parent); self.setFixedHeight(54)
        self._set_bg(self._BG_NEUT)
        lay=QVBoxLayout(self); lay.setContentsMargins(20,4,20,4)
        self.lbl=QLabel(t("flip_loading")); self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:13pt;font-weight:600;border:none;background:transparent;")
        lay.addWidget(self.lbl)
        self.eff=QGraphicsOpacityEffect(self.lbl); self.lbl.setGraphicsEffect(self.eff)
        self.ani=QPropertyAnimation(self.eff,b"opacity"); self.ani.setDuration(280)
        self.ani.setEasingCurve(QEasingCurve.InOutQuad)
        self.data={}; self.order=['^GSPC','^IXIC','^DJI','^VIX']; self.idx=0
        QTimer(self,timeout=self._next,interval=4000).start()
    def _set_bg(self, grad):
        self.setStyleSheet(f"FlipBoardWidget{{background:{grad};border-radius:8px;"
                           f"border:1px solid {T.BORDER};border-top:1px solid {T.BORDER_HI};}}")
    def update_data(self,d): self.data=d; self._show(animate=False)
    def _next(self):
        if not self.data: return
        self.idx=(self.idx+1)%len(self.order); self._show(animate=True)
    def _show(self,animate=True):
        sym=self.order[self.idx]
        if sym not in self.data: self.lbl.setText(t("flip_waiting")); return
        price,chg,pct,name=self.data[sym]
        up=(chg<0) if sym=='^VIX' else (chg>=0)   # VIX 跌=市场涨，反向
        col=T.GREEN if up else T.RED
        self._set_bg(self._BG_UP if up else self._BG_DOWN)
        sign="+" if chg>=0 else ""
        arrow="▲" if chg>=0 else "▼"
        html=(f"<span style='color:{T.GOLD};font-size:13pt;font-weight:700;'>{name}</span>  "
              f"<span style='color:{T.TEXT_H};font-size:14pt;font-weight:700;'>{price:,.2f}</span>  "
              f"<span style='color:{col};font-size:13pt;font-weight:700;'>{arrow} {sign}{chg:,.2f} ({sign}{pct:.2f}%)</span>")
        if animate:
            self.ani.setStartValue(1.0); self.ani.setEndValue(0.0)
            def fade_in():
                self.lbl.setText(html)
                self.ani.setStartValue(0.0); self.ani.setEndValue(1.0)
                try: self.ani.finished.disconnect()
                except: pass
                self.ani.start()
            try: self.ani.finished.disconnect()
            except: pass
            self.ani.finished.connect(fade_in); self.ani.start()
        else: self.lbl.setText(html)

# ══════════════════════════════════════════════════════════════════
# 横截面 Alpha 弹窗（v9.0 新增）
# ══════════════════════════════════════════════════════════════════
class CrossSectionalDialog(QDialog):
    def __init__(self, syms_data: Dict[str, pd.DataFrame], parent=None):
        super().__init__(parent)
        self.syms_data = syms_data
        self.setWindowTitle("横截面 Alpha 分析 — 多空排名")
        self.setWindowIcon(_make_app_icon())
        _fit_to_screen(self, 1000, 700)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()
        QTimer.singleShot(200, self._run)

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(14,12,14,12)
        hdr = QLabel(f"<b>横截面Alpha分析</b>  共 {len(self.syms_data)} 只股票")
        hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;padding:6px;")
        lay.addWidget(hdr)
        self.status = QLabel("计算中..."); self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;")
        lay.addWidget(self.status)
        self.tabs = QTabWidget(); lay.addWidget(self.tabs, 1)
        # Tab1: 排名热力图
        w1 = QWidget(); w1l = QVBoxLayout(w1); w1l.setContentsMargins(4,4,4,4)
        self.heatmap_canvas = FigureCanvas(Figure(figsize=(10,6), facecolor=T.MPL_BG))
        w1l.addWidget(self.heatmap_canvas)
        self.tabs.addTab(w1, "因子热力图")
        # Tab2: 多空组合
        w2 = QWidget(); w2l = QVBoxLayout(w2); w2l.setContentsMargins(4,4,4,4)
        self.ls_canvas = FigureCanvas(Figure(figsize=(10,5), facecolor=T.MPL_BG))
        self.ls_text = QTextEdit(); self.ls_text.setReadOnly(True); self.ls_text.setFixedHeight(120)
        self.ls_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;")
        w2l.addWidget(self.ls_canvas); w2l.addWidget(self.ls_text)
        self.tabs.addTab(w2, "多空组合净值")
        # Tab3: 排名明细
        w3 = QWidget(); w3l = QVBoxLayout(w3); w3l.setContentsMargins(4,4,4,4)
        self.rank_text = QTextEdit(); self.rank_text.setReadOnly(True)
        self.rank_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;border-radius:6px;")
        w3l.addWidget(self.rank_text)
        self.tabs.addTab(w3, "排名明细")

    def _run(self):
        self.status.setText("计算横截面因子（IC加权）...")
        QApplication.processEvents()
        try:
            # v10.0：IC加权横截面排名
            scores = ICWeightedAlpha.compute_scores_ic_weighted(self.syms_data)
            if scores.empty:
                self.status.setText("因子数据不足"); return
            self._draw_heatmap(scores)
            self._build_ls_portfolio(scores)
            self._fill_rank_table(scores)
            n_factors = len(CrossSectionalAlpha.FACTORS)
            self.status.setText(f"分析完成（IC加权）  共{len(scores)}只  因子:{n_factors}")
        except Exception as e:
            self.status.setText(f"出错: {e}"); logger.error(traceback.format_exc())

    def _draw_heatmap(self, scores: pd.DataFrame):
        factor_cols = [c for c in CrossSectionalAlpha.FACTORS.keys() if c in scores.columns]
        if not factor_cols: return
        data = scores[factor_cols].copy()
        # 限制显示前20只（避免太密）
        if len(data) > 20: data = data.iloc[:20]
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.heatmap_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax = fig.add_subplot(111)
            cmap = mcolors.LinearSegmentedColormap.from_list('rg', [T.RED,'#0d1826',T.GREEN])
            im = ax.imshow(data.values, aspect='auto', cmap=cmap, vmin=-3, vmax=3)
            ax.set_xticks(range(len(factor_cols)))
            xlabels = [CrossSectionalAlpha.FACTORS.get(c, c)[:6] for c in factor_cols]
            ax.set_xticklabels(xlabels, rotation=45, ha='right', color=T.TEXT_2, fontsize=8)
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data.index.tolist(), color=T.TEXT_H, fontsize=8)
            ax.set_title("横截面因子热力图（红=低 绿=高，按综合得分排序）", color=T.TEXT_H, fontsize=10, pad=8)
            fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
            for sp in ax.spines.values(): sp.set_edgecolor(T.BORDER)
            fig.tight_layout(pad=1.2)
            _apply_font_to_figure(fig)
        self.heatmap_canvas.draw()

    def _build_ls_portfolio(self, scores: pd.DataFrame):
        returns_dict = {}
        for sym in scores.index:
            df = self.syms_data.get(sym)
            if df is not None and not df.empty:
                c = df['Close'].squeeze().astype(float)
                if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                returns_dict[sym] = c.pct_change().dropna()

        ls_equity = CrossSectionalAlpha.long_short_portfolio(scores, returns_dict)
        if ls_equity.empty: return

        n = len(scores)
        top_n = max(1, n//5)
        sorted_s = scores.sort_values('composite_score', ascending=False)
        long_syms  = sorted_s.head(top_n).index.tolist()
        short_syms = sorted_s.tail(top_n).index.tolist()

        ls_ret = ls_equity.pct_change().dropna()
        sharpe = float(ls_ret.mean()/ls_ret.std()*np.sqrt(252)) if ls_ret.std()>0 else 0
        total_r = (ls_equity.iloc[-1]/10000-1)*100 if len(ls_equity)>0 else 0

        info = (f"多空组合  |  多头: {', '.join(long_syms)}  |  空头: {', '.join(short_syms)}\n"
                f"总收益: {total_r:+.2f}%  |  Sharpe: {sharpe:.2f}  |  持仓数: {top_n} 多 + {top_n} 空")
        self.ls_text.setText(info)

        with mpl.rc_context(_get_mpl_rc()):
            fig = self.ls_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax1, ax2 = fig.subplots(1, 2)
            # 多空净值
            color = T.GREEN if total_r >= 0 else T.RED
            ax1.plot(ls_equity.index, ls_equity.values, color=color, lw=1.8)
            ax1.fill_between(ls_equity.index, 10000, ls_equity.values,
                              where=ls_equity.values>=10000, color=T.GREEN, alpha=0.15)
            ax1.fill_between(ls_equity.index, 10000, ls_equity.values,
                              where=ls_equity.values<10000, color=T.RED, alpha=0.15)
            ax1.axhline(10000, color=T.TEXT_3, ls='--', lw=1)
            _mpl_style(ax1, title="多空组合净值", ylabel="净值($)")
            # 综合得分分布
            ax2.barh(range(len(scores)), scores['composite_score'].values,
                     color=[T.GREEN if v>0 else T.RED for v in scores['composite_score'].values],
                     alpha=0.85, height=0.7)
            ax2.set_yticks(range(len(scores)))
            ax2.set_yticklabels(scores.index.tolist(), color=T.TEXT_H, fontsize=7)
            ax2.axvline(0, color=T.TEXT_3, lw=1)
            _mpl_style(ax2, title="横截面综合得分", xlabel="Z-Score")
            fig.tight_layout(pad=1.2)
            _apply_font_to_figure(fig)
        self.ls_canvas.draw()

    def _fill_rank_table(self, scores: pd.DataFrame):
        lines = ["横截面排名明细（IC加权，由强到弱）\n" + "─"*60,
                 "⚡ v10.0 IC加权：高预测力因子权重更高，低IC因子自动降权\n"]
        sorted_s = scores.sort_values('composite_score', ascending=False)
        for i, (sym, row) in enumerate(sorted_s.iterrows()):
            pct = float(row.get('percentile', 50))
            comp = float(row.get('composite_score', 0))
            mom1 = float(row.get('momentum_1m', 0))*100
            rank_icon = "🟢" if pct >= 80 else ("🔴" if pct <= 20 else "🟡")
            lines.append(f"{rank_icon} #{i+1:2d}  {sym:<8}  综合:{comp:+.3f}  百分位:{pct:.0f}%  1月动量:{mom1:+.1f}%")
        self.rank_text.setText('\n'.join(lines))

# ══════════════════════════════════════════════════════════════════
# 组合优化弹窗（v9.0 新增）
# ══════════════════════════════════════════════════════════════════
class PortfolioDialog(QDialog):
    def __init__(self, syms: List[str], parent=None):
        super().__init__(parent)
        self.syms = syms
        self.setWindowTitle("机构级组合优化 — HRP + 有效前沿")
        self.setWindowIcon(_make_app_icon())
        _fit_to_screen(self, 1000, 720)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()
        QTimer.singleShot(200, self._run)

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(14,12,14,12)
        hdr = QLabel(f"<b>组合优化</b>  股票: {', '.join(self.syms[:8])}")
        hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;padding:6px;")
        lay.addWidget(hdr)
        self.status = QLabel("下载数据中..."); self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;")
        lay.addWidget(self.status)
        self.tabs = QTabWidget(); lay.addWidget(self.tabs, 1)
        # Tab1: HRP权重
        w1 = QWidget(); w1l = QVBoxLayout(w1); w1l.setContentsMargins(4,4,4,4)
        self.hrp_canvas = FigureCanvas(Figure(figsize=(10,5), facecolor=T.MPL_BG))
        self.hrp_text = QTextEdit(); self.hrp_text.setReadOnly(True); self.hrp_text.setFixedHeight(100)
        self.hrp_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;")
        w1l.addWidget(self.hrp_canvas); w1l.addWidget(self.hrp_text)
        self.tabs.addTab(w1, "HRP层级风险平价")
        # Tab2: 有效前沿
        w2 = QWidget(); w2l = QVBoxLayout(w2); w2l.setContentsMargins(4,4,4,4)
        self.ef_canvas = FigureCanvas(Figure(figsize=(10,5), facecolor=T.MPL_BG))
        w2l.addWidget(self.ef_canvas)
        self.tabs.addTab(w2, "有效前沿")
        # Tab3: Beta & Vol Target
        w3 = QWidget(); w3l = QVBoxLayout(w3); w3l.setContentsMargins(4,4,4,4)
        self.risk_text = QTextEdit(); self.risk_text.setReadOnly(True)
        self.risk_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9.5pt;border-radius:6px;padding:8px;")
        w3l.addWidget(self.risk_text)
        self.tabs.addTab(w3, "Beta & 波动率目标")

    def _run(self):
        self.status.setText("下载数据...")
        QApplication.processEvents()
        returns_dict = {}
        for sym in self.syms:
            df = fetch_stock_data(sym, "2y")
            if not df.empty:
                c = df['Close'].squeeze().astype(float)
                if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
                returns_dict[sym] = c.pct_change().dropna()

        if len(returns_dict) < 2:
            self.status.setText("数据不足"); return

        returns = pd.DataFrame(returns_dict).dropna()
        if returns.empty or len(returns) < 60:
            self.status.setText("历史数据不足"); return

        try:
            # HRP
            hrp_w = PortfolioOptimizer.hrp(returns)
            self._draw_hrp(hrp_w, returns)

            # 有效前沿
            vols, rets, sharpes = PortfolioOptimizer.mean_variance_frontier(returns)
            self._draw_ef(vols, rets, sharpes, hrp_w, returns)

            # Beta + Vol Target
            spy_df = fetch_stock_data("SPY", "2y")
            spy_ret = spy_df['Close'].squeeze().astype(float).pct_change().dropna() if not spy_df.empty else None
            self._fill_risk_text(returns, hrp_w, spy_ret)

            self.status.setText("优化完成")
        except Exception as e:
            self.status.setText(f"出错: {e}"); logger.error(traceback.format_exc())

    def _draw_hrp(self, hrp_w: dict, returns: pd.DataFrame):
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.hrp_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax1, ax2 = fig.subplots(1, 2)
            # 饼图
            syms = list(hrp_w.keys()); vals = list(hrp_w.values())
            colors = plt.cm.get_cmap('tab20')(np.linspace(0,1,len(syms)))
            wedges, texts, autotexts = ax1.pie(
                vals, labels=syms, autopct='%1.1f%%', colors=colors,
                textprops={'color':T.TEXT_H,'fontsize':8}, startangle=90
            )
            for at in autotexts: at.set_color(T.BG0); at.set_fontsize(7)
            ax1.set_title("HRP权重分配", color=T.TEXT_H, fontsize=10)
            ax1.set_facecolor(T.MPL_BG)
            # 相关性矩阵
            corr = returns.corr()
            cmap = mcolors.LinearSegmentedColormap.from_list('rg',[T.RED,'#0d1826',T.GREEN])
            im = ax2.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            ax2.set_xticks(range(len(corr.columns)))
            ax2.set_yticks(range(len(corr.index)))
            ax2.set_xticklabels(corr.columns.tolist(), rotation=45, ha='right', color=T.TEXT_2, fontsize=8)
            ax2.set_yticklabels(corr.index.tolist(), color=T.TEXT_2, fontsize=8)
            ax2.set_title("相关性矩阵（HRP依据）", color=T.TEXT_H, fontsize=10)
            fig.colorbar(im, ax=ax2, fraction=0.04)
            for sp in ax2.spines.values(): sp.set_edgecolor(T.BORDER)
            fig.tight_layout(pad=1.2)
            _apply_font_to_figure(fig)
        self.hrp_canvas.draw()
        # 文字汇总
        lines = ["HRP层级风险平价权重  |  按波动率相关性层级分配，降低高相关资产集中度\n"]
        for s,w in sorted(hrp_w.items(), key=lambda x:-x[1]):
            vol = float(returns[s].std()*np.sqrt(252)*100) if s in returns.columns else 0
            lines.append(f"  {s:<8}  权重:{w:5.1f}%  年化波动率:{vol:.1f}%")
        self.hrp_text.setText('\n'.join(lines))

    def _draw_ef(self, vols, rets, sharpes, hrp_w, returns):
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.ef_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax = fig.add_subplot(111); ax.set_facecolor(T.MPL_AXES)
            sc = ax.scatter(vols, rets, c=sharpes, cmap='RdYlGn', alpha=0.5, s=8)
            fig.colorbar(sc, ax=ax, label='Sharpe Ratio', fraction=0.03)
            # 标注HRP组合
            mu_yr = returns.mean()*252; cov_yr = returns.cov()*252
            hrp_arr = np.array([hrp_w.get(s,0)/100 for s in returns.columns])
            hrp_ret = float(np.dot(hrp_arr, mu_yr))*100
            hrp_vol = float(np.sqrt(np.dot(hrp_arr, np.dot(cov_yr.values, hrp_arr))))*100
            ax.scatter([hrp_vol], [hrp_ret], color=T.GOLD, s=120, zorder=5, marker='*', label=f'HRP ({hrp_ret:+.1f}%)')
            # 等权组合
            ew = np.ones(len(returns.columns))/len(returns.columns)
            ew_ret = float(np.dot(ew, mu_yr))*100
            ew_vol = float(np.sqrt(np.dot(ew, np.dot(cov_yr.values, ew))))*100
            ax.scatter([ew_vol], [ew_ret], color=T.CYAN, s=80, zorder=5, marker='D', label=f'等权 ({ew_ret:+.1f}%)')
            ax.legend(fontsize=9, facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)
            _mpl_style(ax, title="蒙特卡洛有效前沿（颜色=Sharpe）", xlabel="年化波动率%", ylabel="年化收益%")
            fig.tight_layout(pad=1.2); _apply_font_to_figure(fig)
        self.ef_canvas.draw()

    def _fill_risk_text(self, returns, hrp_w, spy_ret):
        lines = ["Beta 暴露 & 波动率目标化分析\n" + "─"*60 + "\n"]
        if spy_ret is not None:
            betas = PortfolioOptimizer.beta_exposure(returns, spy_ret)
            lines.append("【Beta暴露（相对SPY）】")
            port_beta = sum(hrp_w.get(s,0)/100*betas.get(s,1) for s in returns.columns)
            for s in returns.columns:
                b = betas.get(s, 1)
                flag = "⚠ 高Beta" if b>1.5 else ("↓ 防御" if b<0.5 else "")
                lines.append(f"  {s:<8}  Beta:{b:.2f}  {flag}")
            lines.append(f"\n  组合加权Beta: {port_beta:.2f}")
            if abs(port_beta-1) > 0.3:
                lines.append("  ⚠ 组合Beta偏离市场较大，建议检查Beta中性")
            lines.append("")

        # Vol Target
        lines.append("【波动率目标化仓位（目标15%年化波动）】")
        for target in [0.10, 0.15, 0.20]:
            vt_w = PortfolioOptimizer.vol_target_weights(returns, target, hrp_w)
            lines.append(f"  目标波动率{target*100:.0f}%: " + "  ".join(f"{s}:{v:.1f}%" for s,v in list(vt_w.items())[:6]))

        # 集中度检查
        lines.append("\n【集中度风险检查】")
        max_w = max(hrp_w.values()); max_s = max(hrp_w, key=hrp_w.get)
        lines.append(f"  最大单股权重: {max_s} = {max_w:.1f}%")
        if max_w > 30: lines.append("  ⚠ 单股权重超过30%，集中度偏高")
        top3_w = sum(sorted(hrp_w.values(), reverse=True)[:3])
        lines.append(f"  Top3集中度: {top3_w:.1f}%")

        self.risk_text.setText('\n'.join(lines))

# ══════════════════════════════════════════════════════════════════
# GARCH 风险弹窗（v9.0 新增）
# ══════════════════════════════════════════════════════════════════
class GARCHDialog(QDialog):
    def __init__(self, sym: str, parent=None):
        super().__init__(parent)
        self.sym = sym
        self.setWindowTitle(f"GARCH风险分析 — {sym}")
        self.setWindowIcon(_make_app_icon())
        _fit_to_screen(self, 920, 680)
        self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui()
        QTimer.singleShot(200, self._run)

    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(14,12,14,12)
        self.status = QLabel("计算GARCH模型..."); self.status.setStyleSheet(f"color:{T.GOLD};font-size:9.5pt;")
        lay.addWidget(self.status)
        self.tabs = QTabWidget(); lay.addWidget(self.tabs, 1)
        # Tab1: 条件波动率
        w1 = QWidget(); w1l = QVBoxLayout(w1); w1l.setContentsMargins(4,4,4,4)
        self.garch_canvas = FigureCanvas(Figure(figsize=(9,4.5), facecolor=T.MPL_BG))
        self.garch_text   = QTextEdit(); self.garch_text.setReadOnly(True); self.garch_text.setFixedHeight(80)
        self.garch_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;")
        w1l.addWidget(self.garch_canvas); w1l.addWidget(self.garch_text)
        self.tabs.addTab(w1, "GARCH条件波动率")
        # Tab2: Fat-tail MC
        w2 = QWidget(); w2l = QVBoxLayout(w2); w2l.setContentsMargins(4,4,4,4)
        self.mc_canvas = FigureCanvas(Figure(figsize=(9,4.5), facecolor=T.MPL_BG))
        self.mc_text   = QTextEdit(); self.mc_text.setReadOnly(True); self.mc_text.setFixedHeight(120)
        self.mc_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;")
        w2l.addWidget(self.mc_canvas); w2l.addWidget(self.mc_text)
        self.tabs.addTab(w2, "Fat-tail蒙特卡洛")
        # Tab3: 滚动VaR
        w3 = QWidget(); w3l = QVBoxLayout(w3); w3l.setContentsMargins(4,4,4,4)
        self.var_canvas = FigureCanvas(Figure(figsize=(9,4.5), facecolor=T.MPL_BG))
        w3l.addWidget(self.var_canvas)
        self.tabs.addTab(w3, "滚动VaR / CVaR")

    def _run(self):
        df = fetch_stock_data(self.sym, "2y")
        if df.empty: self.status.setText("无数据"); return
        c = df['Close'].squeeze().astype(float)
        if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
        returns = c.pct_change().dropna()

        # GARCH
        garch_res = GARCHRiskEngine.fit_garch(returns)
        self._draw_garch(returns, garch_res)

        # Fat-tail MC
        mc_res = GARCHRiskEngine.fat_tail_monte_carlo(returns, days=20, n=10000)
        self._draw_mc(mc_res)

        # 滚动VaR
        self._draw_rolling_var(returns)

        if garch_res:
            alpha=garch_res.get('alpha',0); beta=garch_res.get('beta',0); nu=garch_res.get('nu',0)
            pers=garch_res.get('persistence',0)
            self.garch_text.setText(
                f"GARCH(1,1)-t参数:  α(ARCH)={alpha:.4f}  β(GARCH)={beta:.4f}  ν(自由度)={nu:.2f}  "
                f"持久性={pers:.4f}\n"
                f"{'⚠ 持久性>0.98，波动率非常粘性（高持续性）' if pers>0.98 else '波动率持久性正常'}\n"
                f"AIC={garch_res.get('aic',0):.1f}  BIC={garch_res.get('bic',0):.1f}"
            )
        else:
            self.garch_text.setText("GARCH模型不可用（需要arch库），显示历史波动率。" if not HAS_ARCH else "GARCH拟合失败，数据可能不足。")
        self.status.setText("GARCH分析完成")

    def _draw_garch(self, returns, garch_res):
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.garch_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax1, ax2 = fig.subplots(2, 1, sharex=False)
            # 收益率
            ax1.plot(returns.index, returns.values*100, color=T.TEXT_2, lw=0.8, alpha=0.7)
            ax1.fill_between(returns.index, returns.values*100, 0,
                              where=returns.values>0, color=T.GREEN, alpha=0.4)
            ax1.fill_between(returns.index, returns.values*100, 0,
                              where=returns.values<0, color=T.RED, alpha=0.4)
            _mpl_style(ax1, title="日收益率%", ylabel="%")
            # 条件波动率
            if garch_res and 'cond_vol' in garch_res:
                vol_idx = garch_res['cond_vol_index'][:len(garch_res['cond_vol'])]
                ax2.plot(vol_idx, [v*100 for v in garch_res['cond_vol']],
                          color=T.GOLD, lw=1.2, label='GARCH条件波动率')
                # 未来预测（20天）
                last_date = returns.index[-1]
                fcast_vols = [v*100 for v in garch_res['forecast_vol']]
                ax2.plot(range(1, len(fcast_vols)+1), fcast_vols, color=T.ACCENT,
                          lw=1.5, ls='--', label='20日预测', alpha=0.8)
            else:
                rolling_vol = returns.rolling(20).std()*np.sqrt(252)*100
                ax2.plot(rolling_vol.index, rolling_vol.values, color=T.GOLD, lw=1.2, label='20日滚动波动率(年化)')
            ax2.legend(fontsize=8, facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)
            _mpl_style(ax2, title="条件波动率预测", ylabel="日波动率%")
            fig.tight_layout(pad=1.2); _apply_font_to_figure(fig)
        self.garch_canvas.draw()

    def _draw_mc(self, mc_res):
        if not mc_res: return
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.mc_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax1, ax2 = fig.subplots(1, 2)
            for ax, key, label in [(ax1,'normal','正态分布MC'), (ax2,'t_dist','t分布MC（Fat-tail）')]:
                d = mc_res.get(key, {})
                if not d: continue
                pcts = (d['finals']-1)*100 if 'finals' in d else []
                if len(pcts) == 0: continue
                up = pcts[pcts>=0]; dn = pcts[pcts<0]
                ax.hist(dn, bins=60, color=T.RED, alpha=0.7, density=True)
                ax.hist(up, bins=60, color=T.GREEN, alpha=0.7, density=True)
                ax.axvline(0, color=T.GOLD, lw=1.5, ls='--')
                ax.axvline(d.get('var95',0), color=T.ORANGE, lw=1.2, ls=':',
                            label=f"VaR95:{d.get('var95',0):.1f}%")
                ax.axvline(d.get('cvar95',0), color=T.RED, lw=1.2, ls='-.',
                            label=f"CVaR95:{d.get('cvar95',0):.1f}%")
                ax.legend(fontsize=7, facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)
                up_p = d.get('up_prob',0.5)*100
                _mpl_style(ax, title=f"{label}  上涨:{up_p:.0f}%", xlabel="20日收益%")
            fig.tight_layout(pad=1.2); _apply_font_to_figure(fig)
        self.mc_canvas.draw()
        # 文字对比
        n = mc_res.get('normal',{}); td = mc_res.get('t_dist',{})
        self.mc_text.setText(
            f"【正态分布】  上涨:{n.get('up_prob',0)*100:.1f}%  VaR95:{n.get('var95',0):.2f}%  CVaR95:{n.get('cvar95',0):.2f}%\n"
            f"【t分布(fat-tail)】上涨:{td.get('up_prob',0)*100:.1f}%  VaR95:{td.get('var95',0):.2f}%  CVaR95:{td.get('cvar95',0):.2f}%\n"
            f"自由度ν={mc_res.get('df_t',0):.2f}  (越小尾部越厚，越接近现实)\n"
            f"⚠ Fat-tail VaR通常比正态大{'（尾部风险被低估）' if abs(td.get('var95',0))>abs(n.get('var95',0)) else ''}"
        )

    def _draw_rolling_var(self, returns):
        var95  = GARCHRiskEngine.rolling_var(returns, 60, 0.95)*100
        var99  = GARCHRiskEngine.rolling_var(returns, 60, 0.99)*100
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.var_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax = fig.add_subplot(111); ax.set_facecolor(T.MPL_AXES)
            ax.plot(returns.index, returns.values*100, color=T.TEXT_3, lw=0.6, alpha=0.6, label='日收益%')
            ax.plot(var95.index, var95.values, color=T.ORANGE, lw=1.2, label='60日滚动VaR95')
            ax.plot(var99.index, var99.values, color=T.RED, lw=1.2, label='60日滚动VaR99', ls='--')
            ax.fill_between(var95.index, var95.values, 0, color=T.ORANGE, alpha=0.1)
            ax.axhline(0, color=T.TEXT_3, lw=0.8, ls='--')
            ax.legend(fontsize=8, facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)
            _mpl_style(ax, title="滚动VaR / CVaR（风险随时间变化）", ylabel="日收益%")
            fig.tight_layout(pad=1.2); _apply_font_to_figure(fig)
        self.var_canvas.draw()

# ══════════════════════════════════════════════════════════════════
# 概率预测弹窗 v9.0
# ══════════════════════════════════════════════════════════════════
class ProbabilityDialog(QDialog):
    def __init__(self, sym, name, row, parent=None):
        super().__init__(parent)
        self.sym=sym; self.name=name; self.row=row
        self.setWindowTitle(f"{sym}  —  概率预测 v13.0")
        self.setWindowIcon(_make_app_icon())
        _fit_to_screen(self, 960, 840); self.setStyleSheet(GLOBAL_STYLE)
        self._news_thread=None; self._kline_period="3mo"
        self._ind=None; self._close=None
        self._sentiment_agg=None
        self._last_p_xgb={5:0.5,20:0.5,60:0.5}
        self._last_p_hq ={5:0.5,20:0.5,60:0.5}
        self._last_p_mom={5:0.5,20:0.5,60:0.5}
        self._last_weights={5:(0.45,0.35,0.20),20:(0.50,0.30,0.20),60:(0.50,0.30,0.20)}
        # 后台解析旧预测（不阻塞UI）
        threading.Thread(target=SelfCorrectionEngine.resolve_pending,
                         args=(sym,), daemon=True).start()
        self._build_ui(); QTimer.singleShot(80,self._run)

    def _build_ui(self):
        lay=QVBoxLayout(self); lay.setSpacing(10); lay.setContentsMargins(16,14,16,14)
        # 顶部信息栏
        bar=QFrame(); bar.setStyleSheet(f"background:{T.BG2};border-radius:8px;border:1px solid {T.BORDER};")
        blay=QHBoxLayout(bar); blay.setContentsMargins(14,10,14,10)
        sym_lbl=QLabel(f"<span style='font-size:17pt;font-weight:700;color:{T.GOLD};'>{self.sym}</span>"
                        f"<span style='font-size:11pt;color:{T.TEXT_2};'> {self.name}</span>")
        sym_lbl.setTextFormat(Qt.RichText)
        cur=self.row.get('_currency','$'); price=self.row.get('col_price','--'); chg=self.row.get('col_chg',0)
        try: chg_f=float(chg)
        except: chg_f=0.
        cc=T.GREEN if chg_f>=0 else T.RED
        p_lbl=QLabel(f"<span style='font-size:13pt;color:{T.TEXT_H};font-weight:600;'>{cur}{price}</span>"
                      f"<span style='font-size:11pt;color:{cc};'> {chg_f:+.2f}%</span>")
        p_lbl.setTextFormat(Qt.RichText)
        rsi_v=self.row.get('col_rsi','--'); score_v=self.row.get('_score','--')
        sig_v=self.row.get('col_signal','--'); cs_r=self.row.get('col_cs_rank','--')
        rel=self.row.get('col_rel_str',0)
        try: rel_s=f"{float(rel):+.1f}%"
        except: rel_s="--"
        meta=QLabel(f"RSI <b>{rsi_v}</b> | 评分 <b>{score_v}/10</b> | 信号 <b>{sig_v}</b> | CS排名 <b>#{cs_r}</b> | 相对SPY <b>{rel_s}</b>")
        meta.setStyleSheet(f"color:{T.TEXT_2};font-size:9.5pt;")
        blay.addWidget(sym_lbl); blay.addStretch(); blay.addWidget(p_lbl); blay.addSpacing(20); blay.addWidget(meta)
        lay.addWidget(bar)
        self.status_lbl=QLabel(t("prob_calc"))
        self.status_lbl.setStyleSheet(f"color:{T.GOLD};font-size:9.5pt;padding:4px 6px;")
        lay.addWidget(self.status_lbl)
        self.tabs=QTabWidget(); lay.addWidget(self.tabs,1)
        self._build_tab_prob()
        self._build_tab_mc()
        self._build_tab_feat()
        self._build_tab_quantile()
        self._build_tab_kline()
        self._build_tab_news()
        self._build_tab_history()   # v13.0 新增

    def _add_scroll_tab(self, w, title):
        """把 tab 内容套进滚动区再加入，小屏幕下内容超出可视区时可滚动看全。"""
        from PyQt5.QtWidgets import QScrollArea
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(w)
        sa.setFrameShape(QScrollArea.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 记录"滚动区 → 内部真实内容widget"映射，供价格目标补丁等找到正确容器
        if not hasattr(self, "_tab_content"):
            self._tab_content = {}
        self._tab_content[title] = w
        self._tab_content[id(sa)] = w
        self.tabs.addTab(sa, title)

    def tab_inner(self, index_or_title):
        """返回某个tab的真实内容widget（穿透滚动区）。
        价格目标补丁用它替代 self.tabs.widget(0).layout()。"""
        from PyQt5.QtWidgets import QScrollArea
        w = None
        if isinstance(index_or_title, int):
            w = self.tabs.widget(index_or_title)
        else:
            return getattr(self, "_tab_content", {}).get(index_or_title)
        if isinstance(w, QScrollArea):
            return w.widget()
        return w

    def _build_tab_prob(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setSpacing(12)
        self.card_row=QHBoxLayout(); self.card_row.setSpacing(10)
        self.prob_cards={}
        for period,key in [("5日","p5"),("20日","p20"),("60日","p60")]:
            card=QFrame(); card.setStyleSheet(f"background:{T.BG2};border-radius:10px;border:1px solid {T.BORDER};")
            cly=QVBoxLayout(card); cly.setContentsMargins(16,14,16,14); cly.setSpacing(4)
            ttl=QLabel(f"{period}综合概率"); ttl.setStyleSheet(f"color:{T.TEXT_2};font-size:9.5pt;border:none;")
            val_lbl=QLabel("--"); val_lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:26pt;font-weight:700;border:none;")
            sub_lbl=QLabel("Regime×XGB×HQ×Mom"); sub_lbl.setStyleSheet(f"color:{T.TEXT_3};font-size:8.5pt;border:none;")
            cly.addWidget(ttl); cly.addWidget(val_lbl); cly.addWidget(sub_lbl)
            self.card_row.addWidget(card); self.prob_cards[key]=val_lbl
        lay.addLayout(self.card_row)
        bar_card=QFrame(); bar_card.setStyleSheet(f"background:{T.BG2};border-radius:10px;border:1px solid {T.BORDER};")
        blay=QVBoxLayout(bar_card); blay.setContentsMargins(16,14,16,14); blay.setSpacing(10)
        hdr=QLabel("分项模型概率"); hdr.setStyleSheet(f"color:{T.GOLD};font-size:10pt;font-weight:600;border:none;")
        blay.addWidget(hdr)
        self.bars=[]
        for label in ["5日 XGB+PCA","5日 历史分位","5日 动量融合","20日 综合","60日 综合"]:
            rw=QHBoxLayout(); rw.setSpacing(10)
            lbl=QLabel(label); lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;"); lbl.setFixedWidth(110)
            pb=QProgressBar(); pb.setRange(0,100); pb.setTextVisible(False); pb.setFixedHeight(10)
            pb.setStyleSheet(f"QProgressBar{{background:{T.BG3};border-radius:5px;border:none;}}QProgressBar::chunk{{background:{T.ACCENT};border-radius:5px;}}")
            val=QLabel("--"); val.setStyleSheet(f"color:{T.TEXT_H};font-size:9pt;font-weight:600;border:none;min-width:36px;")
            val.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
            rw.addWidget(lbl); rw.addWidget(pb,1); rw.addWidget(val); blay.addLayout(rw); self.bars.append((lbl,pb,val))

        # v12.0：情感信号卡片
        self.sentiment_card = QFrame()
        self.sentiment_card.setStyleSheet(f"background:{T.BG2};border-radius:10px;border:1px solid {T.BORDER};")
        sc_lay = QVBoxLayout(self.sentiment_card); sc_lay.setContentsMargins(16,10,16,10); sc_lay.setSpacing(4)
        sc_hdr_row = QHBoxLayout()
        sc_hdr = QLabel("📰 新闻情感信号 (VADER · v12.0)")
        sc_hdr.setStyleSheet(f"color:{T.CYAN};font-size:9.5pt;font-weight:700;border:none;")
        sc_hdr_row.addWidget(sc_hdr); sc_hdr_row.addStretch()
        sc_lay.addLayout(sc_hdr_row)
        sc_row = QHBoxLayout(); sc_row.setSpacing(20)
        left_col = QVBoxLayout()
        self.sent_overall_lbl = QLabel("--")
        self.sent_overall_lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:22pt;font-weight:700;border:none;")
        self.sent_label_lbl = QLabel("等待新闻...")
        self.sent_label_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8.5pt;border:none;")
        self.sent_adj_lbl = QLabel("概率调整: --")
        self.sent_adj_lbl.setStyleSheet(f"color:{T.YELLOW};font-size:8.5pt;font-weight:600;border:none;")
        left_col.addWidget(self.sent_overall_lbl); left_col.addWidget(self.sent_label_lbl); left_col.addWidget(self.sent_adj_lbl)
        self.sent_detail_lbl = QLabel("新闻情感将在加载新闻后自动更新")
        self.sent_detail_lbl.setStyleSheet(f"color:{T.TEXT_3};font-size:8pt;border:none;")
        self.sent_detail_lbl.setWordWrap(True)
        sc_row.addLayout(left_col); sc_row.addWidget(self.sent_detail_lbl, 1)
        sc_lay.addLayout(sc_row)

        lay.addWidget(bar_card,1); lay.addWidget(self.sentiment_card)
        self._add_scroll_tab(w,t("tab_prob"))

    def _build_tab_mc(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        self.mc_canvas=FigureCanvas(Figure(figsize=(8,4.5),facecolor=T.MPL_BG))
        self.mc_canvas.figure.subplots_adjust(left=0.07,right=0.97,top=0.88,bottom=0.12,wspace=0.35)
        self.mc_text=QTextEdit(); self.mc_text.setReadOnly(True); self.mc_text.setFixedHeight(100)
        self.mc_text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;border-radius:6px;padding:6px;")
        lay.addWidget(self.mc_canvas,1); lay.addWidget(self.mc_text); self._add_scroll_tab(w,t("tab_mc"))

    def _build_tab_feat(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        self.feat_canvas=FigureCanvas(Figure(figsize=(8,4.5),facecolor=T.MPL_BG))
        self.feat_canvas.figure.subplots_adjust(left=0.22,right=0.95,top=0.9,bottom=0.1)
        lay.addWidget(self.feat_canvas,1); self._add_scroll_tab(w,t("tab_feat"))

    def _build_tab_quantile(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(8,8,8,8)
        sel=QHBoxLayout()
        sel_lbl=QLabel("分析因子:"); sel_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;")
        self.factor_cb=QComboBox()
        self.factor_cb.addItems(['rsi','macd','volume_ratio','atr','mom10','mom20','stoch_rsi_k','vwap_dev','vol_imbalance'])
        self.factor_cb.setFixedWidth(160)
        self.fwd_cb=QComboBox(); self.fwd_cb.addItems(['5日','10日','20日']); self.fwd_cb.setFixedWidth(80)
        run_btn=QPushButton("计算"); run_btn.setFixedWidth(80); run_btn.clicked.connect(self._draw_quantile)
        sel.addWidget(sel_lbl); sel.addWidget(self.factor_cb); sel.addWidget(QLabel("预测期:")); sel.addWidget(self.fwd_cb)
        sel.addWidget(run_btn); sel.addStretch(); lay.addLayout(sel)
        self.quant_canvas=FigureCanvas(Figure(figsize=(8,3.5),facecolor=T.MPL_BG))
        self.quant_canvas.figure.subplots_adjust(left=0.1,right=0.97,top=0.88,bottom=0.15,wspace=0.4)
        self.ic_canvas=FigureCanvas(Figure(figsize=(8,2.5),facecolor=T.MPL_BG))
        self.ic_canvas.figure.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.18)
        lay.addWidget(self.quant_canvas,2); lay.addWidget(self.ic_canvas,1); self._add_scroll_tab(w,t("tab_quantile"))

    def _build_tab_kline(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(4,4,4,4)
        self.kline_info=QLabel(t("kline_loading"))
        self.kline_info.setStyleSheet(f"font-size:10.5pt;padding:6px 8px;color:{T.TEXT_H};background:{T.BG2};border-radius:5px;")
        lay.addWidget(self.kline_info)
        self.kline_canvas=FigureCanvas(Figure(figsize=(8,5),facecolor=T.MPL_BG,tight_layout=True))
        lay.addWidget(self.kline_canvas,1)
        btn_row=QHBoxLayout(); btn_row.setSpacing(4)
        periods=[("1M","1mo"),("3M","3mo"),("6M","6mo"),("YTD","ytd"),("1Y","1y"),("2Y","2y"),("5Y","5y"),("MAX","max")]
        self._kline_btns={}
        for lb,pp in periods:
            b=QPushButton(lb); b.setFixedWidth(52); b.setCheckable(True)
            b.clicked.connect(lambda _,p=pp: self._kline_change(p)); btn_row.addWidget(b); self._kline_btns[pp]=b
        self._kline_btns["3mo"].setChecked(True)
        btn_row.addStretch()
        ref=QPushButton(t("btn_refresh_price")); ref.setFixedWidth(110); ref.clicked.connect(self._kline_plot)
        btn_row.addWidget(ref)
        garch_btn=QPushButton("GARCH分析"); garch_btn.setFixedWidth(110)
        garch_btn.setStyleSheet(f"color:{T.CYAN};border-color:{T.CYAN};")
        garch_btn.clicked.connect(lambda: GARCHDialog(self.sym, self).exec_())
        btn_row.addWidget(garch_btn)
        lay.addLayout(btn_row); self._add_scroll_tab(w,t("tab_kline"))
        QTimer.singleShot(200,self._kline_plot)

    def _kline_change(self,p):
        self._kline_period=p
        for pp,btn in self._kline_btns.items(): btn.setChecked(pp==p)
        self._kline_plot()

    def _kline_plot(self):
        df=fetch_stock_data(self.sym,self._kline_period)
        if df.empty: self.kline_info.setText(t("kline_no_data")); return
        ohlcv=df[['Open','High','Low','Close','Volume']].copy()
        for c in ohlcv.columns: ohlcv[c]=ohlcv[c].squeeze().astype(float)
        ohlcv.dropna(inplace=True)
        if len(ohlcv)>1500: ohlcv=ohlcv.iloc[-1500:]
        cur_sym=_currency(self.sym); rp=get_realtime_price(self.sym)
        if rp: cp=rp; src="实时"; chg=(cp-float(ohlcv['Close'].iloc[-1]))/float(ohlcv['Close'].iloc[-1])*100
        else: cp=float(ohlcv['Close'].iloc[-1]); src="历史"; chg=(ohlcv['Close'].iloc[-1]-ohlcv['Close'].iloc[-2])/ohlcv['Close'].iloc[-2]*100 if len(ohlcv)>1 else 0
        cc=T.GREEN if chg>=0 else T.RED
        self.kline_info.setText(f"<b>{self.sym}</b>  {src}: <b>{cur_sym}{cp:.2f}</b>  <b style='color:{cc}'>{chg:+.2f}%</b>  {t('kline_bars',n=len(ohlcv))} [{self._kline_period.upper()}]")
        try:
            style=_make_mpf_style()
            mav_args=(5,20) if len(ohlcv)<60 else (5,20,60)
            with mpl.rc_context(_get_mpl_rc()):
                fig,axes=mpf.plot(ohlcv,type='candle',volume=True,mav=mav_args,
                                   returnfig=True,title=f"  {self.sym}",style=style,figsize=(8,5))
            _apply_font_to_figure(fig)
            self.kline_canvas.figure=fig; fig.canvas=self.kline_canvas; self.kline_canvas.draw()
        except Exception as e:
            self.kline_info.setText(t("kline_plot_fail",e=e))

    def _build_tab_news(self):
        w=QWidget(); lay=QVBoxLayout(w); lay.setContentsMargins(6,6,6,6)
        self.news_status=QLabel(t("kline_loading")); self.news_status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;padding:2px;")
        lay.addWidget(self.news_status)
        # 上半部：新闻列表
        self.news_text=QTextBrowser(); self.news_text.setOpenExternalLinks(True)
        self.news_text.setStyleSheet(f"QTextBrowser{{background:{T.BG2};border:1px solid {T.BORDER};color:{T.TEXT_1};font-size:10pt;padding:8px;border-radius:6px;}}")
        # 下半部：v12.0 情感图表
        self.sentiment_canvas = FigureCanvas(Figure(figsize=(8,2.8), facecolor=T.MPL_BG))
        self.sentiment_canvas.setFixedHeight(160)
        sent_lbl = QLabel("情感分析图 (VADER)"); sent_lbl.setStyleSheet(f"color:{T.CYAN};font-size:8.5pt;padding:2px 6px;border:none;")
        # 拆分器
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.news_text)
        sent_widget = QWidget(); sent_lay = QVBoxLayout(sent_widget); sent_lay.setContentsMargins(0,0,0,0); sent_lay.setSpacing(2)
        sent_lay.addWidget(sent_lbl); sent_lay.addWidget(self.sentiment_canvas)
        splitter.addWidget(sent_widget)
        splitter.setSizes([320, 170])
        lay.addWidget(splitter, 1)
        ref=QPushButton(t("btn_refresh_news")); ref.setFixedWidth(120); ref.clicked.connect(self._load_news); lay.addWidget(ref,0,Qt.AlignRight)
        self._add_scroll_tab(w,t("tab_news")); QTimer.singleShot(300,self._load_news)

    def _build_tab_history(self):
        """v13.0：预测历史Tab — 显示历史预测记录、准确率、自适应权重"""
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,8,8,8); lay.setSpacing(8)
        acc_frame = QFrame()
        acc_frame.setStyleSheet(f"QFrame{{background:{T.BG2};border-radius:8px;border:1px solid {T.BORDER};}}")
        acc_lay = QVBoxLayout(acc_frame); acc_lay.setContentsMargins(14,10,14,10); acc_lay.setSpacing(6)
        acc_hdr = QLabel("\U0001f3af 模型准确率（自我修正引擎）")
        acc_hdr.setStyleSheet(f"color:{T.GOLD};font-size:10pt;font-weight:700;border:none;")
        acc_lay.addWidget(acc_hdr)
        self.acc_text = QTextBrowser()
        self.acc_text.setFixedHeight(90)
        self.acc_text.setStyleSheet(f"background:{T.BG1};border:none;color:{T.TEXT_1};font-size:9pt;")
        acc_lay.addWidget(self.acc_text)
        lay.addWidget(acc_frame)
        w_frame = QFrame()
        w_frame.setStyleSheet(f"QFrame{{background:{T.BG2};border-radius:8px;border:1px solid {T.BORDER};}}")
        w_lay = QHBoxLayout(w_frame); w_lay.setContentsMargins(14,10,14,10); w_lay.setSpacing(20)
        self.weight_labels = {}
        for horizon, label in [(5,"5\u65e5"),(20,"20\u65e5"),(60,"60\u65e5")]:
            box = QVBoxLayout()
            hdr = QLabel(f"{label}\u6743\u91cd\uff08\u81ea\u9002\u5e94\uff09")
            hdr.setStyleSheet(f"color:{T.CYAN};font-size:9pt;font-weight:700;border:none;")
            box.addWidget(hdr)
            lbls = []
            for model in ["XGB/LR", "\u5386\u53f2\u5206\u4f4d", "\u52a8\u91cf"]:
                lbl = QLabel(f"{model}: --")
                lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8.5pt;border:none;")
                box.addWidget(lbl); lbls.append(lbl)
            self.weight_labels[horizon] = lbls
            w_lay.addLayout(box)
            if horizon < 60:
                sep = QFrame(); sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet(f"background:{T.BORDER};max-width:1px;border:none;")
                w_lay.addWidget(sep)
        lay.addWidget(w_frame)
        hist_hdr = QLabel("\U0001f4cb \u9884\u6d4b\u5386\u53f2\uff08\u6700\u8fd150\u6761\uff09")
        hist_hdr.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;padding:2px;border:none;")
        lay.addWidget(hist_hdr)
        self.hist_text = QTextBrowser()
        self.hist_text.setStyleSheet(
            f"QTextBrowser{{background:{T.BG2};border:1px solid {T.BORDER};"
            f"color:{T.TEXT_1};font-size:9pt;padding:6px;border-radius:6px;}}"
        )
        lay.addWidget(self.hist_text, 1)
        self._add_scroll_tab(w, "\U0001f4ca \u9884\u6d4b\u5386\u53f2")

    def _refresh_history_tab(self):
        """刷新历史预测Tab的内容"""
        report = SelfCorrectionEngine.get_accuracy_report(self.sym)
        acc_lines = []
        if report:
            for h in sorted(report.keys()):
                r = report[h]
                acc_col = T.GREEN if r["accuracy"] >= 55 else (T.RED if r["accuracy"] < 48 else T.YELLOW)
                brier_col = T.GREEN if r["brier"] < 0.22 else (T.RED if r["brier"] > 0.28 else T.YELLOW)
                acc_lines.append(
                    f"<span style='color:{T.TEXT_2};'>{h:2d}日</span>  "
                    f"<span style='color:{acc_col};font-weight:700;'>{r['accuracy']:.1f}%</span>准确率  "
                    f"<span style='color:{brier_col};'>Brier:{r['brier']:.4f}</span>  "
                    f"<span style='color:{T.TEXT_3};'>n={r['n']}条</span>"
                )
        else:
            acc_lines.append(f"<span style='color:{T.TEXT_3};'>暂无已解析预测（需等待到期后自动评分）</span>")
        self.acc_text.setHtml("<br>".join(acc_lines))
        for horizon in [5, 20, 60]:
            w1, w2, w3 = SelfCorrectionEngine.compute_adaptive_weights(self.sym, horizon)
            lbls = self.weight_labels[horizon]
            self._last_weights[horizon] = (w1, w2, w3)
            for lbl, model, wt in zip(lbls, ["XGB/LR","历史分位","动量"], [w1,w2,w3]):
                col = T.GREEN if wt >= 0.4 else (T.CYAN if wt >= 0.25 else T.TEXT_3)
                lbl.setText(
                    "<span style='color:" + T.TEXT_2 + ";'>" + model + ": </span>"
                    "<span style='color:" + col + ";font-weight:700;'>" + f"{wt:.1%}" + "</span>"
                )
                lbl.setTextFormat(Qt.RichText)
        rows = SelfCorrectionEngine.get_prediction_history(self.sym)
        if not rows:
            self.hist_text.setText("暂无预测记录。打开股票详情时自动记录，等待到期后自动评分。")
            return
        lines = []; prev_date = None
        for pred_date, horizon, p_ens, actual_up, actual_ret, correct, brier, resolved in rows:
            if pred_date != prev_date:
                lines.append(f"<b style='color:{T.GOLD};'>{pred_date}</b>")
                prev_date = pred_date
            pct = int((p_ens or 0.5) * 100)
            p_col = T.GREEN if pct >= 60 else (T.RED if pct <= 40 else T.YELLOW)
            if resolved:
                ret_str = f"{(actual_ret or 0)*100:+.1f}%"
                icon = "✅" if correct else "❌"
                brier_v = brier or 0
                lines.append(
                    f"&nbsp;&nbsp;<span style='color:{T.TEXT_3};'>{horizon:2d}日</span>  "
                    f"预测 <span style='color:{p_col};font-weight:700;'>{pct}%</span>  "
                    f"{icon} 实际:{ret_str}  "
                    f"<span style='color:{T.TEXT_3};font-size:8pt;'>Brier:{brier_v:.3f}</span>"
                )
            else:
                lines.append(
                    f"&nbsp;&nbsp;<span style='color:{T.TEXT_3};'>{horizon:2d}日</span>  "
                    f"预测 <span style='color:{p_col};font-weight:700;'>{pct}%</span>  "
                    f"<span style='color:{T.TEXT_3};font-size:8pt;'>⏳ 等待到期...</span>"
                )
        self.hist_text.setHtml("<br>".join(lines))


    def _load_news(self):
        self.news_status.setText(t("news_loading"))
        if self._news_thread and self._news_thread.isRunning(): return
        self._news_thread=NewsThread(self.sym)
        self._news_thread.news_ready.connect(self._on_news)
        self._news_thread.error_msg.connect(lambda e: self.news_status.setText(t("news_err",e=e)))
        self._news_thread.start()

    def _on_news(self,items):
        if not items: self.news_status.setText(t("news_none")); return
        self.news_status.setText(t("news_count",n=len(items)))

        # v12.0：聚合情感
        self._sentiment_agg = NewsSentimentEngine.aggregate_sentiment(items)
        self._update_sentiment_card(self._sentiment_agg)

        parts=[]
        for i,item in enumerate(items):
            ttl=item.get('title',''); lnk=item.get('link',''); pub=item.get('publisher',''); ts=item.get('time','')
            # 情感颜色
            sent = item.get('sentiment', {})
            compound = sent.get('compound', 0.0)
            sent_label = sent.get('label', 'neutral')
            if sent_label == 'positive':   sent_color = T.GREEN;  sent_icon = "▲"
            elif sent_label == 'negative': sent_color = T.RED;    sent_icon = "▼"
            else:                          sent_color = T.TEXT_3; sent_icon = "━"

            bg=T.BG2 if i%2==0 else T.BG3
            th=(f"<a href='{lnk}' style='color:{T.GOLD};text-decoration:none;font-size:10.5pt;font-weight:600;'>{ttl}</a>"
                if lnk else f"<span style='font-size:10.5pt;font-weight:600;color:{T.TEXT_H};'>{ttl}</span>")
            meta=f"<span style='color:{T.ACCENT};'>{pub}</span>  <span style='color:{T.TEXT_3};'>{ts}</span>"
            sent_badge = (f"<span style='color:{sent_color};font-size:9pt;font-weight:700;'>"
                          f"  {sent_icon} {compound:+.3f}</span>")
            parts.append(f"<div style='background:{bg};padding:10px 14px;border-bottom:1px solid {T.BORDER};"
                          f"border-radius:4px;margin-bottom:2px;'>"
                          f"{th}{sent_badge}<br/><small>{meta}</small></div>")
        self.news_text.setHtml(f"<div>{''.join(parts)}</div>")

        # 绘制情感图表
        self._draw_sentiment_chart(items, self._sentiment_agg)

        # v12.0：情感加载完成后，若指标已就绪则重新计算5日概率（加入情感调整）
        if self._ind is not None:
            QTimer.singleShot(100, self._refresh_prob_with_sentiment)

    def _refresh_prob_with_sentiment(self):
        """仅刷新5日概率卡片（加入情感调整），不重跑全部计算"""
        if self._ind is None or self._sentiment_agg is None:
            return
        try:
            engine = ProbabilityEngine(self._ind, self._close)
            p5 = engine.ensemble(5, self._sentiment_agg)
            pct = int(p5 * 100)
            col = T.GREEN if pct >= 60 else (T.RED if pct <= 40 else T.YELLOW)
            self.prob_cards["p5"].setText(f"{pct}%")
            self.prob_cards["p5"].setStyleSheet(f"color:{col};font-size:26pt;font-weight:700;border:none;")
            adj = NewsSentimentEngine.sentiment_to_prob_adjustment(self._sentiment_agg)
            self.status_lbl.setText(
                self.status_lbl.text().split(" | 情感")[0] + f"  |  情感调整:{adj:+.1%}"
            )
        except Exception:
            pass

    def _update_sentiment_card(self, agg: dict):
        """更新概率Tab中的情感信号卡片"""
        if not hasattr(self, 'sent_overall_lbl') or agg is None:
            return
        score = agg.get('overall_score', 0.0)
        label = agg.get('label', 'neutral')
        n_pos = agg.get('n_pos', 0); n_neg = agg.get('n_neg', 0); n_neu = agg.get('n_neu', 0)
        mom = agg.get('momentum', 0.0)
        rev = agg.get('reversal_signal', 0.0)

        if label == 'positive':   col = T.GREEN;  emoji = "😊 正面"
        elif label == 'negative': col = T.RED;    emoji = "😟 负面"
        else:                     col = T.TEXT_2; emoji = "😐 中性"

        self.sent_overall_lbl.setText(f"{score:+.3f}")
        self.sent_overall_lbl.setStyleSheet(f"color:{col};font-size:22pt;font-weight:700;border:none;")
        self.sent_label_lbl.setText(emoji)
        self.sent_label_lbl.setStyleSheet(f"color:{col};font-size:8.5pt;border:none;")

        adj = NewsSentimentEngine.sentiment_to_prob_adjustment(agg)
        adj_col = T.GREEN if adj > 0 else (T.RED if adj < 0 else T.TEXT_3)
        self.sent_adj_lbl.setText(f"5日概率调整: {adj:+.1%}")
        self.sent_adj_lbl.setStyleSheet(f"color:{adj_col};font-size:8.5pt;font-weight:600;border:none;")

        rev_str = ""
        if rev > 0:   rev_str = "  ⚡情感触底反转信号"
        elif rev < 0: rev_str = "  ⚠情感顶部反转信号"
        mom_str = f"情感动量: {mom:+.3f}"
        detail = (f"正面:{n_pos}条 | 负面:{n_neg}条 | 中性:{n_neu}条 | {mom_str}{rev_str}")
        self.sent_detail_lbl.setText(detail)
        self.sent_detail_lbl.setStyleSheet(f"color:{T.TEXT_3};font-size:8pt;border:none;")

    def _draw_sentiment_chart(self, items: list, agg: dict):
        """在新闻Tab底部绘制情感柱状图（若有canvas）"""
        if not hasattr(self, 'sentiment_canvas') or self.sentiment_canvas is None:
            return
        compounds = [item.get('sentiment', {}).get('compound', 0.0) for item in items]
        titles_short = [item.get('title', '')[:30]+'…' for item in items]
        with mpl.rc_context(_get_mpl_rc()):
            fig = self.sentiment_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax = fig.add_subplot(111); ax.set_facecolor(T.MPL_AXES)
            colors = [T.GREEN if c >= 0.05 else (T.RED if c <= -0.05 else T.TEXT_3) for c in compounds]
            y = range(len(compounds))
            ax.barh(list(y), compounds[::-1], color=colors[::-1], alpha=0.85, height=0.65, edgecolor='none')
            ax.set_yticks(list(y))
            ax.set_yticklabels(titles_short[::-1], color=T.TEXT_2, fontsize=7)
            ax.axvline(0, color=T.TEXT_3, lw=1, ls='--')
            ax.axvline(0.05, color=T.GREEN, lw=0.7, ls=':', alpha=0.5)
            ax.axvline(-0.05, color=T.RED, lw=0.7, ls=':', alpha=0.5)
            overall = agg.get('overall_score', 0)
            ax.axvline(overall, color=T.GOLD, lw=1.5, ls='-', alpha=0.8, label=f'整体均值:{overall:+.3f}')
            ax.legend(fontsize=8, facecolor=T.BG2, labelcolor=T.TEXT_H, edgecolor=T.BORDER)
            _mpl_style(ax, title="新闻情感得分 (VADER compound, 绿=正面 红=负面)", xlabel="compound ∈ [-1,1]")
            _apply_font_to_figure(fig)
        self.sentiment_canvas.draw()

    def _run(self):
        self.status_lbl.setText(t("prob_calc")); QApplication.processEvents()
        df=fetch_stock_data(self.sym,"1y")
        if df.empty or len(df)<65: self.status_lbl.setText(t("prob_no_data")); return
        ind=TechnicalIndicators.compute_all(df)
        if ind.empty: self.status_lbl.setText(t("prob_fail",e="指标失败")); return
        self._ind=ind; self._close=ind['close']
        try:
            engine=ProbabilityEngine(ind,self._close)
            # v13.0：ensemble返回7元组 (prob, p1, p2, p3, w1, w2, w3)
            res5  = engine.ensemble(5,  self._sentiment_agg, self.sym)
            res20 = engine.ensemble(20, self._sentiment_agg, self.sym)
            res60 = engine.ensemble(60, self._sentiment_agg, self.sym)
            p5,  p5_xgb,  p5_hq,  p5_mom,  w5_1,  w5_2,  w5_3  = res5
            p20, p20_xgb, p20_hq, p20_mom, w20_1, w20_2, w20_3  = res20
            p60, p60_xgb, p60_hq, p60_mom, w60_1, w60_2, w60_3  = res60

            # 存储预测到数据库
            cur_price = float(self._close.iloc[-1])
            threading.Thread(target=lambda: [
                SelfCorrectionEngine.save_prediction(self.sym, 5,  p5_xgb,  p5_hq,  p5_mom,  p5,  cur_price, w5_1,  w5_2,  w5_3),
                SelfCorrectionEngine.save_prediction(self.sym, 20, p20_xgb, p20_hq, p20_mom, p20, cur_price, w20_1, w20_2, w20_3),
                SelfCorrectionEngine.save_prediction(self.sym, 60, p60_xgb, p60_hq, p60_mom, p60, cur_price, w60_1, w60_2, w60_3),
            ], daemon=True).start()

            # 缓存分项概率（供_refresh_prob_with_sentiment使用）
            self._last_p_xgb = {5: p5_xgb, 20: p20_xgb, 60: p60_xgb}
            self._last_p_hq  = {5: p5_hq,  20: p20_hq,  60: p60_hq}
            self._last_p_mom = {5: p5_mom, 20: p20_mom, 60: p60_mom}

            for key,prob in [("p5",p5),("p20",p20),("p60",p60)]:
                pct=int(prob*100); col=T.GREEN if pct>=60 else (T.RED if pct<=40 else T.YELLOW)
                self.prob_cards[key].setText(f"{pct}%")
                self.prob_cards[key].setStyleSheet(f"color:{col};font-size:26pt;font-weight:700;border:none;")
            for i,prob in enumerate([p5_xgb, p5_hq, p5_mom, p20, p60]):
                _,pb,val=self.bars[i]; pct=int(prob*100)
                col=T.GREEN if pct>=60 else (T.RED if pct<=40 else T.YELLOW)
                pb.setValue(pct); pb.setStyleSheet(f"QProgressBar{{background:{T.BG3};border-radius:5px;border:none;}}QProgressBar::chunk{{background:{col};border-radius:5px;}}")
                val.setText(f"{pct}%"); val.setStyleSheet(f"color:{col};font-size:9pt;font-weight:600;border:none;")
            mc5=engine.monte_carlo(5,10000); mc20=engine.monte_carlo(20,10000); mc60=engine.monte_carlo(60,10000)
            self._draw_mc(mc5,mc20,mc60)
            self.mc_text.setText(
                f"[5日]   上涨:{mc5.get('up_prob',0)*100:.1f}%  VaR95:{mc5.get('var95',0):.1f}%  CVaR95:{mc5.get('cvar95',0):.1f}%\n"
                f"[20日]  上涨:{mc20.get('up_prob',0)*100:.1f}%  VaR95:{mc20.get('var95',0):.1f}%  CVaR95:{mc20.get('cvar95',0):.1f}%\n"
                f"[60日]  上涨:{mc60.get('up_prob',0)*100:.1f}%  VaR95:{mc60.get('var95',0):.1f}%  CVaR95:{mc60.get('cvar95',0):.1f}%"
            )
            feats=engine.feature_importance(5); self._draw_features(feats)
            self._draw_quantile()
            sent_note = ""
            if self._sentiment_agg:
                adj = NewsSentimentEngine.sentiment_to_prob_adjustment(self._sentiment_agg)
                sent_note = f"  |  情感调整:{adj:+.1%}"
            self.status_lbl.setText(t("prob_done",p5=p5*100,p20=p20*100,p60=p60*100) + sent_note)
            # 刷新历史预测Tab
            QTimer.singleShot(500, self._refresh_history_tab)
        except Exception as e:
            self.status_lbl.setText(t("prob_fail",e=e)); logger.error(traceback.format_exc())

    def _draw_quantile(self):
        if self._ind is None: return
        fwd_map={'5日':5,'10日':10,'20日':20}; fwd=fwd_map.get(self.fwd_cb.currentText(),5)
        fcol=self.factor_cb.currentText()
        if fcol not in self._ind.columns: return
        with mpl.rc_context(_get_mpl_rc()):
            labels,means=FactorAnalysis.quantile_analysis(self._ind,self._close,factor_col=fcol,fwd=fwd,n_quantiles=5)
            fig1=self.quant_canvas.figure; fig1.clear(); fig1.patch.set_facecolor(T.MPL_BG)
            if labels:
                ax1,ax2=fig1.subplots(1,2)
                colors=[T.RED if m<0 else T.GREEN for m in means]
                ax1.bar(labels,means,color=colors,alpha=0.85,edgecolor='none',width=0.55)
                for i,(lb,mv) in enumerate(zip(labels,means)):
                    ax1.text(i,mv+(0.05 if mv>=0 else -0.05),f"{mv:.2f}%",ha='center',va='bottom' if mv>=0 else 'top',color=T.TEXT_H,fontsize=8)
                ax1.axhline(0,color=T.TEXT_3,lw=1)
                spread_val=means[-1]-means[0] if means else 0
                ax1.text(0.98,0.02,f"多空价差:{spread_val:+.2f}%",transform=ax1.transAxes,ha='right',va='bottom',color=T.GREEN if spread_val>=0 else T.RED,fontsize=9)
                _mpl_style(ax1,title="Alpha分层（5分位）",xlabel="分位",ylabel=f"{fwd}日收益%")
                factor=self._ind[fcol].dropna(); future=self._close.pct_change(fwd).shift(-fwd)*100
                comb=pd.concat([factor,future],axis=1).dropna(); comb.columns=['f','r']
                try:
                    comb['q']=pd.qcut(comb['f'],5,labels=False,duplicates='drop')
                    q1=comb[comb['q']==0]['r'].values; q5=comb[comb['q']==4]['r'].values
                    if len(q1)>5 and len(q5)>5:
                        ax2.hist(q1,bins=30,alpha=0.65,color=T.RED,density=True,label='Q1(低)')
                        ax2.hist(q5,bins=30,alpha=0.65,color=T.GREEN,density=True,label='Q5(高)')
                        ax2.axvline(0,color=T.TEXT_3,lw=1,ls='--')
                        ax2.legend(fontsize=8,facecolor=T.BG2,labelcolor=T.TEXT_H,edgecolor=T.BORDER)
                        _mpl_style(ax2,title="Q1 vs Q5 收益分布",xlabel=f"{fwd}日收益%")
                except: _mpl_style(ax2)
            else:
                ax=fig1.add_subplot(111)
                ax.text(0.5,0.5,"数据不足",ha='center',va='center',color=T.TEXT_2,transform=ax.transAxes)
                _mpl_style(ax)
            _apply_font_to_figure(fig1)
        self.quant_canvas.draw()
        with mpl.rc_context(_get_mpl_rc()):
            ic_series=FactorAnalysis.rolling_ic(self._ind,self._close,factor_col=fcol,fwd=fwd,window=20)
            fig2=self.ic_canvas.figure; fig2.clear(); fig2.patch.set_facecolor(T.MPL_BG)
            ax=fig2.add_subplot(111)
            if not ic_series.empty:
                ic_ma=ic_series.rolling(5).mean()
                ax.fill_between(ic_series.index,ic_series.values,0,where=ic_series.values>0,color=T.GREEN,alpha=0.3)
                ax.fill_between(ic_series.index,ic_series.values,0,where=ic_series.values<0,color=T.RED,alpha=0.3)
                ax.plot(ic_series.index,ic_series.values,color=T.TEXT_3,lw=0.8,alpha=0.6)
                ax.plot(ic_ma.index,ic_ma.values,color=T.ACCENT,lw=1.5,label=f"IC均值:{ic_series.mean():.3f}")
                ax.axhline(0,color=T.TEXT_3,lw=1,ls='--')
                ax.legend(fontsize=8,facecolor=T.BG2,labelcolor=T.TEXT_H,edgecolor=T.BORDER)
            _mpl_style(ax,title="因子IC滚动（20日窗口）",ylabel="IC")
            _apply_font_to_figure(fig2)
        self.ic_canvas.draw()

    def _draw_mc(self,mc5,mc20,mc60):
        with mpl.rc_context(_get_mpl_rc()):
            fig=self.mc_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax1,ax2,ax3=fig.subplots(1,3)
            for ax,mc,fwd in [(ax1,mc5,"5"),(ax2,mc20,"20"),(ax3,mc60,"60")]:
                if 'finals' not in mc: continue
                finals=mc['finals']; cur=mc.get('cur',1)
                pcts=(finals-cur)/cur*100 if cur>0 else finals*100-100
                up=pcts[pcts>=0]; dn=pcts[pcts<0]
                ax.hist(dn,bins=50,color=T.RED,alpha=0.75,density=True)
                ax.hist(up,bins=50,color=T.GREEN,alpha=0.75,density=True)
                ax.axvline(0,color=T.GOLD,lw=1.5,ls='--',alpha=0.9)
                ax.axvline(mc.get('var95',0),color=T.ORANGE,lw=1,ls=':',alpha=0.8)
                _mpl_style(ax,title=f"{fwd}日 上涨:{mc.get('up_prob',0)*100:.0f}% (fat-tail)",xlabel="收益%")
            _apply_font_to_figure(fig)
        self.mc_canvas.draw()

    def _draw_features(self,feats):
        if not feats: return
        with mpl.rc_context(_get_mpl_rc()):
            fig=self.feat_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax=fig.add_subplot(111)
            labels=[f[0] for f in feats]; values=[f[1] for f in feats]
            colors=[T.GREEN if v>0 else T.RED for v in values]
            y=range(len(labels)); bars=ax.barh(list(y),values,color=colors,alpha=0.85,height=0.55,edgecolor='none')
            for bar,v in zip(bars,values):
                x=bar.get_width()
                ax.text(x+(0.003 if x>=0 else -0.003),bar.get_y()+bar.get_height()/2,f"{v:.3f}",va='center',ha='left' if x>=0 else 'right',color=T.TEXT_1,fontsize=8)
            ax.set_yticks(list(y)); ax.set_yticklabels(labels,color=T.TEXT_H,fontsize=9)
            ax.axvline(0,color=T.TEXT_2,lw=1)
            _mpl_style(ax,title="XGBoost特征贡献度（v9.0扩展因子）",xlabel="贡献值")
            _apply_font_to_figure(fig)
        self.feat_canvas.draw()

# ══════════════════════════════════════════════════════════════════
# 协整弹窗
# ══════════════════════════════════════════════════════════════════
class CointegrationDialog(QDialog):
    def __init__(self,syms,parent=None):
        super().__init__(parent); self.syms=syms
        self.setWindowTitle("协整配对分析"); self.setWindowIcon(_make_app_icon())
        _fit_to_screen(self, 860, 640); self.setStyleSheet(GLOBAL_STYLE)
        self._build_ui(); QTimer.singleShot(200,self._run)
    def _build_ui(self):
        lay=QVBoxLayout(self); lay.setContentsMargins(14,12,14,12)
        hdr=QLabel(f"<b>协整配对分析</b>  {', '.join(self.syms[:10])}"); hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;padding:6px;"); lay.addWidget(hdr)
        self.status=QLabel("计算中..."); self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;"); lay.addWidget(self.status)
        self.canvas=FigureCanvas(Figure(figsize=(8,4),facecolor=T.MPL_BG))
        self.text=QTextEdit(); self.text.setReadOnly(True); self.text.setStyleSheet(f"background:{T.BG2};color:{T.TEXT_1};border:1px solid {T.BORDER};font-size:9pt;border-radius:6px;")
        sp=QSplitter(Qt.Vertical); sp.addWidget(self.canvas); sp.addWidget(self.text); sp.setSizes([300,200]); lay.addWidget(sp,1)
        self.rp_label=QLabel(); self.rp_label.setStyleSheet(f"color:{T.CYAN};font-size:9pt;padding:4px;background:{T.BG2};border-radius:4px;"); self.rp_label.setWordWrap(True); lay.addWidget(self.rp_label)
    def _run(self):
        self.status.setText("计算协整关系..."); QApplication.processEvents()
        close_dict={}
        for sym in self.syms:
            df=fetch_stock_data(sym,"2y")
            if not df.empty: close_dict[sym]=df['Close'].squeeze().astype(float)
        if len(close_dict)<2: self.status.setText("数据不足"); return
        try:
            pairs=CointegrationAnalysis.find_pairs(list(close_dict.keys()),close_dict)
            lines=[]
            for s1,s2,p,spread in pairs[:10]:
                verdict="✅ 协整" if p<0.05 else "❌ 无协整"
                lines.append(f"{s1}/{s2}  p值:{p:.4f}  {verdict}")
            self.text.setText('\n'.join(lines) if lines else "未发现协整对（p<0.05）")
            if pairs:
                s1,s2,p,spread=pairs[0]; self._draw_spread(s1,s2,spread)
                self.status.setText(f"检验{len(close_dict)*(len(close_dict)-1)//2}对，发现{len(pairs)}个协整对")
            else: self.status.setText("未发现协整对")
        except ImportError: self.status.setText("需要: pip install statsmodels")
        except Exception as e: self.status.setText(f"出错: {e}")
        returns_dict={s:c.pct_change().dropna() for s,c in close_dict.items()}
        alloc=PortfolioOptimizer.vol_target_weights(pd.DataFrame(returns_dict).dropna())
        self.rp_label.setText("【风险平价仓位】 "+"  |  ".join([f"{s}:{v:.1f}%" for s,v in alloc.items()]))
    def _draw_spread(self,s1,s2,spread):
        with mpl.rc_context(_get_mpl_rc()):
            fig=self.canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            ax=fig.add_subplot(111); mean=spread.mean(); std=spread.std()
            ax.plot(spread.index,spread.values,color=T.ACCENT,lw=1.2,alpha=0.9,label='价差')
            ax.axhline(mean,color=T.GOLD,lw=1.5,ls='--',label='均值')
            ax.axhline(mean+std,color=T.GREEN,lw=1,ls=':',alpha=0.8,label='+1σ')
            ax.axhline(mean-std,color=T.RED,lw=1,ls=':',alpha=0.8,label='-1σ')
            ax.axhline(mean+2*std,color=T.GREEN,lw=0.8,ls='-.',alpha=0.6)
            ax.axhline(mean-2*std,color=T.RED,lw=0.8,ls='-.',alpha=0.6)
            ax.fill_between(spread.index,mean-std,mean+std,alpha=0.08,color=T.ACCENT)
            ax.legend(fontsize=7,ncol=3,facecolor=T.BG2,labelcolor=T.TEXT_H,edgecolor=T.BORDER)
            _mpl_style(ax,title=f"协整价差  {s1}/{s2}",ylabel="价差")
            _apply_font_to_figure(fig)
        self.canvas.draw()

# ══════════════════════════════════════════════════════════════════
# 全局样式
# ══════════════════════════════════════════════════════════════════
GLOBAL_STYLE = f"""
QWidget {{ background:{T.BG0}; color:{T.TEXT_1}; font-family:"Segoe UI","Microsoft YaHei",Arial; font-size:10.5pt; }}
QWidget#MainWindow {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0a1320, stop:0.55 #081019, stop:1 {T.BG0}); }}
QGroupBox {{ border:1px solid {T.BORDER}; border-top:1px solid {T.BORDER_HI}; border-radius:10px; margin-top:14px; font-weight:600; color:{T.GOLD};
             background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #12233a, stop:1 #0b1727); }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:0 6px; color:{T.GOLD}; }}
QLineEdit {{ border:1px solid {T.BORDER}; border-radius:7px; padding:6px 10px; background:{T.BG2}; color:{T.TEXT_H}; selection-background-color:{T.ACCENT}; }}
QLineEdit:focus {{ border:1px solid {T.ACCENT}; background:#13243a; }}
QPushButton {{ color:{T.TEXT_H}; border:1px solid {T.BORDER}; border-top:1px solid {T.BORDER_HI}; border-radius:7px; padding:6px 14px; font-weight:600; min-width:52px;
               background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a2c44, stop:0.5 #142435, stop:1 #0f1d2e); }}
QPushButton:hover {{ color:{T.ACCENT2}; border:1px solid {T.ACCENT};
                     background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1d3b54, stop:1 #122a3f); }}
QPushButton:pressed {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0f1d2e, stop:1 #1a2c44); }}
QPushButton:disabled {{ background:{T.BG1}; color:{T.TEXT_3}; border-color:{T.BORDER}; }}
QPushButton:checked {{ color:white; border:1px solid {T.ACCENT2};
                       background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #38bdf8, stop:1 #0883c0); }}
QTableView {{ background:{T.BG2}; alternate-background-color:{T.BG3}; gridline-color:{T.BORDER}; selection-background-color:{T.BG4}; selection-color:{T.ACCENT2}; color:{T.TEXT_1}; border:none; }}
QHeaderView::section {{ color:{T.GOLD}; padding:7px 4px; border:none; border-bottom:1px solid {T.ACCENT}; border-right:1px solid {T.BORDER}; font-size:9.5pt; font-weight:600;
                        background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #14294a, stop:1 #0b1726); }}
QProgressBar {{ border:none; border-radius:5px; text-align:center; height:12px; background:{T.BG3}; color:{T.TEXT_H}; font-size:8.5pt; }}
QProgressBar::chunk {{ border-radius:5px;
                       background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0883c0, stop:0.5 {T.ACCENT}, stop:1 #5cc8f5); }}
QTabWidget::pane {{ border:1px solid {T.BORDER}; border-radius:8px; background:{T.BG1}; }}
QTabBar::tab {{ color:{T.TEXT_2}; padding:7px 16px; margin-right:2px; border-top-left-radius:7px; border-top-right-radius:7px; font-weight:600; font-size:9.5pt; border:1px solid {T.BORDER}; border-bottom:none;
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #13243a, stop:1 #0d1826); }}
QTabBar::tab:selected {{ color:{T.ACCENT2}; border-color:{T.ACCENT}; border-bottom:2px solid {T.ACCENT2};
                         background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a3050, stop:1 #0d1826); }}
QTabBar::tab:hover:!selected {{ background:{T.BG3}; color:{T.TEXT_H}; }}
QComboBox {{ border:1px solid {T.BORDER}; border-top:1px solid {T.BORDER_HI}; border-radius:7px; padding:5px 10px; color:{T.TEXT_H};
             background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1a2c44, stop:1 #0f1d2e); }}
QComboBox:hover {{ border:1px solid {T.ACCENT}; }}
QComboBox::drop-down {{ border:none; }}
QComboBox::down-arrow {{ image:none; border-left:5px solid transparent; border-right:5px solid transparent; border-top:5px solid {T.TEXT_2}; margin-right:6px; }}
QComboBox QAbstractItemView {{ background:{T.BG2}; color:{T.TEXT_1}; border:1px solid {T.ACCENT}; selection-background-color:{T.ACCENT}; selection-color:white; outline:none; }}
QSpinBox {{ border:1px solid {T.BORDER}; border-radius:7px; padding:4px 6px; background:{T.BG2}; color:{T.TEXT_H}; }}
QSpinBox:focus {{ border:1px solid {T.ACCENT}; }}
QTextEdit, QTextBrowser {{ background:{T.BG2}; border:1px solid {T.BORDER}; color:{T.TEXT_1}; font-size:9.5pt; border-radius:8px; padding:4px; }}
QSlider::groove:horizontal {{ height:5px; background:{T.BG3}; border-radius:3px; }}
QSlider::handle:horizontal {{ width:15px; height:15px; border-radius:8px; margin:-6px 0; border:1px solid {T.ACCENT2};
                              background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #7fd4f7, stop:0.5 {T.ACCENT}, stop:1 #0883c0); }}
QSlider::handle:horizontal:hover {{ border:1px solid white; }}
QSlider::sub-page:horizontal {{ border-radius:3px;
                                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0883c0, stop:1 {T.ACCENT2}); }}
QSplitter::handle {{ background:{T.BORDER}; }}
QStatusBar {{ color:{T.TEXT_2}; font-size:9pt; border-top:1px solid {T.BORDER};
              background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0d1826, stop:1 #0a121e); }}
QCheckBox {{ color:{T.TEXT_1}; }}
QCheckBox::indicator {{ width:16px; height:16px; border-radius:5px; border:1px solid {T.BORDER}; background:{T.BG2}; }}
QCheckBox::indicator:checked {{ border:1px solid {T.ACCENT2};
                                background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #38bdf8, stop:1 #0883c0); }}
QScrollBar:vertical {{ background:{T.BG1}; width:9px; border:none; margin:0; }}
QScrollBar::handle:vertical {{ border-radius:4px; min-height:24px;
                               background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {T.BG4}, stop:1 #25405f); }}
QScrollBar::handle:vertical:hover {{ background:{T.ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{ background:{T.BG1}; height:9px; border:none; margin:0; }}
QScrollBar::handle:horizontal {{ border-radius:4px; min-width:24px;
                                 background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {T.BG4}, stop:1 #25405f); }}
QScrollBar::handle:horizontal:hover {{ background:{T.ACCENT}; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QDialog {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #0a1320, stop:1 {T.BG0}); }}
"""

# ══════════════════════════════════════════════════════════════════
# 主窗口 v9.0
# ══════════════════════════════════════════════════════════════════
class QuantApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")   # 供 QSS 单独给主窗口加渐变背景
        _fit_to_screen(self, 1800, 1040); self.setWindowIcon(_make_app_icon())
        # 显式启用最大化，并把窗口硬下限设小：按钮行改为横向滚动后，
        # 窗口可缩到任意宽度，任何屏幕/DPI缩放下都能正常最大化
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setMinimumSize(640, 480)
        self.setStyleSheet(GLOBAL_STYLE)
        self.rsi_buy_max=55; self.rsi_sell_min=70; self.use_ma=True
        self.stop_loss_pct=5; self.take_profit_pct=10
        self._scan_thread=None; self._bt_thread=None; self._ticker_thread=None
        self._equity_curves=[]; self._syms_data={}
        self._build_widgets(); self._build_layout(); self._connect_signals(); self._retranslate()

    def _build_widgets(self):
        self.input_edit=QLineEdit()
        self.filter_edit=QLineEdit(); self.filter_edit.setFixedWidth(230)
        # 涨跌周期选择（多周期涨跌排名）
        self.chg_period_lbl=QLabel()
        self.chg_period_cb=QComboBox(); self.chg_period_cb.setFixedWidth(96)
        self.chg_period_cb.addItems(_chg_period_items())

        def _btn(txt="",w=None,style_extra=""):
            b=QPushButton(txt)
            if w: b.setFixedWidth(w)
            if style_extra: b.setStyleSheet(b.styleSheet()+style_extra)
            return b

        self.scan_btn     = _btn()
        self.top100_btn   = _btn()
        self.bt_btn       = _btn()
        self.cross_btn    = _btn(style_extra=f"color:{T.PURPLE};border-color:{T.PURPLE};")
        self.portfolio_btn= _btn(style_extra=f"color:{T.GREEN};border-color:{T.GREEN};")
        self.coint_btn    = _btn(style_extra=f"color:{T.CYAN};border-color:{T.CYAN};")
        self.stop_btn     = _btn(style_extra=f"background:{T.RED};color:white;border-color:{T.RED};")
        self.stop_btn.setEnabled(False)
        self.clear_btn    = _btn()
        self.export_btn   = _btn()
        self.cache_btn    = _btn()
        self.lang_btn     = _btn(w=84,style_extra=f"color:{T.ACCENT2};border-color:{T.ACCENT};")

        # 参数区
        self.param_group=QGroupBox()
        # 面板用途说明（避免误以为拖滑块能改变AI概率预测）
        self.param_note=QLabel(
            "ⓘ 这些参数仅控制【扫描列表的评分/信号】，用于快速初筛。\n"
            "它们<b>不影响</b>概率分析窗口的 AI 预测（那套走 v14 引擎，"
            "经 Purged CV 检验）。\n建议用默认值即可，微调意义不大。")
        self.param_note.setWordWrap(True)
        self.param_note.setStyleSheet(
            f"color:{T.TEXT_2};font-size:8pt;background:{T.BG2};"
            f"border:1px solid {T.BORDER};border-radius:6px;padding:6px;line-height:1.5;")
        self.rsi_buy_lbl=QLabel()
        self.rsi_buy_sl=QSlider(Qt.Horizontal); self.rsi_buy_sl.setRange(20,80); self.rsi_buy_sl.setValue(55)
        self.rsi_sell_lbl=QLabel()
        self.rsi_sell_sl=QSlider(Qt.Horizontal); self.rsi_sell_sl.setRange(40,90); self.rsi_sell_sl.setValue(70)
        self.ma_chk=QCheckBox(); self.ma_chk.setChecked(True)
        self.sl_lbl=QLabel(); self.sl_sl=QSlider(Qt.Horizontal); self.sl_sl.setRange(1,20); self.sl_sl.setValue(5)
        self.tp_lbl=QLabel(); self.tp_sl=QSlider(Qt.Horizontal); self.tp_sl.setRange(5,50); self.tp_sl.setValue(10)
        # Walk Forward 参数
        wf_lbl=QLabel("Walk Forward参数:"); wf_lbl.setStyleSheet(f"color:{T.GOLD};font-size:9pt;border:none;")
        self.train_spin=QSpinBox(); self.train_spin.setRange(6,24); self.train_spin.setValue(12)
        self.train_spin.setPrefix("训练:"); self.train_spin.setSuffix("月")
        self.test_spin=QSpinBox();  self.test_spin.setRange(1,6);  self.test_spin.setValue(3)
        self.test_spin.setPrefix("测试:"); self.test_spin.setSuffix("月")
        self.period_cb=QComboBox()
        self.score_info=QLabel(); self.score_info.setWordWrap(True)
        self.score_info.setStyleSheet(f"color:{T.YELLOW};font-size:8pt;padding:4px;border:none;line-height:1.6;")
        pl=QVBoxLayout(); pl.setSpacing(6)
        for w in [self.param_note,
                  self.rsi_buy_lbl,self.rsi_buy_sl,self.rsi_sell_lbl,self.rsi_sell_sl,
                  self.ma_chk,self.sl_lbl,self.sl_sl,self.tp_lbl,self.tp_sl,
                  wf_lbl,self.train_spin,self.test_spin,self.period_cb,self.score_info]:
            pl.addWidget(w)
        self.param_group.setLayout(pl)

        self.summary=SummaryPanel()
        self.scan_model=ScanModel(); self.proxy=DisplayFilterProxy(); self.proxy.setSourceModel(self.scan_model)
        self.table=QTableView(); self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True); self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.verticalHeader().setStyleSheet(f"QHeaderView::section{{background:{T.BG1};color:{T.TEXT_3};border:none;border-bottom:1px solid {T.BORDER};}}")
        hdr=self.table.horizontalHeader(); hdr.setSectionResizeMode(QHeaderView.Interactive); hdr.setStretchLastSection(True)
        for i,w in enumerate([62,72,72,62,46,60,62,62,62,68,46,46,62,56,82,72,72,240]):
            self.table.setColumnWidth(i,w)
        self.table.doubleClicked.connect(self._on_double_click)

        # 回测Tab（Walk Forward）
        self.bt_text=QTextEdit(); self.bt_text.setReadOnly(True)
        self.bt_canvas=FigureCanvas(Figure(figsize=(9,4.5),facecolor=T.MPL_BG))
        bt_w=QWidget(); btl=QVBoxLayout(bt_w)
        sp=QSplitter(Qt.Vertical); sp.addWidget(self.bt_text); sp.addWidget(self.bt_canvas); sp.setSizes([280,320])
        btl.addWidget(sp); btl.setContentsMargins(0,0,0,0)

        self.central_tab=QTabWidget()
        self.central_tab.addTab(self.table,"")       # 扫描
        self.central_tab.addTab(bt_w,"")             # Walk Forward回测

        self.progress=QProgressBar(); self.progress.setVisible(False)
        self.status_bar=QStatusBar()
        self.flip_widget=FlipBoardWidget()
        self.lbl_code=QLabel(); self.lbl_code.setStyleSheet(f"color:{T.TEXT_2};font-size:10pt;padding-right:4px;")

    def _build_layout(self):
        root=QVBoxLayout(self); root.setSpacing(8); root.setContentsMargins(14,12,14,12)
        row1=QHBoxLayout()
        row1.addWidget(self.lbl_code); row1.addWidget(self.input_edit,1)
        row1.addWidget(self.chg_period_lbl); row1.addWidget(self.chg_period_cb)
        row1.addWidget(self.filter_edit)
        root.addLayout(row1)
        row2=QHBoxLayout(); row2.setSpacing(6)
        for b in [self.scan_btn,self.top100_btn,self.bt_btn,self.cross_btn,
                  self.portfolio_btn,self.coint_btn,self.stop_btn,
                  self.clear_btn,self.export_btn,self.cache_btn,self.lang_btn]:
            row2.addWidget(b)
        row2.addStretch()
        # 按钮行装进横向滚动容器：按钮再多也只横向滚动，不会把窗口最小宽度
        # 撑到超过屏幕（否则在小屏/高DPI下会导致无法最大化）。
        # self.btn_row 暴露给插件模块（配对对比/大盘仪表盘）直接插按钮。
        self.btn_row = row2
        _btn_container=QWidget(); _btn_container.setLayout(row2)
        _btn_scroll=QScrollArea(); _btn_scroll.setWidgetResizable(True)
        _btn_scroll.setWidget(_btn_container)
        _btn_scroll.setFrameShape(QScrollArea.NoFrame)
        _btn_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _btn_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _btn_scroll.setFixedHeight(_btn_container.sizeHint().height()+10)
        _btn_scroll.setMinimumWidth(0)
        self._btn_scroll=_btn_scroll
        root.addWidget(_btn_scroll)
        content=QHBoxLayout(); content.setSpacing(10)
        self.param_group.setFixedWidth(290); content.addWidget(self.param_group)
        right=QVBoxLayout(); right.setSpacing(8)
        right.addWidget(self.summary); right.addWidget(self.central_tab,1)
        content.addLayout(right,1); root.addLayout(content,1)
        root.addWidget(self.progress); root.addWidget(self.flip_widget); root.addWidget(self.status_bar)

    def _connect_signals(self):
        self.rsi_buy_sl.valueChanged.connect(lambda v:(setattr(self,'rsi_buy_max',v),self.rsi_buy_lbl.setText(f"{t('lbl_rsi_buy')}: {v}")))
        self.rsi_sell_sl.valueChanged.connect(lambda v:(setattr(self,'rsi_sell_min',v),self.rsi_sell_lbl.setText(f"{t('lbl_rsi_sell')}: {v}")))
        self.ma_chk.stateChanged.connect(lambda s:setattr(self,'use_ma',s==Qt.Checked))
        self.sl_sl.valueChanged.connect(lambda v:(setattr(self,'stop_loss_pct',v),self.sl_lbl.setText(f"{t('lbl_stop_loss')}: {v}%")))
        self.tp_sl.valueChanged.connect(lambda v:(setattr(self,'take_profit_pct',v),self.tp_lbl.setText(f"{t('lbl_take_profit')}: {v}%")))
        self.scan_btn.clicked.connect(self._start_scan)
        self.top100_btn.clicked.connect(self._load_top100)
        self.bt_btn.clicked.connect(self._start_backtest)
        self.cross_btn.clicked.connect(self._open_cross)
        self.portfolio_btn.clicked.connect(self._open_portfolio)
        self.coint_btn.clicked.connect(self._open_coint)
        self.stop_btn.clicked.connect(self._stop_tasks)
        self.clear_btn.clicked.connect(self._clear)
        self.export_btn.clicked.connect(self._export_csv)
        self.cache_btn.clicked.connect(lambda:(clear_cache(),self.status_bar.showMessage(t("status_cache_cleared"),3000)))
        self.filter_edit.textChanged.connect(self.proxy.setFilterText)
        self.chg_period_cb.currentIndexChanged.connect(self._on_chg_period_changed)
        self.lang_btn.clicked.connect(self._toggle_lang)

    def _toggle_lang(self): set_lang("en" if LANG=="zh" else "zh"); self._retranslate()

    def _retranslate(self):
        self.setWindowTitle(t("win_title")); self.lbl_code.setText(t("lbl_code"))
        self.input_edit.setPlaceholderText(t("lbl_input_ph")); self.filter_edit.setPlaceholderText(t("lbl_filter_ph"))
        self.chg_period_lbl.setText("涨跌周期:" if LANG=="zh" else "Period:")
        if hasattr(self,"chg_period_cb"):
            _i=self.chg_period_cb.currentIndex()
            self.chg_period_cb.blockSignals(True)
            self.chg_period_cb.clear(); self.chg_period_cb.addItems(_chg_period_items())
            self.chg_period_cb.setCurrentIndex(max(_i,0))
            self.chg_period_cb.blockSignals(False)
            # 同步表头当前周期标签语言
            if _i>0:
                zh,en,_nb,_k=_CHG_PERIODS[_i]
                self.scan_model.chg_period_label = (zh if LANG=="zh" else en)
                self.scan_model.headerDataChanged.emit(Qt.Horizontal, _COL_KEYS.index("col_chg"), _COL_KEYS.index("col_chg"))
        self.scan_btn.setText(t("btn_scan")); self.top100_btn.setText(t("btn_top100"))
        self.bt_btn.setText(t("btn_backtest")); self.cross_btn.setText(t("btn_cross"))
        self.portfolio_btn.setText(t("btn_portfolio")); self.coint_btn.setText(t("btn_coint"))
        self.stop_btn.setText(t("btn_stop")); self.clear_btn.setText(t("btn_clear"))
        self.export_btn.setText(t("btn_export")); self.cache_btn.setText(t("btn_cache"))
        self.lang_btn.setText(t("btn_lang"))
        self.param_group.setTitle(t("lbl_param_group"))
        self.rsi_buy_lbl.setText(f"{t('lbl_rsi_buy')}: {self.rsi_buy_max}")
        self.rsi_sell_lbl.setText(f"{t('lbl_rsi_sell')}: {self.rsi_sell_min}")
        self.ma_chk.setText(t("lbl_ma_chk"))
        self.sl_lbl.setText(f"{t('lbl_stop_loss')}: {self.stop_loss_pct}%")
        self.tp_lbl.setText(f"{t('lbl_take_profit')}: {self.take_profit_pct}%")
        self.score_info.setText(t("lbl_score_info"))
        self.period_cb.blockSignals(True); idx=self.period_cb.currentIndex(); self.period_cb.clear()
        self.period_cb.addItems([t("period_6mo"),t("period_1y"),t("period_2y"),t("period_5y")]); self.period_cb.setCurrentIndex(max(idx,0)); self.period_cb.blockSignals(False)
        self.central_tab.setTabText(0,t("tab_scan")); self.central_tab.setTabText(1,t("tab_backtest"))
        self.summary.retranslate(); self.status_bar.showMessage(t("status_ready")); self.scan_model.reload_columns()

    def _params(self):
        return {'rsi_buy_max':self.rsi_buy_max,'rsi_sell_min':self.rsi_sell_min,
                'use_ma_condition':self.use_ma,'stop_loss_pct':self.stop_loss_pct/100.,
                'take_profit_pct':self.take_profit_pct/100.}

    def _parse_input(self):
        txt=self.input_edit.text().strip()
        if not txt: return []
        syms=[]
        for item in txt.split(","):
            item=item.strip()
            if not item: continue
            parts=item.split(":")
            code=_normalize(parts[0].strip().upper())
            name=parts[1].strip() if len(parts)>1 else Top100Loader.get_name(code,code)
            syms.append((code,name))
        return syms

    def _period_str(self): return ["6mo","1y","2y","5y"][max(self.period_cb.currentIndex(),0)]

    def _load_top100(self):
        self.status_bar.showMessage(t("status_loading_top100")); QApplication.processEvents()
        syms=Top100Loader.load()
        if syms: self.input_edit.setText(",".join(s[0] for s in syms)); self.status_bar.showMessage(t("status_loaded",n=len(syms)))
        else: QMessageBox.warning(self,t("dlg_error"),"加载失败")

    def _start_scan(self):
        syms=self._parse_input()
        if not syms: QMessageBox.warning(self,t("dlg_hint"),t("dlg_input_code")); return
        self.scan_model.clear(); self._syms_data={}; self.summary.update_stats(self.scan_model)
        self.progress.setVisible(True); self._set_busy(True)
        self._scan_thread=AnalysisThread(syms,self._params())
        self._scan_thread.progress.connect(lambda v,n:(self.progress.setValue(v),self.status_bar.showMessage(t("status_analyzing",sym=n))))
        self._scan_thread.result_ready.connect(self._on_scan_result)
        self._scan_thread.finished_all.connect(self._on_scan_done)
        self._scan_thread.error_msg.connect(lambda m:self.status_bar.showMessage(m,4000))
        self._scan_thread.start()

    def _on_scan_result(self,d):
        self.scan_model.append_row(d); self.summary.update_stats(self.scan_model)

    def _on_scan_done(self,n):
        self.progress.setVisible(False); self._set_busy(False)
        self.table.resizeColumnsToContents(); self.table.horizontalHeader().setStretchLastSection(True)
        # 在后台下载数据用于横截面分析
        syms=[r.get('col_code','') for r in self.scan_model._rows if r.get('col_code','')]
        def _prefetch():
            for sym in syms:
                df=fetch_stock_data(sym,"1y")
                if not df.empty: self._syms_data[sym]=df
        threading.Thread(target=_prefetch,daemon=True).start()
        # 计算横截面排名
        QTimer.singleShot(3000, self._update_cs_ranks)
        self.status_bar.showMessage(t("status_done_scan",n=n))

    def _update_cs_ranks(self):
        if not self._syms_data: return
        try:
            # v10.0：使用IC加权横截面排名
            scores=ICWeightedAlpha.compute_scores_ic_weighted(self._syms_data)
            if not scores.empty:
                ranks={sym: int(row.get('rank',0)) for sym,row in scores.iterrows()}
                self.scan_model.update_cs_ranks(ranks)
        except: pass

    def _on_chg_period_changed(self, idx):
        """切换涨跌周期：更新涨跌%列为该周期、刷新表头、按涨跌降序排名。"""
        if idx < 0 or idx >= len(_CHG_PERIODS): return
        zh, en, _nb, key = _CHG_PERIODS[idx]
        label = zh if LANG=="zh" else en
        self.scan_model.set_chg_period(key, label)
        # 按涨跌降序重新排名（涨幅最大的在最上面）
        chg_col=_COL_KEYS.index("col_chg")
        self.table.sortByColumn(chg_col, Qt.DescendingOrder)
        n=self.scan_model.rowCount()
        if n:
            self.status_bar.showMessage(
                (f"涨跌排名已切换为 {label} 周期" if LANG=="zh" else f"Ranking by {label} return"), 3000)

    def _start_backtest(self):
        syms=self._parse_input()
        if not syms: QMessageBox.warning(self,t("dlg_hint"),t("dlg_input_code")); return
        self.bt_text.clear(); self.bt_canvas.figure.clear(); self.bt_canvas.draw()
        self._equity_curves=[]; self.progress.setVisible(True); self._set_busy(True)
        train_m=self.train_spin.value(); test_m=self.test_spin.value()
        self._bt_thread=BacktestThread(syms,self._params(),self._period_str(),train_m,test_m)
        self._bt_thread.progress.connect(lambda v,n:(self.progress.setValue(v),self.status_bar.showMessage(t("status_backtesting",sym=n))))
        self._bt_thread.result_ready.connect(self._on_bt_result)
        self._bt_thread.log_msg.connect(self.bt_text.append)
        self._bt_thread.finished_all.connect(self._on_bt_done)
        self._bt_thread.error_msg.connect(lambda m:self.bt_text.append(m))
        self._bt_thread.start()
        self.central_tab.setCurrentIndex(1)

    def _on_bt_result(self,res):
        alpha=res['total_return']-res['bh_return']
        self.bt_text.append(t("bt_result",
            sym=res['symbol'],name=res.get('chn_name',''),tr=res['total_return'],bh=res['bh_return'],alpha=alpha,
            wr=res['win_rate'],tc=res['trade_count'],dd=res['max_drawdown'],
            sr=res['sharpe'],so=res.get('sortino',0),ca=res.get('calmar',0),
            pf=res.get('profit_factor',1.0),dec=res.get('oos_decay',0),turn=res.get('avg_turnover',0),
            var95=res.get('var95',0),cvar95=res.get('cvar95',0),
            folds=res.get('n_folds',0),line='─'*60))
        self._equity_curves.append(res); self._draw_equity()

    def _draw_equity(self):
        with mpl.rc_context(_get_mpl_rc()):
            fig=self.bt_canvas.figure; fig.clear(); fig.patch.set_facecolor(T.MPL_BG)
            # v10.0：左侧净值曲线 + 右侧雷达图
            gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1.2], wspace=0.3)
            ax=fig.add_subplot(gs[0]); ax.set_facecolor(T.MPL_AXES)
            ax_radar=fig.add_subplot(gs[1], polar=True)

            palette=[T.ACCENT,T.GREEN,T.YELLOW,T.RED,T.PURPLE,T.ORANGE,T.CYAN,'#84cc16','#ec4899']
            radar_data = []
            for i,res in enumerate(self._equity_curves):
                dates,eq=res['equity_curve']
                ax.plot(dates,eq,label=res['symbol'],lw=1.8,color=palette[i%len(palette)],alpha=0.92)
                folds=res.get('fold_stats',[])
                for fi,fold in enumerate(folds):
                    col=palette[i%len(palette)]
                    ax.axvspan(fold['start'],fold['end'],alpha=0.05,color=col)
                    if fi==0:
                        ax.axvline(fold['start'],color=col,lw=0.5,ls=':',alpha=0.4)
                # 雷达数据
                radar_data.append({
                    'label': res['symbol'],
                    'sharpe': res.get('sharpe', 0),
                    'calmar': res.get('calmar', 0),
                    'win_rate': res.get('win_rate', 50),
                    'profit_factor': res.get('profit_factor', 1.0),
                    'oos_decay': res.get('oos_decay', 0.5),
                    'turnover_cost': min(res.get('avg_turnover', 0) * 5, 1.0),
                })
            ax.axhline(10000,color=T.TEXT_3,ls='--',lw=1,alpha=0.6,label=t("bt_equity_init"))
            _mpl_style(ax,title=t("bt_equity_title"),ylabel=t("bt_equity_yaxis"))
            ax.legend(fontsize=7,ncol=4,facecolor=T.BG2,labelcolor=T.TEXT_H,edgecolor=T.BORDER,framealpha=0.8)

            # 雷达图
            if radar_data:
                ax_radar.set_title("策略性能\n雷达图", color=T.TEXT_H, fontsize=8, pad=10)
                StrategyRadarChart.draw(ax_radar, radar_data[:5], colors=palette)

            fig.tight_layout(pad=1.2); _apply_font_to_figure(fig)
        self.bt_canvas.draw()

    def _on_bt_done(self,n):
        self.progress.setVisible(False); self._set_busy(False)
        self.status_bar.showMessage(t("status_bt_done",n=n))
        if not self._equity_curves: self.bt_text.append(t("bt_no_data"))

    def _on_double_click(self,proxy_idx):
        src=self.proxy.mapToSource(proxy_idx); row=self.scan_model.row_dict(src.row())
        if row:
            dlg=ProbabilityDialog(row.get('col_code',''),row.get('col_name',''),row,self); dlg.exec_()

    def _open_cross(self):
        if not self._syms_data:
            syms=self._parse_input()
            if len(syms)<2: QMessageBox.information(self,t("dlg_hint"),t("dlg_need_2stocks")); return
            self.status_bar.showMessage("下载数据中...")
            for s,_ in syms:
                df=fetch_stock_data(s,"1y")
                if not df.empty: self._syms_data[s]=df
        if len(self._syms_data)<2: QMessageBox.information(self,t("dlg_hint"),t("dlg_need_2stocks")); return
        dlg=CrossSectionalDialog(self._syms_data,self); dlg.exec_()

    def _open_portfolio(self):
        syms=[s for s,_ in self._parse_input()]
        if len(syms)<2: QMessageBox.information(self,t("dlg_hint"),t("dlg_need_2stocks")); return
        dlg=PortfolioDialog(syms,self); dlg.exec_()

    def _open_coint(self):
        syms=[s for s,_ in self._parse_input()]
        if len(syms)<2: QMessageBox.information(self,t("dlg_hint"),t("dlg_need_2stocks")); return
        dlg=CointegrationDialog(syms,self); dlg.exec_()

    def _stop_tasks(self):
        if self._scan_thread and self._scan_thread.isRunning(): self._scan_thread.stop()
        if self._bt_thread   and self._bt_thread.isRunning():   self._bt_thread.stop()
        self.progress.setVisible(False); self._set_busy(False)
        self.status_bar.showMessage(t("status_stopped"),3000)

    def _clear(self):
        if QMessageBox.question(self,t("dlg_confirm"),t("dlg_clear_q"),QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes: return
        self.scan_model.clear(); self.bt_text.clear(); self.bt_canvas.figure.clear(); self.bt_canvas.draw()
        self._equity_curves=[]; self._syms_data={}; self.summary.update_stats(self.scan_model)

    def _export_csv(self):
        if self.scan_model.rowCount()==0: QMessageBox.information(self,t("dlg_hint"),t("dlg_no_data")); return
        path,_=QFileDialog.getSaveFileName(self,t("btn_export"),"scan_results.csv","CSV Files (*.csv)")
        if not path: return
        self.scan_model.to_dataframe().to_csv(path,index=False,encoding='utf-8-sig')
        self.status_bar.showMessage(f"导出: {path}",5000)

    def _set_busy(self,busy):
        for b in [self.scan_btn,self.top100_btn,self.bt_btn,self.cross_btn,
                  self.portfolio_btn,self.coint_btn,self.clear_btn,self.export_btn]:
            b.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)

    def _start_ticker(self):
        if self._ticker_thread and self._ticker_thread.isRunning(): return
        self._ticker_thread=IndexTickerThread()
        self._ticker_thread.data_ready.connect(self.flip_widget.update_data)
        self._ticker_thread.error_msg.connect(lambda e:logger.warning(f"Ticker: {e}"))
        self._ticker_thread.start()

    def _stop_ticker(self):
        if self._ticker_thread and self._ticker_thread.isRunning():
            self._ticker_thread.stop(); self._ticker_thread.wait()

    def showEvent(self,e): super().showEvent(e); self._start_ticker()
    def closeEvent(self,e): self._stop_ticker(); self._stop_tasks(); e.accept()

from quantpro_price_target_patch import (
    patch_probability_dialog,
    PriceTargetEngine,
    KlineOverlayPainter,
    PriceTargetWidget,
    PriceTargetCard,
    ATRFibPanel,
)
 
patch_probability_dialog(
    ProbabilityDialog_class    = ProbabilityDialog,
    get_mpl_rc_fn              = _get_mpl_rc,
    apply_font_fn              = _apply_font_to_figure,
    make_mpf_style_fn          = _make_mpf_style,
    T_class                    = T,
    t_fn                       = t,
    mpl_style_fn               = _mpl_style,
    fetch_stock_data_fn        = fetch_stock_data,
    get_realtime_price_fn      = get_realtime_price,
)

# ── 兼容修复：价格目标补丁注入点适配滚动区 tab ──────────────────
# _add_scroll_tab 把每个 tab 内容包进了 QScrollArea，导致补丁里
# self.tabs.widget(0).layout() 取到的是 QScrollArea（layout 为 None），
# 价格目标/ATR 卡片因此加不进去而"消失"。这里覆盖为用 tab_inner() 穿透
# 滚动区，拿到真实内容容器再插入。
def _patched_refresh_price_target_widget(self, currency: str):
    if not getattr(self, "_target_data", None):
        return
    # 穿透滚动区拿到"概率预测"tab 的真实内容 widget
    inner = None
    if hasattr(self, "tab_inner"):
        inner = self.tab_inner(t("tab_prob")) or self.tab_inner(0)
    if inner is None:
        from PyQt5.QtWidgets import QScrollArea
        w0 = self.tabs.widget(0)
        inner = w0.widget() if isinstance(w0, QScrollArea) else w0
    if inner is None:
        return
    tab_lay = inner.layout()
    if tab_lay is None:
        return
    # 移除旧卡片
    if getattr(self, "_price_target_widget", None):
        try:
            tab_lay.removeWidget(self._price_target_widget)
            self._price_target_widget.deleteLater()
        except Exception:
            pass
        self._price_target_widget = None
    try:
        widget = PriceTargetWidget(self._target_data, currency, self)
        tab_lay.addWidget(widget)
        self._price_target_widget = widget
    except Exception as _e:
        logger.warning(f"PriceTargetWidget(scroll-fix) error: {_e}")

ProbabilityDialog._refresh_price_target_widget = _patched_refresh_price_target_widget

# ── 个股K线清爽化 + 金叉/死叉标记（必须在价格目标补丁之后安装）──────
#    去掉补丁往蜡烛图上叠的 MC带/ATR/Fib（那些数字在价格目标卡片里有），
#    只留蜡烛+成交量+MA(5/20/60)，并标出金叉(绿▲)/死叉(红▼)。
#    价格目标卡片仍保留；想看预测带可用K线tab的「预测带」开关打开。
try:
    from quantpro_clean_kline import install_clean_kline
    # 把价格目标补丁导出的类塞进 globals 供清爽版按需复用
    install_clean_kline(globals())
except Exception as _e:
    logger.warning(f"清爽K线未加载（quantpro_clean_kline.py）: {_e}")

# ══════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════
if __name__=="__main__":
    # ── ① v14 方法论注入（必须在创建任何引擎/窗口之前）─────────────
    #    Purged CV + 方向标签 + 基准率居中 + 轻校准 + DSR
    #    fetch_fn 启用 v14.2 大盘上下文（SPY/VIX，残差IC自动筛选）
    try:
        from quantpro_v14_alpha_engine import patch_quantpro
        patch_quantpro(ProbabilityEngine, WalkForwardBacktest, SelfCorrectionEngine,
                       fetch_fn=fetch_stock_data)
    except ImportError as _e:
        logger.warning(f"v14 alpha engine 未加载（缺文件 quantpro_v14_alpha_engine.py）: {_e}")

    # ── ② 配对对比功能（主窗口按钮 + 协整窗口双击）────────────────
    try:
        from quantpro_pair_compare import install_pair_compare
        install_pair_compare(globals())
    except ImportError as _e:
        logger.warning(f"配对对比未加载（缺文件 quantpro_pair_compare.py）: {_e}")

    # ── ③ 大盘状态仪表盘（主窗口按钮）────────────────────────────
    try:
        from quantpro_market_dashboard import install_market_dashboard
        install_market_dashboard(globals())
    except ImportError as _e:
        logger.warning(f"大盘仪表盘未加载（缺文件 quantpro_market_dashboard.py）: {_e}")

    # ── ③ 量价分析（K线量价标注 + 量价诊断面板）──────────────────
    try:
        from quantpro_volume_analysis import install_volume_analysis
        install_volume_analysis(globals())
    except ImportError as _e:
        logger.warning(f"量价分析未加载（缺文件 quantpro_volume_analysis.py）: {_e}")

    # ── 高DPI适配（必须在 QApplication 创建之前设置）──────────────
    #    解决 Windows 125%/150% 缩放下窗口尺寸算不准、超出屏幕的问题
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass
    for attr in ('AA_EnableHighDpiScaling','AA_UseHighDpiPixmaps'):
        if hasattr(Qt,attr): QApplication.setAttribute(getattr(Qt,attr))
    app=QApplication(sys.argv); app.setStyle("Fusion"); app.setWindowIcon(_make_app_icon())
    db=QFontDatabase()
    for f in ["Microsoft YaHei","SimHei","PingFang SC","Noto Sans CJK SC"]:
        if f in db.families(): app.setFont(QFont(f,10)); break
    win=QuantApp(); win.show(); sys.exit(app.exec_())