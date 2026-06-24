"""
QuantPro — 配对对比模块（PairCompareDialog）v1.0
══════════════════════════════════════════════════════════════════
对比任意两个标的的走势与对冲关系（如 KO vs INTC、^GSPC vs ^VIX）。

为什么不是"双K线"：两个标的价格量纲不可比（KO≈60美元、VIX≈15点），
直接叠加K线毫无意义。正确做法（本模块实现）：
  主图   : 归一化走势（起点=100）/ 比价图（A÷B）一键切换
  副图1  : 滚动60日相关系数 —— 对冲关系的量化呈现（VIX vs SPX 应深度负相关）
  副图2  : 对数比价 z-score（滚动60日），±2σ 标色 = 配对交易进出场区
  信息条 : 协整p值 · 全期/近60日相关 · beta对冲比率 · 双边年化波动

集成（在 quantpro_v1_6.py 末尾、if __name__=="__main__": 块内、
创建 QuantApp 之前调用，与 patch_quantpro 并列）：

    from quantpro_pair_compare import install_pair_compare
    install_pair_compare(globals())

两个入口自动生效：
  ① 主窗口按钮行新增「配对对比」按钮（协整按钮右侧）
     —— 若输入框里已有逗号分隔的代码，自动取前两个预填
  ② 协整分析窗口结果列表中【双击】任意一行 "A/B p值:..."
     —— 直接弹出该对的对比窗口

依赖：仅用主程序已有的 PyQt5 / matplotlib / pandas / numpy。
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import re
import logging
from typing import Optional, Callable

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QApplication, QWidget
)
from PyQt5.QtCore import Qt, QTimer, QObject, QEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib as mpl
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 计算层（纯函数，便于单测）
# ══════════════════════════════════════════════════════════════════
def compute_pair_stats(ca: pd.Series, cb: pd.Series,
                       corr_win: int = 60,
                       coint_test_fn: Optional[Callable] = None) -> dict:
    """
    对齐两条收盘价序列并计算全部对比统计量。
    返回 dict：normA/normB, ratio, roll_corr, ratio_z, 以及标量指标。
    """
    df = pd.concat({"a": ca.astype(float), "b": cb.astype(float)}, axis=1).dropna()
    if len(df) < 70:
        return {"ok": False, "msg": f"重叠交易日不足（{len(df)} < 70）"}
    a, b = df["a"], df["b"]
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    common = ra.index.intersection(rb.index)
    ra, rb = ra.loc[common], rb.loc[common]

    # 归一化走势（起点=100）
    norm_a = a / a.iloc[0] * 100.0
    norm_b = b / b.iloc[0] * 100.0

    # 比价 与 对数比价滚动 z-score
    ratio = a / b
    log_ratio = np.log(ratio)
    rm = log_ratio.rolling(corr_win).mean()
    rs = log_ratio.rolling(corr_win).std()
    ratio_z = (log_ratio - rm) / (rs + 1e-12)

    # 滚动相关
    roll_corr = ra.rolling(corr_win).corr(rb)

    # 标量指标
    corr_full = float(ra.corr(rb)) if len(ra) > 2 else np.nan
    corr_recent = float(roll_corr.dropna().iloc[-1]) if roll_corr.notna().any() else np.nan
    var_b = float(rb.var())
    beta = float(ra.cov(rb) / var_b) if var_b > 0 else np.nan   # A 对 B 的对冲比率
    ann = np.sqrt(252.0)
    vol_a = float(ra.std() * ann * 100)
    vol_b = float(rb.std() * ann * 100)

    coint_p = np.nan
    if coint_test_fn is not None:
        try:
            res = coint_test_fn(a, b)          # CointegrationAnalysis.test_pair
            # 兼容 (pvalue, spread) / (stat, pvalue, ...) / 标量 等返回形态
            if isinstance(res, (tuple, list)):
                for v in res:
                    if isinstance(v, (int, float, np.floating)) and 0 <= float(v) <= 1:
                        coint_p = float(v); break
            elif isinstance(res, (int, float, np.floating)):
                coint_p = float(res)
        except Exception as e:
            logger.warning(f"coint test: {e}")

    return {
        "ok": True, "index": df.index,
        "norm_a": norm_a, "norm_b": norm_b,
        "ratio": ratio, "ratio_z": ratio_z, "roll_corr": roll_corr,
        "corr_full": corr_full, "corr_recent": corr_recent,
        "beta": beta, "vol_a": vol_a, "vol_b": vol_b,
        "coint_p": coint_p, "n": len(df), "corr_win": corr_win,
    }


# ══════════════════════════════════════════════════════════════════
# 对比窗口
# ══════════════════════════════════════════════════════════════════
class PairCompareDialog(QDialog):
    """
    双标的走势/对冲关系对比窗口。
    env: 主程序 globals()，需要 fetch_stock_data / T / GLOBAL_STYLE /
         _make_app_icon / _get_mpl_rc / _apply_font_to_figure / _mpl_style /
         CointegrationAnalysis（均为主文件已有对象）
    """

    PERIODS = ["6mo", "1y", "2y", "5y"]

    def __init__(self, env: dict, sym_a: str = "", sym_b: str = "", parent=None):
        super().__init__(parent)
        self.env = env
        T = env["T"]
        self.setWindowTitle("配对对比分析")
        try:
            self.setWindowIcon(env["_make_app_icon"]())
        except Exception:
            pass
        self.resize(980, 760)
        # 补最小化/最大化按钮，去掉无用的帮助(?)按钮
        try:
            _f = self.windowFlags()
            _f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
            _f &= ~Qt.WindowContextHelpButtonHint
            self.setWindowFlags(_f)
        except Exception:
            pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self._stats: Optional[dict] = None
        self._mode_ratio = False          # False=归一化, True=比价
        self._build_ui(T, sym_a, sym_b)
        if sym_a and sym_b:
            QTimer.singleShot(150, self._run)

    # ── UI ───────────────────────────────────────────────────────
    def _build_ui(self, T, sym_a: str, sym_b: str):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12); lay.setSpacing(8)

        hdr = QLabel("<b>配对对比分析</b>　归一化走势 · 滚动相关 · 比价z-score")
        hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;padding:4px;border:none;")
        lay.addWidget(hdr)

        # 输入行
        row = QHBoxLayout(); row.setSpacing(6)
        self.ed_a = QLineEdit(sym_a); self.ed_a.setPlaceholderText("标的A 如 KO / ^GSPC")
        self.ed_b = QLineEdit(sym_b); self.ed_b.setPlaceholderText("标的B 如 INTC / ^VIX")
        self.period_cb = QComboBox(); self.period_cb.addItems(self.PERIODS)
        self.period_cb.setCurrentText("1y")
        self.run_btn = QPushButton("对比")
        self.run_btn.setStyleSheet(self.run_btn.styleSheet() +
                                   f"color:{T.CYAN};border-color:{T.CYAN};")
        self.mode_btn = QPushButton("切换: 比价图")
        self.mode_btn.setStyleSheet(self.mode_btn.styleSheet() +
                                    f"color:{T.PURPLE};border-color:{T.PURPLE};")
        for w, st in [(QLabel("A:"), 0), (self.ed_a, 1), (QLabel("B:"), 0),
                      (self.ed_b, 1), (self.period_cb, 0),
                      (self.run_btn, 0), (self.mode_btn, 0)]:
            row.addWidget(w, st)
        lay.addLayout(row)

        self.status = QLabel("输入两个代码后点击「对比」")
        self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
        lay.addWidget(self.status)

        self.canvas = FigureCanvas(Figure(figsize=(9, 6), facecolor=T.MPL_BG))
        lay.addWidget(self.canvas, 1)

        self.info = QLabel(); self.info.setWordWrap(True)
        self.info.setStyleSheet(f"color:{T.CYAN};font-size:9.5pt;padding:6px;"
                                f"background:{T.BG2};border-radius:6px;")
        lay.addWidget(self.info)

        self.run_btn.clicked.connect(self._run)
        self.mode_btn.clicked.connect(self._toggle_mode)
        self.ed_a.returnPressed.connect(self._run)
        self.ed_b.returnPressed.connect(self._run)

    # ── 数据 + 统计 ──────────────────────────────────────────────
    def _run(self):
        env = self.env
        sa = self.ed_a.text().strip().upper()
        sb = self.ed_b.text().strip().upper()
        if not sa or not sb:
            self.status.setText("请填写两个代码"); return
        if sa == sb:
            self.status.setText("两个代码不能相同"); return
        period = self.period_cb.currentText()
        self.status.setText(f"拉取 {sa} 与 {sb} ({period}) ...")
        QApplication.processEvents()

        fetch = env["fetch_stock_data"]
        try:
            dfa = fetch(sa, period); dfb = fetch(sb, period)
        except Exception as e:
            self.status.setText(f"数据拉取失败: {e}"); return
        if dfa is None or dfb is None or dfa.empty or dfb.empty:
            self.status.setText("某个代码无数据（指数请用 ^GSPC / ^VIX 格式）"); return

        ca = dfa["Close"].squeeze().astype(float)
        cb = dfb["Close"].squeeze().astype(float)

        coint_fn = None
        CA = env.get("CointegrationAnalysis")
        if CA is not None and hasattr(CA, "test_pair"):
            coint_fn = CA.test_pair

        st = compute_pair_stats(ca, cb, corr_win=60, coint_test_fn=coint_fn)
        if not st.get("ok"):
            self.status.setText(st.get("msg", "计算失败")); return
        self._stats = st
        self._sa, self._sb = sa, sb
        self.status.setText(f"{sa} vs {sb}  重叠交易日 {st['n']}  [{period}]")
        self._draw()
        self._fill_info()

    def _toggle_mode(self):
        self._mode_ratio = not self._mode_ratio
        self.mode_btn.setText("切换: 归一化" if self._mode_ratio else "切换: 比价图")
        if self._stats:
            self._draw()

    # ── 绘图 ─────────────────────────────────────────────────────
    def _draw(self):
        env = self.env; T = env["T"]; st = self._stats
        get_rc = env["_get_mpl_rc"]; mpl_style = env["_mpl_style"]
        apply_font = env["_apply_font_to_figure"]
        sa, sb = self._sa, self._sb
        win = st["corr_win"]

        with mpl.rc_context(get_rc()):
            fig = self.canvas.figure; fig.clear()
            fig.patch.set_facecolor(T.MPL_BG)
            gs = gridspec.GridSpec(3, 1, height_ratios=[3.0, 1.3, 1.3],
                                   hspace=0.42, figure=fig)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax3 = fig.add_subplot(gs[2], sharex=ax1)

            # ── 主图 ──
            if not self._mode_ratio:
                ax1.plot(st["norm_a"].index, st["norm_a"].values,
                         color=T.ACCENT, lw=1.4, label=f"{sa} (起点=100)")
                ax1.plot(st["norm_b"].index, st["norm_b"].values,
                         color=T.GOLD, lw=1.4, label=f"{sb} (起点=100)")
                ax1.axhline(100, color=T.TEXT_3, lw=0.8, ls="--", alpha=0.6)
                mpl_style(ax1, title=f"归一化走势  {sa} vs {sb}", ylabel="指数(起点=100)")
            else:
                r = st["ratio"]
                ax1.plot(r.index, r.values, color=T.PURPLE, lw=1.4,
                         label=f"比价 {sa}/{sb}")
                rm = r.rolling(win).mean()
                ax1.plot(rm.index, rm.values, color=T.GOLD, lw=1.0, ls="--",
                         alpha=0.8, label=f"{win}日均值")
                mpl_style(ax1, title=f"比价图  {sa} ÷ {sb}（向上 = {sa} 跑赢）",
                          ylabel="比价")
            ax1.legend(fontsize=7.5, ncol=2, facecolor=T.BG2,
                       labelcolor=T.TEXT_H, edgecolor=T.BORDER, framealpha=0.85)

            # ── 副图1：滚动相关 ──
            rc = st["roll_corr"]
            ax2.plot(rc.index, rc.values, color=T.CYAN, lw=1.1)
            ax2.axhline(0, color=T.TEXT_3, lw=0.8, alpha=0.7)
            ax2.axhline(0.5, color=T.GREEN, lw=0.6, ls=":", alpha=0.6)
            ax2.axhline(-0.5, color=T.RED, lw=0.6, ls=":", alpha=0.6)
            ax2.fill_between(rc.index, 0, rc.values,
                             where=(rc.values < 0), color=T.RED, alpha=0.12)
            ax2.fill_between(rc.index, 0, rc.values,
                             where=(rc.values >= 0), color=T.GREEN, alpha=0.10)
            ax2.set_ylim(-1.05, 1.05)
            mpl_style(ax2, title=f"滚动{win}日相关系数（<0 = 对冲关系成立）",
                      ylabel="相关")

            # ── 副图2：比价 z-score ──
            z = st["ratio_z"]
            ax3.plot(z.index, z.values, color=T.ACCENT, lw=1.0)
            ax3.axhline(0, color=T.GOLD, lw=1.0, ls="--", alpha=0.8)
            for lv, col in [(2, T.RED), (-2, T.GREEN)]:
                ax3.axhline(lv, color=col, lw=0.8, ls=":", alpha=0.8)
            zv = z.values.astype(float)
            ax3.fill_between(z.index, 2, zv, where=(zv > 2), color=T.RED, alpha=0.25)
            ax3.fill_between(z.index, -2, zv, where=(zv < -2), color=T.GREEN, alpha=0.25)
            mpl_style(ax3, title=f"对数比价 z-score（滚动{win}日，|z|>2 = 极端偏离区）",
                      ylabel="z")

            apply_font(fig)
        self.canvas.draw()

    # ── 信息条 ───────────────────────────────────────────────────
    def _fill_info(self):
        st = self._stats; sa, sb = self._sa, self._sb
        p = st["coint_p"]
        if np.isfinite(p):
            coint_txt = f"协整p值 <b>{p:.4f}</b> {'✅协整(可做配对)' if p < 0.05 else '❌无协整'}"
        else:
            coint_txt = "协整p值 --（需 statsmodels）"
        cf, cr = st["corr_full"], st["corr_recent"]
        rel = ("强对冲" if cr < -0.5 else
               "弱对冲/低相关" if cr < 0.2 else
               "同向联动" if cr < 0.7 else "高度同向")
        beta_txt = (f"beta(对冲比率) <b>{st['beta']:.2f}</b>"
                    f"（做多1份{sa} ≈ 用{abs(st['beta']):.2f}份{sb}对冲）"
                    if np.isfinite(st["beta"]) else "beta --")
        zlast = st["ratio_z"].dropna()
        z_txt = f"当前比价z <b>{float(zlast.iloc[-1]):+.2f}</b>" if len(zlast) else "比价z --"
        self.info.setText(
            f"{coint_txt}　|　全期相关 <b>{cf:+.2f}</b> · 近60日 <b>{cr:+.2f}</b>"
            f" → <b>{rel}</b>　|　{beta_txt}　|　{z_txt}　|　"
            f"年化波动 {sa} {st['vol_a']:.0f}% / {sb} {st['vol_b']:.0f}%"
            f"<br><span style='font-size:8pt;'>* 统计描述仅供研究参考，"
            f"相关性与协整关系会随行情变化，不构成投资建议。</span>"
        )


# ══════════════════════════════════════════════════════════════════
# 入口② —— 协整窗口结果区双击打开对比
# ══════════════════════════════════════════════════════════════════
_PAIR_LINE_RE = re.compile(r"^\s*(\S+)\s*/\s*(\S+)\s+p值")


class _CointDblClickFilter(QObject):
    """安装到 CointegrationDialog.text 上：双击行 'A/B p值:...' → 打开对比窗口"""

    def __init__(self, env: dict, dialog):
        super().__init__(dialog)
        self.env = env
        self.dialog = dialog

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            try:
                cursor = obj.cursorForPosition(event.pos())
                line = cursor.block().text()
                m = _PAIR_LINE_RE.match(line)
                if m:
                    sa, sb = m.group(1), m.group(2)
                    dlg = PairCompareDialog(self.env, sa, sb, parent=self.dialog)
                    dlg.show()
                    return True
            except Exception as e:
                logger.warning(f"coint dblclick: {e}")
        return False


# ══════════════════════════════════════════════════════════════════
# 一键安装两个入口
# ══════════════════════════════════════════════════════════════════
def install_pair_compare(env: dict):
    """
    env = 主程序的 globals()。安装：
      ① QuantApp 按钮行（协整按钮右侧）新增「配对对比」
      ② CointegrationDialog 结果区双击行打开对比窗口
    必须在创建 QuantApp() 之前调用。
    """
    required = ["QuantApp", "CointegrationDialog", "fetch_stock_data", "T",
                "_make_app_icon", "_get_mpl_rc", "_apply_font_to_figure",
                "_mpl_style"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_pair_compare: 主程序缺少 {missing}，"
                           f"请传入 globals()")

    QuantApp_cls = env["QuantApp"]
    CointDialog_cls = env["CointegrationDialog"]
    T = env["T"]

    # ── ① 主窗口按钮 ────────────────────────────────────────────
    _orig_app_init = QuantApp_cls.__init__

    def _app_init(self, *a, **kw):
        _orig_app_init(self, *a, **kw)
        try:
            from PyQt5.QtWidgets import QPushButton, QHBoxLayout
            btn = QPushButton("配对对比")
            btn.setStyleSheet(btn.styleSheet() +
                              f"color:{T.ORANGE};border-color:{T.ORANGE};")
            inserted = False
            # 首选：主程序暴露的 self.btn_row（插在末尾 stretch 之前）
            row = getattr(self, "btn_row", None)
            if isinstance(row, QHBoxLayout):
                idx = row.indexOf(self.coint_btn)
                if idx != -1:
                    row.insertWidget(idx + 1, btn)
                else:
                    row.insertWidget(max(0, row.count() - 1), btn)
                inserted = True
            if not inserted:
                # 兼容旧版主程序：在 root 里搜索包含 coint_btn 的行
                root = self.layout()
                for i in range(root.count()):
                    item = root.itemAt(i)
                    r = item.layout()
                    if isinstance(r, QHBoxLayout) and r.indexOf(self.coint_btn) != -1:
                        r.insertWidget(r.indexOf(self.coint_btn) + 1, btn)
                        inserted = True
                        break
            if not inserted:
                self.layout().addWidget(btn)

            def _open_pair():
                # 输入框里有逗号分隔代码 → 取前两个预填
                sa = sb = ""
                txt = self.input_edit.text().strip()
                if txt:
                    parts = [p.strip().upper() for p in
                             re.split(r"[,，;\s]+", txt) if p.strip()]
                    if len(parts) >= 1: sa = parts[0]
                    if len(parts) >= 2: sb = parts[1]
                dlg = PairCompareDialog(env, sa, sb, parent=self)
                dlg.show()

            btn.clicked.connect(_open_pair)
            self.pair_btn = btn
        except Exception as e:
            logger.warning(f"pair button install: {e}")

    QuantApp_cls.__init__ = _app_init

    # ── ② 协整窗口双击 ──────────────────────────────────────────
    _orig_coint_init = CointDialog_cls.__init__

    def _coint_init(self, *a, **kw):
        _orig_coint_init(self, *a, **kw)
        try:
            f = _CointDblClickFilter(env, self)
            self.text.installEventFilter(f)
            self._pair_dbl_filter = f             # 持引用防GC
            # 提示文案
            tip = self.status.text()
            self.status.setText(tip + "　（双击结果行可打开走势对比）"
                                if tip else "双击结果行可打开走势对比")
        except Exception as e:
            logger.warning(f"coint dblclick install: {e}")

    CointDialog_cls.__init__ = _coint_init

    logger.info("[pair_compare] 已安装：主窗口「配对对比」按钮 + 协整窗口双击入口")
    return PairCompareDialog


# ══════════════════════════════════════════════════════════════════
# 独立自检（offscreen，无需主程序）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # 合成一对负相关序列（模拟 SPX vs VIX）
    rng = np.random.default_rng(8)
    n = 380
    idx = pd.bdate_range("2024-06-03", periods=n)
    spx_ret = rng.normal(0.0004, 0.009, n)
    spx = pd.Series(4500 * np.cumprod(1 + spx_ret), index=idx)
    vix = pd.Series(np.clip(16 - spx_ret * 900 +
                            rng.normal(0, 0.9, n).cumsum() * 0.15, 9, 60), index=idx)

    st = compute_pair_stats(spx, vix)
    print("compute_pair_stats:",
          f"n={st['n']}  全期相关={st['corr_full']:+.2f}（应为负）",
          f"近60日={st['corr_recent']:+.2f}  beta={st['beta']:+.1f}")
    assert st["ok"] and st["corr_full"] < 0, "负相关对的统计方向错误"

    # 最小 env + 对话框冒烟（offscreen 渲染三联图）
    class _T:
        MPL_BG = "#0d1826"; BG2 = "#111f30"; BORDER = "#1e3452"
        GOLD = "#f59e0b"; ACCENT = "#0ea5e9"; CYAN = "#06b6d4"
        PURPLE = "#a78bfa"; GREEN = "#10b981"; RED = "#ef4444"
        ORANGE = "#f97316"; TEXT_H = "#e2eaf3"; TEXT_2 = "#8ba3be"
        TEXT_3 = "#4e6a82"

    def _fetch(sym, period="1y"):
        return pd.DataFrame({"Close": spx if "GSPC" in sym else vix})

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(_T.MPL_BG)
        ax.set_title(title, color=_T.TEXT_H, fontsize=9)
        ax.tick_params(colors=_T.TEXT_2, labelsize=7)
        for s in ax.spines.values(): s.set_color(_T.BORDER)

    env = {"QuantApp": None, "CointegrationDialog": None,
           "fetch_stock_data": _fetch, "T": _T, "GLOBAL_STYLE": "",
           "_make_app_icon": lambda: None,
           "_get_mpl_rc": lambda: {}, "_mpl_style": _style,
           "_apply_font_to_figure": lambda fig: None,
           "CointegrationAnalysis": None}

    app = QApplication(sys.argv)
    dlg = PairCompareDialog(env, "^GSPC", "^VIX")
    dlg._run()
    n_axes = len(dlg.canvas.figure.axes)
    print(f"Dialog冒烟: 子图数={n_axes}（应=3） 信息条非空={bool(dlg.info.text())}")
    dlg._toggle_mode(); dlg._draw()
    print("比价模式切换: OK")

    # 双击行解析
    m = _PAIR_LINE_RE.match("KO/INTC  p值:0.0312  ✅ 协整")
    print(f"协整行解析: {m.group(1)}/{m.group(2)}（应=KO/INTC）")
    assert m and m.group(1) == "KO" and m.group(2) == "INTC"
    print("全部自检通过 ✓")
