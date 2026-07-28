"""
QuantPro — 模拟自选 / 建仓模拟器（PaperWatchlistDialog）v1.0
══════════════════════════════════════════════════════════════════
需求：假设在当前时点以某个价格"买入"雪佛龙(CVX)、微软(MSFT)、英特尔(INTC)、
      可口可乐(KO)、洛克希德马丁(LMT)等标的若干股，记录下来；过几天/几个月
      后回来看，对比实际价格变化，算出涨跌幅——并且能同时维护"几套方案"
      互相对比（例如"防御组合" vs "科技组合" vs "等权五巨头"）。

设计：
  ① 数据落地 SQLite（quantpro_paper.db，与 SelfCorrectionEngine 的
     quantpro_predictions.db 同目录同套路），程序关掉/换天重开都还在。
  ② 「方案」= 一组模拟持仓的容器，可以建任意多套，互相独立对比。
  ③ 每套方案里可以逐笔手动加仓（自选代码 + 数量 + 买入价，价格可一键
     取当前实时价，也可手工改成"如果当时是XX价买的"做回溯模拟）。
  ④ 「一键建仓」：勾几只预设/自定义标的 + 一个总预算，等金额或等股数
     自动拆分买入价，秒建一套方案——适合快速起好几套方案做对比。
  ⑤ 「刷新」拉最新价，算每笔/整套的盈亏与盈亏%，同时把当天的总市值
     写一条快照（同一天只保留一条），攒够几天/几个月后画出资金曲线，
     一眼看出这套方案跑赢/跑输大盘（同期基准用 ^GSPC 归一化叠加对比）。

集成（quantpro_v1_6.py 末尾、install_market_dashboard(...) 之后、
入口 QuantApp() 创建之前调用）：

    from quantpro_paper_watchlist import install_paper_watchlist
    install_paper_watchlist(globals())

入口：主窗口按钮行新增「模拟自选」按钮。

依赖：仅用主程序已有的 PyQt5 / matplotlib / pandas / numpy / sqlite3。
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QDoubleSpinBox,
    QDateEdit, QListWidget, QListWidgetItem, QMessageBox, QInputDialog,
    QTabWidget, QWidget, QAbstractItemView, QSplitter, QSpinBox, QApplication,
    QProgressBar, QTextEdit, QButtonGroup,
)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    import mplfinance as mpf
    HAS_MPF = True
except ImportError:
    HAS_MPF = False

logger = logging.getLogger(__name__)

# 五个用户点名的标的（可在弹窗里自由增删，这只是默认建议清单）
PRESET_STOCKS = [
    ("CVX", "雪佛龙"),
    ("MSFT", "微软"),
    ("INTC", "英特尔"),
    ("KO", "可口可乐"),
    ("LMT", "洛克希德马丁"),
]

# 历史走势Tab的周期按钮：跟主程序 _CHG_PERIODS 用同一套交易日换算口径，
# 保证"3年"这种yfinance本身不直接支持的周期，切出来的区间跟主界面
# 别的地方看到的涨跌幅是同一个算法、对得上号。None=全部历史。
HIST_PERIODS = [
    ("1天", 1), ("5天", 5), ("1月", 21), ("3月", 63),
    ("1年", 252), ("3年", 756), ("5年", 1260), ("全部", None),
]

# 持仓明细表的排序方式（默认涨幅%从高到低——一眼看到今天谁涨得最好）
SORT_MODES = ["涨幅%（高→低）", "涨幅%（低→高）", "仓位占比（大→小）", "盈亏金额（高→低）", "股票代码（A→Z）"]


def _slice_period(df: pd.DataFrame, days: Optional[int]) -> pd.DataFrame:
    """按交易日数从df尾部切片；days=None返回全部。纯函数，不依赖Qt/网络。"""
    if df is None or df.empty:
        return df
    if days is None or len(df) <= days + 1:
        return df
    return df.iloc[-(days + 1):]


# ══════════════════════════════════════════════════════════════════
# 数据层：SQLite 持久化（方案 / 持仓 / 每日快照）
# ══════════════════════════════════════════════════════════════════
class PaperWatchlistStore:
    _DB_PATH: str = "quantpro_paper.db"
    _LOCK = threading.Lock()

    @classmethod
    def _conn(cls):
        return sqlite3.connect(cls._DB_PATH)

    @classmethod
    def init_db(cls):
        with cls._LOCK:
            try:
                conn = cls._conn(); c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS paper_plans (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        name         TEXT NOT NULL,
                        created_date TEXT NOT NULL,
                        note         TEXT
                    )
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS paper_positions (
                        id         INTEGER PRIMARY KEY AUTOINCREMENT,
                        plan_id    INTEGER NOT NULL,
                        symbol     TEXT NOT NULL,
                        name       TEXT,
                        qty        REAL NOT NULL,
                        buy_price  REAL NOT NULL,
                        buy_date   TEXT NOT NULL,
                        currency   TEXT DEFAULT '$'
                    )
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS paper_snapshots (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        plan_id     INTEGER NOT NULL,
                        snap_date   TEXT NOT NULL,
                        total_cost  REAL,
                        total_value REAL,
                        pnl_pct     REAL,
                        UNIQUE(plan_id, snap_date)
                    )
                """)
                c.execute("""
                    CREATE TABLE IF NOT EXISTS paper_config (
                        key   TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                conn.commit()
            except Exception as e:
                logger.warning(f"PaperWatchlist DB init: {e}")
            finally:
                try: conn.close()
                except Exception: pass

    # ── 方案 ──────────────────────────────────────────────────
    @classmethod
    def list_plans(cls) -> List[Dict]:
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("SELECT id,name,created_date,note FROM paper_plans ORDER BY id DESC")
                rows = c.fetchall()
            finally:
                conn.close()
        return [{"id": r[0], "name": r[1], "created_date": r[2], "note": r[3]} for r in rows]

    @classmethod
    def create_plan(cls, name: str, note: str = "") -> int:
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("INSERT INTO paper_plans(name,created_date,note) VALUES(?,?,?)",
                          (name, date.today().isoformat(), note))
                conn.commit()
                return c.lastrowid
            finally:
                conn.close()

    @classmethod
    def delete_plan(cls, plan_id: int):
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("DELETE FROM paper_positions WHERE plan_id=?", (plan_id,))
                c.execute("DELETE FROM paper_snapshots WHERE plan_id=?", (plan_id,))
                c.execute("DELETE FROM paper_plans WHERE id=?", (plan_id,))
                conn.commit()
            finally:
                conn.close()

    # ── 持仓 ──────────────────────────────────────────────────
    @classmethod
    def add_position(cls, plan_id: int, symbol: str, name: str, qty: float,
                     buy_price: float, buy_date: str, currency: str = "$"):
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""INSERT INTO paper_positions
                    (plan_id,symbol,name,qty,buy_price,buy_date,currency)
                    VALUES(?,?,?,?,?,?,?)""",
                    (plan_id, symbol.upper(), name, qty, buy_price, buy_date, currency))
                conn.commit()
            finally:
                conn.close()

    @classmethod
    def delete_position(cls, pos_id: int):
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("DELETE FROM paper_positions WHERE id=?", (pos_id,))
                conn.commit()
            finally:
                conn.close()

    @classmethod
    def update_position(cls, pos_id: int, qty: float, buy_price: float, buy_date: str):
        """编辑已建仓的一笔：改数量/买入价/买入日期（用于建仓后发现数目填错要改）。"""
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""UPDATE paper_positions SET qty=?, buy_price=?, buy_date=?
                             WHERE id=?""", (qty, buy_price, buy_date, pos_id))
                conn.commit()
            finally:
                conn.close()

    @classmethod
    def list_positions(cls, plan_id: int) -> List[Dict]:
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""SELECT id,symbol,name,qty,buy_price,buy_date,currency
                             FROM paper_positions WHERE plan_id=? ORDER BY id""", (plan_id,))
                rows = c.fetchall()
            finally:
                conn.close()
        keys = ["id", "symbol", "name", "qty", "buy_price", "buy_date", "currency"]
        return [dict(zip(keys, r)) for r in rows]

    # ── 快照（资金曲线用）────────────────────────────────────
    @classmethod
    def record_snapshot(cls, plan_id: int, total_cost: float, total_value: float):
        pnl_pct = ((total_value - total_cost) / total_cost * 100.0) if total_cost else 0.0
        today = date.today().isoformat()
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""INSERT INTO paper_snapshots(plan_id,snap_date,total_cost,total_value,pnl_pct)
                             VALUES(?,?,?,?,?)
                             ON CONFLICT(plan_id,snap_date)
                             DO UPDATE SET total_cost=excluded.total_cost,
                                           total_value=excluded.total_value,
                                           pnl_pct=excluded.pnl_pct""",
                          (plan_id, today, total_cost, total_value, pnl_pct))
                conn.commit()
            finally:
                conn.close()

    @classmethod
    def list_snapshots(cls, plan_id: int) -> List[Dict]:
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""SELECT snap_date,total_cost,total_value,pnl_pct
                             FROM paper_snapshots WHERE plan_id=? ORDER BY snap_date""", (plan_id,))
                rows = c.fetchall()
            finally:
                conn.close()
        keys = ["snap_date", "total_cost", "total_value", "pnl_pct"]
        return [dict(zip(keys, r)) for r in rows]

    # ── 通用配置（目前用于存 AI大模型API设置）────────────────
    @classmethod
    def get_config(cls, key: str, default: str = "") -> str:
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("SELECT value FROM paper_config WHERE key=?", (key,))
                row = c.fetchone()
                return row[0] if row and row[0] is not None else default
            finally:
                conn.close()

    @classmethod
    def set_config(cls, key: str, value: str):
        with cls._LOCK:
            conn = cls._conn(); c = conn.cursor()
            try:
                c.execute("""INSERT INTO paper_config(key,value) VALUES(?,?)
                             ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (key, value))
                conn.commit()
            finally:
                conn.close()


# ══════════════════════════════════════════════════════════════════
# AI 趋势分析层：复用主程序已有的 ProbabilityEngine（含 v14 增强补丁的话
# 就是 Purged CV + 样本唯一性权重 + isotonic 校准 + 多种子bagging + 大盘
# 上下文那一整套），不重新发明轮子——这里只是把它对准"你手里这几只模拟
# 持仓"跑一遍，并结合买入成本给出人话版的持仓解读。
#
# 说明放在前面：这不是文艺复兴科技的 Medallion，那是几十年、海量数据、
# 数千维另类因子外加顶尖团队的成果，任何单机开源方案都学不来。这里做的
# 是同一大类方法论的一个诚实、小型版本——概率预测 + 不确定性 + 历史命中
# 率追踪，仅供参考，不是能稳定赚钱的黑箱。
# ══════════════════════════════════════════════════════════════════

# 综合信号阈值（5/20/60日概率按短中长权重合成后判定）
_SIGNAL_BANDS = [
    (0.62, "强烈看多"), (0.55, "看多"), (0.45, "中性偏多"),
    (0.38, "中性偏空"), (0.0, "看空"),
]


def _signal_label(composite: float) -> Tuple[str, str]:
    """综合概率 → (文字, 颜色key)。颜色key用 'GREEN'/'RED'/'TEXT_2' 去 T 里取。"""
    if composite >= 0.62: return "强烈看多", "GREEN"
    if composite >= 0.55: return "看多", "GREEN"
    if composite >= 0.45: return "中性", "TEXT_2"
    if composite >= 0.38: return "看空", "RED"
    return "强烈看空", "RED"


def _position_advice(signal_label: str, pnl_pct: Optional[float]) -> str:
    """结合浮盈/浮亏状态 + AI信号，给一句人话版解读（不是投资建议）。"""
    if pnl_pct is None:
        bullish = signal_label in ("强烈看多", "看多")
        bearish = signal_label in ("强烈看空", "看空")
        if bullish: return "AI偏多，可关注建仓时机"
        if bearish: return "AI偏空，暂缓追高"
        return "AI中性，观察为主"
    profit = pnl_pct >= 0
    bullish = signal_label in ("强烈看多", "看多")
    bearish = signal_label in ("强烈看空", "看空")
    if profit and bullish:   return "浮盈+AI看多 → 趋势未变，可持有"
    if profit and bearish:   return "浮盈但AI转弱 → 注意保护利润"
    if (not profit) and bullish:  return "浮亏但AI看多 → 模型判断趋势未破，可观察"
    if (not profit) and bearish:  return "浮亏+AI看空 → 风险叠加，建议重新评估"
    return "AI中性 → 维持观察，不必急于动作"


def _aggregate_positions_by_symbol(positions: List[Dict]) -> Dict[str, Dict]:
    """同一方案里同一代码可能分多笔买入，按数量加权算出平均成本，供AI分析用。"""
    agg: Dict[str, Dict] = {}
    for p in positions:
        sym = p["symbol"]
        a = agg.setdefault(sym, {"name": p.get("name") or "", "qty": 0.0, "cost_sum": 0.0})
        a["qty"] += p["qty"]
        a["cost_sum"] += p["qty"] * p["buy_price"]
        if not a["name"] and p.get("name"): a["name"] = p["name"]
    for sym, a in agg.items():
        a["avg_cost"] = (a["cost_sum"] / a["qty"]) if a["qty"] else 0.0
    return agg


class AIAnalysisThread(QThread):
    """
    对一批代码逐个跑：技术因子 → ProbabilityEngine 集成概率(5/20/60日)
    → Monte Carlo 20日价格带 → 特征重要性 → 历史命中率(若DB里已有数据)。
    每完成一只发一次 result_ready，避免标的多时界面卡死等全部跑完。
    """
    progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(dict)
    finished_all = pyqtSignal(int)
    error_msg = pyqtSignal(str)

    def __init__(self, symbols: List[Tuple[str, Dict]], env: dict):
        """symbols: [(sym, {"name":..,"qty":..,"avg_cost":..}), ...]"""
        super().__init__()
        self.symbols = symbols
        self.env = env
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        total = len(self.symbols); done = ok = 0
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(self._analyze, sym, meta): sym for sym, meta in self.symbols}
            for fut in as_completed(futs):
                if self._stop: break
                sym = futs[fut]
                try:
                    d = fut.result()
                    if d: self.result_ready.emit(d); ok += 1
                except Exception as e:
                    self.error_msg.emit(f"{sym}: {e}")
                done += 1
                self.progress.emit(int(done / total * 100), sym)
        self.finished_all.emit(ok)

    def _analyze(self, sym: str, meta: Dict) -> Optional[Dict]:
        env = self.env
        fetch = env["fetch_stock_data"]; get_rt = env["get_realtime_price"]
        TechnicalIndicators = env["TechnicalIndicators"]
        ProbabilityEngine = env["ProbabilityEngine"]
        SelfCorrectionEngine = env.get("SelfCorrectionEngine")

        df_full = fetch(sym, "10y")
        if df_full is None or df_full.empty or len(df_full) < 260:
            return None
        df = df_full.tail(500) if len(df_full) > 500 else df_full
        ind = TechnicalIndicators.compute_all(df)
        if ind is None or ind.empty or len(ind) < 120:
            return None
        close = ind["close"]

        cur_price = get_rt(sym)
        if cur_price is None:
            cur_price = float(close.iloc[-1])

        pe = ProbabilityEngine(ind, close)
        regime = pe.detect_regime()

        probs: Dict[int, float] = {}
        for h in (5, 20, 60):
            try:
                base, p1, p2, p3, w1, w2, w3 = pe.ensemble(fwd=h, symbol=sym)
            except Exception as e:
                logger.warning(f"ensemble {sym} fwd={h}: {e}")
                base, p1, p2, p3, w1, w2, w3 = 0.5, 0.5, 0.5, 0.5, 0.45, 0.35, 0.20
            probs[h] = base
            if SelfCorrectionEngine is not None:
                try:
                    SelfCorrectionEngine.save_prediction(sym, h, p1, p2, p3, base, cur_price, w1, w2, w3)
                except Exception as e:
                    logger.warning(f"save_prediction {sym}: {e}")

        composite = 0.25 * probs[5] + 0.45 * probs[20] + 0.30 * probs[60]
        sig_text, sig_color = _signal_label(composite)

        feat_imp = []
        try:
            feat_imp = pe.feature_importance(fwd=20)[:5]
        except Exception as e:
            logger.warning(f"feature_importance {sym}: {e}")

        mc = None
        try:
            mc = pe.monte_carlo(days=20, n=3000)
        except Exception as e:
            logger.warning(f"monte_carlo {sym}: {e}")

        acc_report = {}
        if SelfCorrectionEngine is not None:
            try:
                acc_report = SelfCorrectionEngine.get_accuracy_report(sym) or {}
            except Exception:
                acc_report = {}

        qty = meta.get("qty", 0.0); avg_cost = meta.get("avg_cost", 0.0)
        pnl_pct = ((cur_price - avg_cost) / avg_cost * 100.0) if avg_cost else None
        advice = _position_advice(sig_text, pnl_pct)

        return {
            "symbol": sym, "name": meta.get("name") or "",
            "cur_price": cur_price, "currency": env["_currency"](sym),
            "qty": qty, "avg_cost": avg_cost, "pnl_pct": pnl_pct,
            "regime": {"high_vol": "高波动", "low_vol": "低波动", "neutral": "中性"}.get(regime, regime),
            "probs": probs, "composite": composite,
            "signal_text": sig_text, "signal_color": sig_color,
            "feat_imp": feat_imp, "mc": mc, "acc_report": acc_report,
            "advice": advice,
        }


# ══════════════════════════════════════════════════════════════════
# 「一键建仓」子弹窗：勾选标的 + 预算/权重方式 → 秒建一套方案
# ══════════════════════════════════════════════════════════════════
class QuickBuildDialog(QDialog):
    def __init__(self, env: dict, parent=None):
        super().__init__(parent)
        self.env = env
        T = env["T"]
        self.setWindowTitle("一键建仓")
        try: self.setWindowIcon(env["_make_app_icon"]())
        except Exception: pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self.resize(560, 420)
        self._get_rt = env["get_realtime_price"]

        v = QVBoxLayout(self)
        tip = QLabel("勾选要建仓的标的，设置总预算和分配方式，会以【当前实时价】模拟买入。\n"
                     "同一批标的可以用不同预算/权重多建几次，形成不同方案互相对比。")
        tip.setWordWrap(True); tip.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        v.addWidget(tip)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("方案名称:"))
        self.name_edit = QLineEdit(f"方案-{datetime.now().strftime('%m%d-%H%M')}")
        name_row.addWidget(self.name_edit, 1)
        v.addLayout(name_row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "代码", "名称"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self._chk_items: Dict[int, QTableWidgetItem] = {}
        for sym, name in PRESET_STOCKS:
            self._add_row(sym, name, checked=True)
        v.addWidget(self.table, 1)

        add_row = QHBoxLayout()
        self.custom_edit = QLineEdit(); self.custom_edit.setPlaceholderText("自定义代码，如 NVDA")
        add_btn = QPushButton("加入列表"); add_btn.clicked.connect(self._add_custom)
        add_row.addWidget(self.custom_edit, 1); add_row.addWidget(add_btn)
        v.addLayout(add_row)

        budget_row = QHBoxLayout()
        budget_row.addWidget(QLabel("总预算($):"))
        self.budget_spin = QDoubleSpinBox(); self.budget_spin.setRange(100, 100_000_000)
        self.budget_spin.setDecimals(0); self.budget_spin.setSingleStep(1000); self.budget_spin.setValue(50000)
        budget_row.addWidget(self.budget_spin)
        budget_row.addWidget(QLabel("分配方式:"))
        self.mode_cb = QComboBox(); self.mode_cb.addItems(["等金额买入", "等股数买入"])
        budget_row.addWidget(self.mode_cb)
        v.addLayout(budget_row)

        btn_row = QHBoxLayout()
        ok_btn = QPushButton("建仓（取当前实时价）")
        ok_btn.setStyleSheet(f"background:{T.GREEN};color:white;")
        ok_btn.clicked.connect(self._build)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(ok_btn)
        v.addLayout(btn_row)

        self.result_plan_id: Optional[int] = None

    def _add_row(self, sym: str, name: str, checked: bool):
        r = self.table.rowCount(); self.table.insertRow(r)
        chk = QTableWidgetItem(); chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chk.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.table.setItem(r, 0, chk)
        self.table.setItem(r, 1, QTableWidgetItem(sym.upper()))
        self.table.setItem(r, 2, QTableWidgetItem(name))

    def _add_custom(self):
        s = self.custom_edit.text().strip().upper()
        if not s: return
        self._add_row(s, "", checked=True)
        self.custom_edit.clear()

    def _build(self):
        rows = []
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).checkState() == Qt.Checked:
                sym = self.table.item(r, 1).text().strip().upper()
                nm = self.table.item(r, 2).text().strip()
                if sym: rows.append((sym, nm))
        if not rows:
            QMessageBox.information(self, "提示", "请至少勾选一只标的"); return
        name = self.name_edit.text().strip() or f"方案-{datetime.now().strftime('%m%d-%H%M')}"

        # 拉实时价（拉不到的用历史收盘兜底）
        prices = {}
        fetch = self.env["fetch_stock_data"]
        for sym, _ in rows:
            p = self._get_rt(sym)
            if p is None:
                try:
                    df = fetch(sym, "5d")
                    if df is not None and not df.empty:
                        p = float(df["Close"].iloc[-1])
                except Exception:
                    p = None
            prices[sym] = p
        missing = [s for s, p in prices.items() if not p]
        if missing:
            QMessageBox.warning(self, "部分标的取价失败",
                                "以下代码取不到价格，已跳过：\n" + ", ".join(missing))
        rows = [(s, n) for s, n in rows if prices.get(s)]
        if not rows:
            QMessageBox.warning(self, "错误", "全部标的都取价失败，无法建仓"); return

        budget = float(self.budget_spin.value())
        mode = self.mode_cb.currentText()
        n = len(rows)
        plan_id = PaperWatchlistStore.create_plan(name, note=f"一键建仓 · {mode} · 预算${budget:,.0f}")
        today = date.today().isoformat()
        if mode == "等金额买入":
            per = budget / n
            for sym, nm in rows:
                p = prices[sym]
                qty = round(per / p, 4)
                PaperWatchlistStore.add_position(plan_id, sym, nm, qty, p, today)
        else:  # 等股数买入：先按等金额估算一个基准股数，再统一套用
            base_price = np.mean([prices[s] for s, _ in rows])
            base_qty = max(1, round((budget / n) / base_price))
            for sym, nm in rows:
                PaperWatchlistStore.add_position(plan_id, sym, nm, base_qty, prices[sym], today)

        self.result_plan_id = plan_id
        self.accept()


# ══════════════════════════════════════════════════════════════════
# 「编辑持仓」子弹窗：建仓后改数量/买入价/买入日期用
# ══════════════════════════════════════════════════════════════════
class EditPositionDialog(QDialog):
    def __init__(self, env: dict, pos: dict, parent=None):
        super().__init__(parent)
        self.env = env; self.pos = pos
        T = env["T"]
        self.setWindowTitle(f"编辑持仓 — {pos['symbol']}")
        try: self.setWindowIcon(env["_make_app_icon"]())
        except Exception: pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))

        v = QVBoxLayout(self)
        hdr = QLabel(f"<b>{pos['symbol']}</b>  {pos.get('name') or ''}")
        hdr.setStyleSheet(f"color:{T.TEXT_H};font-size:11pt;")
        v.addWidget(hdr)

        form = QHBoxLayout()
        form.addWidget(QLabel("数量:"))
        self.qty_spin = QDoubleSpinBox(); self.qty_spin.setRange(0.0001, 10_000_000)
        self.qty_spin.setDecimals(4); self.qty_spin.setValue(float(pos["qty"]))
        form.addWidget(self.qty_spin)
        form.addWidget(QLabel("买入价:"))
        self.price_spin = QDoubleSpinBox(); self.price_spin.setRange(0.0001, 1_000_000)
        self.price_spin.setDecimals(2); self.price_spin.setValue(float(pos["buy_price"]))
        form.addWidget(self.price_spin)
        fill_btn = QPushButton("取当前价"); fill_btn.clicked.connect(self._fill_current_price)
        form.addWidget(fill_btn)
        v.addLayout(form)

        date_row = QHBoxLayout()
        date_row.addWidget(QLabel("买入日期:"))
        self.date_edit = QDateEdit(QDate.fromString(pos["buy_date"], "yyyy-MM-dd"))
        self.date_edit.setCalendarPopup(True)
        date_row.addWidget(self.date_edit); date_row.addStretch()
        v.addLayout(date_row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("保存"); save_btn.setStyleSheet(f"background:{T.GREEN};color:white;")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(save_btn)
        v.addLayout(btn_row)

    def _fill_current_price(self):
        p = self.env["get_realtime_price"](self.pos["symbol"])
        if p is None:
            try:
                df = self.env["fetch_stock_data"](self.pos["symbol"], "5d")
                if df is not None and not df.empty:
                    p = float(df["Close"].iloc[-1])
            except Exception:
                p = None
        if p is None:
            QMessageBox.warning(self, "取价失败", f"{self.pos['symbol']} 取不到当前价，请手动输入"); return
        self.price_spin.setValue(round(p, 2))

    def _save(self):
        qty = float(self.qty_spin.value())
        price = float(self.price_spin.value())
        if qty <= 0 or price <= 0:
            QMessageBox.information(self, "提示", "数量和买入价都需要大于0"); return
        buy_date = self.date_edit.date().toString("yyyy-MM-dd")
        PaperWatchlistStore.update_position(self.pos["id"], qty, price, buy_date)
        self.accept()


# ══════════════════════════════════════════════════════════════════
# AI分析详情弹窗：单只标的的概率柱状图 + 特征重要性 + 历史命中率
# ══════════════════════════════════════════════════════════════════
class AIDetailDialog(QDialog):
    def __init__(self, env: dict, result: dict, parent=None):
        super().__init__(parent)
        self.env = env; self.result = result
        T = env["T"]
        sym = result["symbol"]
        self.setWindowTitle(f"AI趋势分析详情 — {sym}")
        try: self.setWindowIcon(env["_make_app_icon"]())
        except Exception: pass
        try:
            _f = self.windowFlags()
            _f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
            _f &= ~Qt.WindowContextHelpButtonHint
            self.setWindowFlags(_f)
        except Exception:
            pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self.resize(760, 640)

        v = QVBoxLayout(self)
        cur = result.get("currency", "$")
        hdr = QLabel(
            f"<b>{sym}</b> {result.get('name') or ''}　现价 <b>{cur}{result['cur_price']:.2f}</b>　"
            f"市场状态: {result['regime']}　综合信号: "
            f"<b style='color:{getattr(T, result['signal_color'])}'>{result['signal_text']}</b>")
        hdr.setStyleSheet(f"color:{T.TEXT_H};font-size:10pt;"); hdr.setWordWrap(True)
        v.addWidget(hdr)

        advice = QLabel(f"持仓解读：{result['advice']}　（仅供参考，不构成投资建议）")
        advice.setStyleSheet(f"color:{T.GOLD};font-size:9pt;"); advice.setWordWrap(True)
        v.addWidget(advice)

        mc = result.get("mc")
        if mc:
            mc_lbl = QLabel(
                f"20日蒙特卡洛价格带：P5 {mc['low_pct']:+.1f}%　"
                f"中位数 {mc['median_pct']:+.1f}%　P95 {mc['high_pct']:+.1f}%　"
                f"上涨概率 {mc['up_prob']*100:.0f}%")
            mc_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
            v.addWidget(mc_lbl)

        acc = result.get("acc_report") or {}
        if acc:
            parts = []
            for h in (5, 20, 60):
                a = acc.get(h)
                if a: parts.append(f"{h}日:{a['accuracy']:.0f}%命中({a['n']}次)")
            if parts:
                acc_lbl = QLabel("历史模型命中率（越多样本越可信）： " + "　".join(parts))
                acc_lbl.setStyleSheet(f"color:{T.CYAN};font-size:8pt;")
                v.addWidget(acc_lbl)
        else:
            note = QLabel("历史命中率：暂无已到期的历史预测记录（多跑几次、隔几天回来看会逐步积累）")
            note.setStyleSheet(f"color:{T.TEXT_3};font-size:8pt;")
            v.addWidget(note)

        self.canvas = FigureCanvas(Figure(figsize=(7, 5), facecolor=T.MPL_BG))
        v.addWidget(self.canvas, 1)
        self._draw()

        close_btn = QPushButton("关闭"); close_btn.clicked.connect(self.accept)
        row = QHBoxLayout(); row.addStretch(); row.addWidget(close_btn)
        v.addLayout(row)

    def _draw(self):
        env = self.env; T = env["T"]; result = self.result
        get_rc = env["_get_mpl_rc"]; apply_font = env["_apply_font_to_figure"]
        mpl_style = env.get("_mpl_style")
        fig = self.canvas.figure; fig.clear()

        with mpl.rc_context(get_rc()):
            ax1 = fig.add_subplot(211)
            hs = [5, 20, 60]
            vals = [result["probs"][h] * 100 for h in hs]
            colors = [T.GREEN if v >= 50 else T.RED for v in vals]
            ax1.bar([f"{h}日" for h in hs], vals, color=colors, alpha=0.85)
            ax1.axhline(50, color=T.TEXT_3, lw=0.8, ls="--")
            for i, v_ in enumerate(vals):
                ax1.text(i, v_ + 1, f"{v_:.0f}%", ha="center", color=T.TEXT_1, fontsize=8)
            if mpl_style: mpl_style(ax1, title="上涨概率（AI集成模型：XGB+历史分位+动量）", ylabel="概率%")
            ax1.set_ylim(0, 100)

            ax2 = fig.add_subplot(212)
            feat_imp = result.get("feat_imp") or []
            if feat_imp:
                names = [f for f, _ in feat_imp][::-1]
                imps = [v_ for _, v_ in feat_imp][::-1]
                ax2.barh(names, imps, color=T.PURPLE if hasattr(T, "PURPLE") else T.GOLD, alpha=0.85)
                if mpl_style: mpl_style(ax2, title="20日预测 · 特征重要性Top5（模型主要看这些）", xlabel="重要性")
            else:
                ax2.text(0.5, 0.5, "特征重要性数据不足", ha="center", va="center",
                         color=T.TEXT_2, transform=ax2.transAxes)
                if mpl_style: mpl_style(ax2)
            fig.tight_layout(pad=1.5)
        apply_font(fig)
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════════
# 组合诊断报告：把已有数据（持仓/AI概率/收益走势/大盘对比）拼成一段
# 人话版的组合分析——效果对标 tradure.com 那种"AI组合摘要"，但离线也能
# 用（纯模板拼接，不依赖任何外部服务）；配置好OpenAI兼容的大模型API后，
# 可以额外让大模型把这些数据润色成更自然的点评（可选，不影响离线部分）。
# ══════════════════════════════════════════════════════════════════
def _build_diagnosis_report(plan_name: str, agg: Dict[str, Dict], ai_results: Dict[str, Dict],
                            snaps: List[Dict], bench_pct: Optional[float],
                            colors: Dict[str, str]) -> str:
    """纯函数，返回HTML字符串（供QTextEdit.setHtml直接渲染），不依赖Qt对象。"""
    g, r, gold, t2 = colors["green"], colors["red"], colors["gold"], colors["text2"]

    def pct_span(v: float) -> str:
        c = g if v >= 0 else r
        return f"<span style='color:{c}'>{v:+.2f}%</span>"

    total_cost = total_value = 0.0
    rows = []
    for sym, a in agg.items():
        qty, avg_cost = a["qty"], a["avg_cost"]
        cost = qty * avg_cost
        ar = ai_results.get(sym)
        cur_price = ar["cur_price"] if ar else avg_cost  # 没跑过AI分析就按成本估，避免总值失真
        value = qty * cur_price
        total_cost += cost; total_value += value
        rows.append({"sym": sym, "name": a.get("name") or "", "qty": qty, "cost": cost,
                     "value": value, "ai": ar})

    total_pnl_pct = ((total_value - total_cost) / total_cost * 100.0) if total_cost else 0.0

    parts = [f"<h3 style='margin:2px 0;'>《{plan_name}》组合诊断</h3>"]

    # ① 总览
    parts.append(
        f"<p><b>总览：</b>持仓{len(rows)}只，总成本 ${total_cost:,.2f}，总现值 ${total_value:,.2f}，"
        f"浮动盈亏 {pct_span(total_pnl_pct)}。</p>")

    # ② 对比大盘
    if len(snaps) >= 2 and bench_pct is not None:
        plan_pct = snaps[-1]["pnl_pct"] - snaps[0]["pnl_pct"] + snaps[0]["pnl_pct"]  # 用最新快照收益
        plan_pct = snaps[-1]["pnl_pct"]
        diff = plan_pct - bench_pct
        verdict = "跑赢" if diff >= 0 else "跑输"
        parts.append(
            f"<p><b>对比大盘：</b>自有快照记录以来，本方案累计收益 {pct_span(plan_pct)}，"
            f"同期^GSPC(标普500) {pct_span(bench_pct)}，"
            f"<b style='color:{g if diff>=0 else r}'>{verdict}大盘 {abs(diff):.2f} 个百分点</b>。</p>")
    elif len(snaps) < 2:
        parts.append(f"<p><b>对比大盘：</b>快照不足2条，暂时算不出跟大盘的对比（多点几次「刷新价格/记录快照」积累）。</p>")

    # ③ 持仓集中度
    if total_value > 0 and rows:
        top = max(rows, key=lambda x: x["value"])
        top_pct = top["value"] / total_value * 100
        conc_note = "，集中度偏高，单一标的波动会明显放大组合波动" if top_pct >= 40 else ""
        parts.append(
            f"<p><b>持仓集中度：</b>最大仓位是 <b>{top['sym']}</b>"
            f"（{top.get('name') or ''}），占组合现值 <b>{top_pct:.1f}%</b>{conc_note}。</p>")

    # ④ 逐仓AI观点（按信号强度排序，最有信息量的排前面）
    with_ai = [x for x in rows if x["ai"]]
    without_ai = [x for x in rows if not x["ai"]]
    if with_ai:
        with_ai.sort(key=lambda x: abs(x["ai"]["composite"] - 0.5), reverse=True)
        parts.append("<p><b>逐仓AI历史数据预测：</b></p><ul style='margin-top:2px;'>")
        for x in with_ai:
            ar = x["ai"]
            sc = g if ar["signal_color"] == "GREEN" else (r if ar["signal_color"] == "RED" else t2)
            p20 = ar["probs"][20] * 100
            parts.append(
                f"<li><b>{x['sym']}</b> {x.get('name') or ''}：20日上涨概率 {p20:.0f}%，"
                f"综合信号 <span style='color:{sc}'>{ar['signal_text']}</span>，{ar['advice']}</li>")
        parts.append("</ul>")
    if without_ai:
        syms = "、".join(x["sym"] for x in without_ai)
        parts.append(f"<p style='color:{t2};font-size:8pt;'>{syms} 还没跑过AI趋势分析，"
                     f"先去「AI趋势分析」Tab跑一遍再来看诊断会更完整。</p>")

    parts.append(
        f"<p style='color:{t2};font-size:8pt;margin-top:8px;'>"
        "⚠ 以上基于历史数据统计与概率模型生成，不构成投资建议，模型不保证未来表现。</p>")
    return "".join(parts)


def _diagnosis_report_plaintext(html_free_ai_results: Dict[str, Dict], plan_name: str,
                                agg: Dict[str, Dict], total_cost: float, total_value: float,
                                bench_pct: Optional[float]) -> str:
    """给大模型看的纯文本版数据摘要（不含HTML标签，比人类报告更紧凑）。"""
    lines = [f"方案名称：{plan_name}", f"总成本：${total_cost:,.2f}", f"总现值：${total_value:,.2f}"]
    if total_cost:
        lines.append(f"总浮动盈亏：{(total_value-total_cost)/total_cost*100:+.2f}%")
    if bench_pct is not None:
        lines.append(f"同期标普500基准涨跌：{bench_pct:+.2f}%")
    lines.append("持仓明细：")
    for sym, a in agg.items():
        ar = html_free_ai_results.get(sym)
        if ar:
            lines.append(
                f"- {sym}({a.get('name') or ''}): 数量{a['qty']:g} 成本${a['avg_cost']:.2f} "
                f"现价${ar['cur_price']:.2f} 5日↑概率{ar['probs'][5]*100:.0f}% "
                f"20日↑概率{ar['probs'][20]*100:.0f}% 60日↑概率{ar['probs'][60]*100:.0f}% "
                f"市场状态{ar['regime']} 综合信号{ar['signal_text']}")
        else:
            lines.append(f"- {sym}({a.get('name') or ''}): 数量{a['qty']:g} 成本${a['avg_cost']:.2f}（未跑AI分析）")
    return "\n".join(lines)


# 单只标的（历史走势Tab用）：给大模型的系统提示——跟组合级别的不一样，
# 这里是"要不要现在建仓"的决策辅助场景，不是已有持仓的组合诊断。
SINGLE_SYMBOL_SYSTEM_PROMPT = (
    "你是一名谨慎的证券分析助理。用户会给你一只股票的历史价格区间统计和"
    "AI概率模型（历史数据训练）的输出，用户正在考虑是否要模拟建仓这只票。"
    "请用中文写一段150-300字的分析：概括近期走势特征、AI模型怎么看后续"
    "5/20/60日的方向、如果现在建仓有什么需要留意的风险点。不要给出具体"
    "买卖点位、仓位大小或收益承诺，结尾提醒这是基于历史数据的概率参考，"
    "不构成投资建议。"
)


def _single_symbol_context_text(sym: str, name: str, period_label: str,
                                period_start_price: float, cur_price: float,
                                period_high: float, period_low: float,
                                ai_result: Optional[Dict]) -> str:
    """纯函数：把历史区间统计 + （若已跑过）AI概率结果，拼成给大模型看的文本。"""
    chg_pct = ((cur_price - period_start_price) / period_start_price * 100.0) if period_start_price else 0.0
    lines = [
        f"标的：{sym}（{name}）",
        f"统计区间：{period_label}",
        f"区间起始价：${period_start_price:.2f}　现价：${cur_price:.2f}　区间涨跌：{chg_pct:+.2f}%",
        f"区间最高：${period_high:.2f}　区间最低：${period_low:.2f}",
    ]
    if ai_result:
        p5 = ai_result["probs"][5] * 100; p20 = ai_result["probs"][20] * 100; p60 = ai_result["probs"][60] * 100
        lines.append(f"AI市场状态识别：{ai_result['regime']}")
        lines.append(f"AI上涨概率：5日{p5:.0f}%　20日{p20:.0f}%　60日{p60:.0f}%")
        lines.append(f"AI综合信号：{ai_result['signal_text']}")
        if ai_result.get("mc"):
            mc = ai_result["mc"]
            lines.append(f"20日蒙特卡洛价格带：P5 {mc['low_pct']:+.1f}% ~ P95 {mc['high_pct']:+.1f}%"
                         f"（中位数{mc['median_pct']:+.1f}%）")
    else:
        lines.append("（尚未跑AI趋势预测，只有历史价格统计，没有概率模型输出）")
    return "\n".join(lines)


class LLMNarrativeThread(QThread):
    """调用OpenAI兼容的 chat/completions 接口，把统计数据润色成一段自然语言点评。可选功能。"""
    result_ready = pyqtSignal(str)
    error_msg = pyqtSignal(str)

    _SYSTEM_PROMPT = (
        "你是一名谨慎的投资组合分析助理。用户会给你一份模拟持仓的统计数据摘要"
        "（成本、现值、AI模型给出的历史数据预测概率等）。请用中文写一段"
        "200-400字的组合点评：解读现状、指出风险点（例如集中度、近期弱势标的），"
        "语气克制、专业，不要给出具体买卖指令，不要承诺收益，结尾提醒这只是"
        "基于历史数据的概率分析，不构成投资建议。"
    )

    def __init__(self, base_url: str, api_key: str, model: str, context_text: str,
                system_prompt: Optional[str] = None):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.context_text = context_text
        self.system_prompt = system_prompt or self._SYSTEM_PROMPT

    def run(self):
        try:
            import requests
        except ImportError:
            self.error_msg.emit("未安装 requests 库（pip install requests）"); return
        if not self.base_url or not self.api_key:
            self.error_msg.emit("尚未配置API地址/密钥"); return
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.context_text},
                    ],
                    "temperature": 0.4,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            if not text:
                self.error_msg.emit("大模型返回了空内容"); return
            self.result_ready.emit(text)
        except Exception as e:
            self.error_msg.emit(str(e))


class LLMSettingsDialog(QDialog):
    """配置OpenAI兼容的大模型API（DeepSeek/Kimi/通义千问/OpenAI等都支持这个协议）。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("配置AI大模型（可选）")
        self.resize(480, 260)
        v = QVBoxLayout(self)
        tip = QLabel(
            "填任意 OpenAI 兼容接口的地址/密钥/模型名即可，比如 DeepSeek、Kimi、通义千问、"
            "OpenAI 官方等。API Key 只存在本地数据库文件里，不会上传到任何地方。\n"
            "不填也没关系——「生成诊断报告」离线模板本来就能用，这个只是让点评读起来更像人写的。")
        tip.setWordWrap(True); v.addWidget(tip)

        self.url_edit = QLineEdit(PaperWatchlistStore.get_config("llm_base_url"))
        self.url_edit.setPlaceholderText("例如 https://api.deepseek.com/v1")
        self.key_edit = QLineEdit(PaperWatchlistStore.get_config("llm_api_key"))
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.model_edit = QLineEdit(PaperWatchlistStore.get_config("llm_model"))
        self.model_edit.setPlaceholderText("例如 deepseek-chat")

        for label, w in [("接口地址(base_url):", self.url_edit),
                         ("API Key:", self.key_edit),
                         ("模型名:", self.model_edit)]:
            row = QHBoxLayout(); lbl = QLabel(label); lbl.setFixedWidth(120)
            row.addWidget(lbl); row.addWidget(w); v.addLayout(row)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("保存"); save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("取消"); cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch(); btn_row.addWidget(cancel_btn); btn_row.addWidget(save_btn)
        v.addLayout(btn_row)

    def _save(self):
        PaperWatchlistStore.set_config("llm_base_url", self.url_edit.text().strip())
        PaperWatchlistStore.set_config("llm_api_key", self.key_edit.text().strip())
        PaperWatchlistStore.set_config("llm_model", self.model_edit.text().strip())
        self.accept()


# ══════════════════════════════════════════════════════════════════
# 主弹窗：方案列表 + 持仓明细 + 收益走势
# ══════════════════════════════════════════════════════════════════
class PaperWatchlistDialog(QDialog):
    def __init__(self, env: dict, parent=None):
        super().__init__(parent)
        self.env = env
        self.T = T = env["T"]
        self.setWindowTitle("模拟自选 — 建仓模拟器")
        try: self.setWindowIcon(env["_make_app_icon"]())
        except Exception: pass
        # 补最小化/最大化按钮，去掉无用的帮助(?)按钮 —— QDialog默认没有最大化按钮，
        # 跟主程序其它子窗口（配对对比/大盘仪表盘）统一体验，窗口可以放大铺满屏幕。
        try:
            _f = self.windowFlags()
            _f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
            _f &= ~Qt.WindowContextHelpButtonHint
            self.setWindowFlags(_f)
        except Exception:
            pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self.resize(1180, 720)

        PaperWatchlistStore.init_db()
        self.cur_plan_id: Optional[int] = None
        self._plan_pnl_cache: Dict[int, Optional[float]] = {}

        root = QHBoxLayout(self)

        # ── 左：方案列表 ─────────────────────────────────────
        left = QVBoxLayout()
        left.addWidget(QLabel("<b>方案列表</b>（可以建好几套互相对比）"))
        self.plan_list = QListWidget(); self.plan_list.setFixedWidth(260)
        self.plan_list.currentItemChanged.connect(self._on_plan_changed)
        left.addWidget(self.plan_list, 1)
        plan_btn_row = QHBoxLayout()
        new_btn = QPushButton("新建空方案"); new_btn.clicked.connect(self._new_plan)
        quick_btn = QPushButton("一键建仓")
        quick_btn.setStyleSheet(f"color:{T.GOLD};border-color:{T.GOLD};")
        quick_btn.clicked.connect(self._quick_build)
        del_btn = QPushButton("删除方案"); del_btn.setStyleSheet(f"color:{T.RED};")
        del_btn.clicked.connect(self._delete_plan)
        plan_btn_row.addWidget(new_btn); plan_btn_row.addWidget(quick_btn)
        left.addLayout(plan_btn_row); left.addWidget(del_btn)
        self.plan_note_lbl = QLabel(); self.plan_note_lbl.setWordWrap(True)
        self.plan_note_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        left.addWidget(self.plan_note_lbl)
        root.addLayout(left)

        # ── 右：Tab（持仓明细 / 收益走势）───────────────────
        right = QVBoxLayout()
        self.summary_lbl = QLabel("请选择或新建一套方案")
        self.summary_lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:11pt;padding:4px;")
        right.addWidget(self.summary_lbl)

        self.tabs = QTabWidget()
        right.addWidget(self.tabs, 1)

        # Tab1 持仓明细
        hold_w = QWidget(); hold_l = QVBoxLayout(hold_w)
        add_row = QHBoxLayout()
        self.sym_cb = QComboBox(); self.sym_cb.setEditable(True)
        self.sym_cb.addItems([f"{s} {n}" for s, n in PRESET_STOCKS])
        self.sym_cb.setFixedWidth(180)
        self.qty_spin = QDoubleSpinBox(); self.qty_spin.setRange(0.0001, 1_000_000); self.qty_spin.setValue(10)
        self.qty_spin.setDecimals(4)
        self.price_spin = QDoubleSpinBox(); self.price_spin.setRange(0.0001, 1_000_000); self.price_spin.setDecimals(2)
        fill_btn = QPushButton("取当前价"); fill_btn.clicked.connect(self._fill_current_price)
        self.date_edit = QDateEdit(QDate.currentDate()); self.date_edit.setCalendarPopup(True)
        add_btn = QPushButton("添加持仓"); add_btn.setStyleSheet(f"background:{T.GREEN};color:white;")
        add_btn.clicked.connect(self._add_position)
        add_row.addWidget(QLabel("代码:")); add_row.addWidget(self.sym_cb)
        add_row.addWidget(QLabel("数量:")); add_row.addWidget(self.qty_spin)
        add_row.addWidget(QLabel("买入价:")); add_row.addWidget(self.price_spin)
        add_row.addWidget(fill_btn)
        add_row.addWidget(QLabel("买入日期:")); add_row.addWidget(self.date_edit)
        add_row.addWidget(add_btn)
        hold_l.addLayout(add_row)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("排序:"))
        self.sort_combo = QComboBox(); self.sort_combo.addItems(SORT_MODES)
        self.sort_combo.currentIndexChanged.connect(lambda _i: self._reload_positions())
        sort_row.addWidget(self.sort_combo)
        sort_hint = QLabel("默认按涨幅%从高到低——今天谁涨得最好、谁拖后腿一眼看到；刷新价格后自动按当前排序方式重排。")
        sort_hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;"); sort_hint.setWordWrap(True)
        sort_row.addWidget(sort_hint, 1)
        hold_l.addLayout(sort_row)

        self.pos_table = QTableWidget(0, 12)
        self.pos_table.setHorizontalHeaderLabels(
            ["排名", "代码", "名称", "数量", "买入价", "买入日期", "现价", "仓位占比%",
             "现值", "浮动盈亏", "盈亏%", "持有天数"])
        self.pos_table.horizontalHeader().setStretchLastSection(True)
        self.pos_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.pos_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 单元格本身不可直接改，双击弹编辑框
        self.pos_table.doubleClicked.connect(lambda _idx: self._edit_position())
        hold_l.addWidget(self.pos_table, 1)
        hint = QLabel("双击某一行，或选中后点「编辑数量/价格」— 建仓后填错了可以改。")
        hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        hold_l.addWidget(hint)

        bottom_row = QHBoxLayout()
        edit_pos_btn = QPushButton("编辑数量/价格"); edit_pos_btn.setStyleSheet(f"color:{T.GOLD};border-color:{T.GOLD};")
        edit_pos_btn.clicked.connect(self._edit_position)
        del_pos_btn = QPushButton("删除选中持仓"); del_pos_btn.setStyleSheet(f"color:{T.RED};")
        del_pos_btn.clicked.connect(self._delete_position)
        refresh_btn = QPushButton("刷新价格 / 记录快照")
        refresh_btn.setStyleSheet(f"background:{T.ACCENT};color:white;")
        refresh_btn.clicked.connect(self._refresh_prices)
        bottom_row.addWidget(edit_pos_btn); bottom_row.addWidget(del_pos_btn)
        bottom_row.addStretch(); bottom_row.addWidget(refresh_btn)
        hold_l.addLayout(bottom_row)
        self.tabs.addTab(hold_w, "持仓明细")

        # Tab1.5 历史走势预测：建仓前先看历史K线（跟主界面一样的多周期切换），
        # 一键取当前价直接建仓，或跑AI预测+大模型点评辅助决策
        hist_w = QWidget(); hist_l = QVBoxLayout(hist_w)
        hist_hint = QLabel("挑个代码，切换周期看历史走势（跟主界面K线Tab一个逻辑），"
                           "看完可以直接「用此价格/代码建仓」，或跑「AI趋势预测」结合大模型给一段点评再决定。")
        hist_hint.setWordWrap(True); hist_hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        hist_l.addWidget(hist_hint)

        hist_top_row = QHBoxLayout()
        self.hist_sym_cb = QComboBox(); self.hist_sym_cb.setEditable(True)
        self.hist_sym_cb.addItems([f"{s} {n}" for s, n in PRESET_STOCKS])
        self.hist_sym_cb.setFixedWidth(180)
        hist_load_btn = QPushButton("加载"); hist_load_btn.clicked.connect(self._refresh_history)
        hist_top_row.addWidget(QLabel("代码:")); hist_top_row.addWidget(self.hist_sym_cb)
        hist_top_row.addWidget(hist_load_btn); hist_top_row.addStretch()
        hist_l.addLayout(hist_top_row)

        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("周期:"))
        self.hist_period_group = QButtonGroup(self); self.hist_period_group.setExclusive(True)
        self._hist_period_label = "1年"
        for label, _days in HIST_PERIODS:
            btn = QPushButton(label); btn.setCheckable(True)
            btn.setChecked(label == self._hist_period_label)
            btn.clicked.connect(lambda _checked, lbl=label: self._on_hist_period_clicked(lbl))
            self.hist_period_group.addButton(btn)
            period_row.addWidget(btn)
        period_row.addStretch()
        hist_l.addLayout(period_row)

        self.hist_info_lbl = QLabel("选个代码点「加载」开始")
        self.hist_info_lbl.setStyleSheet(f"color:{T.TEXT_H};font-size:9pt;")
        hist_l.addWidget(self.hist_info_lbl)

        self.hist_canvas = FigureCanvas(Figure(figsize=(9, 4.0), facecolor=T.MPL_BG))
        hist_l.addWidget(self.hist_canvas, 1)

        hist_btn_row = QHBoxLayout()
        use_price_btn = QPushButton("用此价格/代码建仓")
        use_price_btn.setStyleSheet(f"background:{T.GREEN};color:white;")
        use_price_btn.clicked.connect(self._use_hist_for_position)
        hist_ai_btn = QPushButton("AI趋势预测（含大模型点评）")
        hist_ai_btn.setStyleSheet(f"color:{T.GOLD};border-color:{T.GOLD};")
        hist_ai_btn.clicked.connect(self._run_hist_ai_predict)
        self.hist_ai_btn = hist_ai_btn
        hist_btn_row.addWidget(use_price_btn); hist_btn_row.addWidget(hist_ai_btn); hist_btn_row.addStretch()
        hist_l.addLayout(hist_btn_row)

        self.hist_status_lbl = QLabel(""); self.hist_status_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        hist_l.addWidget(self.hist_status_lbl)
        self.hist_ai_text = QTextEdit(); self.hist_ai_text.setReadOnly(True)
        self.hist_ai_text.setMaximumHeight(160)
        self.hist_ai_text.setPlaceholderText("点「AI趋势预测」后，概率结果和大模型点评（如已配置）会显示在这里。")
        hist_l.addWidget(self.hist_ai_text)

        self.tabs.addTab(hist_w, "历史走势预测")
        self._hist_df_cache: Dict[str, pd.DataFrame] = {}
        self._hist_cur_sym = ""
        self._hist_ai_thread = None
        self._hist_llm_thread = None
        self._hist_ai_result_cache: Dict[str, Dict] = {}

        # Tab2 收益走势
        chart_w = QWidget(); chart_l = QVBoxLayout(chart_w)
        chart_hint = QLabel("每次点「刷新价格/记录快照」会记一条当日总市值；攒够几天/几个月的快照后这里会画出走势，"
                            "并与同期 ^GSPC(标普500) 归一化对比，看这套方案有没有跑赢大盘。")
        chart_hint.setWordWrap(True); chart_hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        chart_l.addWidget(chart_hint)
        self.chart_canvas = FigureCanvas(Figure(figsize=(9, 4.5), facecolor=T.MPL_BG))
        chart_l.addWidget(self.chart_canvas, 1)
        self.tabs.addTab(chart_w, "收益走势")
        self.tabs.currentChanged.connect(lambda _i: self._refresh_chart())

        # Tab3 AI趋势分析
        ai_w = QWidget(); ai_l = QVBoxLayout(ai_w)
        ai_hint = QLabel(
            "对本方案里的持仓逐个跑一遍AI集成预测（XGBoost + 历史分位 + 动量，"
            "有装v14引擎补丁的话再加 Purged CV / 校准 / 大盘上下文那一套），"
            "给出5/20/60日上涨概率、市场状态、蒙特卡洛价格带，并结合你的买入成本给出解读。\n"
            "⚠ 标的越多跑得越久（每只都要训练模型），耐心等；这是概率预测，不是保证，仅供参考。")
        ai_hint.setWordWrap(True); ai_hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        ai_l.addWidget(ai_hint)

        ai_btn_row = QHBoxLayout()
        self.ai_run_btn = QPushButton("运行AI分析（本方案持仓）")
        self.ai_run_btn.setStyleSheet(f"background:{T.PURPLE if hasattr(T,'PURPLE') else T.GOLD};color:white;")
        self.ai_run_btn.clicked.connect(self._run_ai_analysis)
        self.ai_status_lbl = QLabel("")
        self.ai_status_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        ai_btn_row.addWidget(self.ai_run_btn); ai_btn_row.addWidget(self.ai_status_lbl); ai_btn_row.addStretch()
        ai_l.addLayout(ai_btn_row)

        self.ai_progress = QProgressBar(); self.ai_progress.setVisible(False)
        ai_l.addWidget(self.ai_progress)

        self.ai_table = QTableWidget(0, 10)
        self.ai_table.setHorizontalHeaderLabels(
            ["代码", "名称", "现价", "持仓盈亏%", "市场状态", "5日↑概率", "20日↑概率",
             "60日↑概率", "综合信号", "持仓解读"])
        self.ai_table.horizontalHeader().setStretchLastSection(True)
        self.ai_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ai_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ai_table.doubleClicked.connect(lambda _idx: self._open_ai_detail())
        ai_l.addWidget(self.ai_table, 1)
        ai_tip = QLabel("双击某一行看详情（概率柱状图 + 特征重要性 + 历史模型命中率）。")
        ai_tip.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        ai_l.addWidget(ai_tip)
        self.tabs.addTab(ai_w, "AI趋势分析")

        # Tab4 组合诊断（对标 tradure.com 的AI组合摘要，离线模板 + 可选大模型润色）
        diag_w = QWidget(); diag_l = QVBoxLayout(diag_w)
        diag_hint = QLabel(
            "把上面「AI趋势分析」跑出来的结果，加上收益走势/大盘对比、持仓集中度，"
            "拼成一份组合诊断报告——效果类似 tradure.com 的AI组合摘要。"
            "离线模板马上能用；如果配置了OpenAI兼容的大模型API，还能让大模型把这些数据润色成更自然的点评（可选）。")
        diag_hint.setWordWrap(True); diag_hint.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        diag_l.addWidget(diag_hint)

        diag_btn_row = QHBoxLayout()
        gen_btn = QPushButton("生成诊断报告"); gen_btn.setStyleSheet(f"background:{T.ACCENT};color:white;")
        gen_btn.clicked.connect(self._generate_diagnosis)
        llm_btn = QPushButton("AI大模型润色（可选）")
        llm_btn.setStyleSheet(f"color:{T.GOLD};border-color:{T.GOLD};")
        llm_btn.clicked.connect(self._run_llm_narrative)
        cfg_btn = QPushButton("配置AI大模型"); cfg_btn.clicked.connect(self._open_llm_settings)
        copy_btn = QPushButton("复制报告"); copy_btn.clicked.connect(self._copy_diagnosis)
        diag_btn_row.addWidget(gen_btn); diag_btn_row.addWidget(llm_btn)
        diag_btn_row.addWidget(cfg_btn); diag_btn_row.addWidget(copy_btn); diag_btn_row.addStretch()
        diag_l.addLayout(diag_btn_row)

        self.diag_status_lbl = QLabel(""); self.diag_status_lbl.setStyleSheet(f"color:{T.TEXT_2};font-size:8pt;")
        diag_l.addWidget(self.diag_status_lbl)

        self.diag_text = QTextEdit(); self.diag_text.setReadOnly(True)
        self.diag_text.setPlaceholderText("点「生成诊断报告」开始（建议先跑一遍「AI趋势分析」，诊断会更完整）")
        diag_l.addWidget(self.diag_text, 1)
        self.tabs.addTab(diag_w, "组合诊断")
        self._llm_thread = None
        self._diag_context_cache = None

        self._ai_thread = None
        self._ai_results: Dict[str, Dict] = {}

        root.addLayout(right, 1)

        self._reload_plans()

    # ── 方案管理 ─────────────────────────────────────────────
    def _compute_plan_pnl_pct(self, plan_id: int) -> Optional[float]:
        """算一套方案的整体涨跌%，用于左侧方案列表的徽章和默认排序。"""
        positions = PaperWatchlistStore.list_positions(plan_id)
        if not positions:
            return None
        total_cost = total_value = 0.0
        for p in positions:
            cost = p["qty"] * p["buy_price"]; total_cost += cost
            cur_price = self.env["get_realtime_price"](p["symbol"])
            if cur_price is None:
                try:
                    df = self.env["fetch_stock_data"](p["symbol"], "5d")
                    cur_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else p["buy_price"]
                except Exception:
                    cur_price = p["buy_price"]
            total_value += p["qty"] * cur_price
        return ((total_value - total_cost) / total_cost * 100.0) if total_cost else None

    def _reload_plans(self, select_id: Optional[int] = None):
        T = self.T
        plans = PaperWatchlistStore.list_plans()
        for p in plans:
            if p["id"] not in self._plan_pnl_cache:
                self._plan_pnl_cache[p["id"]] = self._compute_plan_pnl_pct(p["id"])
        # 默认按方案涨幅%降序；没有持仓(None)的排最后，不参与涨跌排序
        plans.sort(key=lambda p: (self._plan_pnl_cache.get(p["id"]) is None,
                                  -(self._plan_pnl_cache.get(p["id"]) or 0)))

        self.plan_list.blockSignals(True)
        self.plan_list.clear()
        target_row = 0
        for i, p in enumerate(plans):
            pnl = self._plan_pnl_cache.get(p["id"])
            if pnl is None:
                text = f"{p['name']}   ({p['created_date']})"
            else:
                icon = "🟢" if pnl >= 0 else "🔴"
                text = f"{icon} {p['name']}  {pnl:+.2f}%   ({p['created_date']})"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, p)
            if pnl is not None:
                it.setForeground(QColor(T.GREEN if pnl >= 0 else T.RED))
            self.plan_list.addItem(it)
            if select_id is not None and p["id"] == select_id:
                target_row = i
        self.plan_list.blockSignals(False)
        if plans:
            self.plan_list.setCurrentRow(target_row)
        else:
            self.cur_plan_id = None
            self.pos_table.setRowCount(0)
            self.summary_lbl.setText("还没有方案 — 点左边「新建空方案」或「一键建仓」")
            self.plan_note_lbl.setText("")

    def _update_plan_pnl_badge(self):
        """持仓有变动/刷新价格后调用：重算当前方案的涨幅%，左侧列表徽章+排序跟着刷新。"""
        if self.cur_plan_id is None: return
        self._plan_pnl_cache[self.cur_plan_id] = self._compute_plan_pnl_pct(self.cur_plan_id)
        self._reload_plans(select_id=self.cur_plan_id)

    def _on_plan_changed(self, cur, _prev):
        if cur is None:
            self.cur_plan_id = None; return
        p = cur.data(Qt.UserRole)
        same_plan = (self.cur_plan_id == p["id"])  # 只是徽章刷新触发的"重选同一方案"，别清掉AI/诊断结果
        self.cur_plan_id = p["id"]
        self.plan_note_lbl.setText(p.get("note") or "")
        self._reload_positions()
        self._refresh_chart()
        if not same_plan:
            self.ai_table.setRowCount(0)
            self._ai_results = {}
            self.ai_status_lbl.setText("")
            self.diag_text.clear()
            self._diag_context_cache = None
            self.diag_status_lbl.setText("")

    def _new_plan(self):
        name, ok = QInputDialog.getText(self, "新建方案", "方案名称（例如：防御组合 / 科技组合）:")
        if not ok or not name.strip(): return
        pid = PaperWatchlistStore.create_plan(name.strip())
        self._reload_plans(select_id=pid)

    def _quick_build(self):
        dlg = QuickBuildDialog(self.env, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_plan_id:
            self._reload_plans(select_id=dlg.result_plan_id)

    def _delete_plan(self):
        if self.cur_plan_id is None:
            QMessageBox.information(self, "提示", "请先选中一套方案"); return
        if QMessageBox.question(self, "确认", "删除这套方案及其全部持仓/快照？此操作不可撤销。",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self._plan_pnl_cache.pop(self.cur_plan_id, None)
        PaperWatchlistStore.delete_plan(self.cur_plan_id)
        self._reload_plans()

    # ── 持仓管理 ─────────────────────────────────────────────
    def _fill_current_price(self):
        sym = self._current_sym_input()
        if not sym: return
        p = self.env["get_realtime_price"](sym)
        if p is None:
            try:
                df = self.env["fetch_stock_data"](sym, "5d")
                if df is not None and not df.empty:
                    p = float(df["Close"].iloc[-1])
            except Exception:
                p = None
        if p is None:
            QMessageBox.warning(self, "取价失败", f"{sym} 取不到当前价，请手动输入"); return
        self.price_spin.setValue(round(p, 2))

    def _current_sym_input(self) -> str:
        raw = self.sym_cb.currentText().strip()
        return raw.split()[0].upper() if raw else ""

    def _add_position(self):
        if self.cur_plan_id is None:
            QMessageBox.information(self, "提示", "请先在左边选中或新建一套方案"); return
        sym = self._current_sym_input()
        if not sym:
            QMessageBox.information(self, "提示", "请输入代码"); return
        qty = float(self.qty_spin.value())
        price = float(self.price_spin.value())
        if price <= 0:
            QMessageBox.information(self, "提示", "买入价需要大于0（可点「取当前价」自动填）"); return
        buy_date = self.date_edit.date().toString("yyyy-MM-dd")
        name = ""
        for s, n in PRESET_STOCKS:
            if s == sym: name = n; break
        currency = self.env["_currency"](sym)
        PaperWatchlistStore.add_position(self.cur_plan_id, sym, name, qty, price, buy_date, currency)
        self._reload_positions()
        self._update_plan_pnl_badge()

    def _delete_position(self):
        row = self.pos_table.currentRow()
        if row < 0: return
        pos_id = self.pos_table.item(row, 0).data(Qt.UserRole)
        if pos_id is None: return
        PaperWatchlistStore.delete_position(pos_id)
        self._reload_positions()
        self._update_plan_pnl_badge()

    def _edit_position(self):
        row = self.pos_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "提示", "请先在表格里选中（或双击）要编辑的一行"); return
        pos_id = self.pos_table.item(row, 0).data(Qt.UserRole)
        if pos_id is None or self.cur_plan_id is None: return
        pos = next((p for p in PaperWatchlistStore.list_positions(self.cur_plan_id) if p["id"] == pos_id), None)
        if pos is None: return
        dlg = EditPositionDialog(self.env, pos, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._reload_positions()
            self._update_plan_pnl_badge()

    def _reload_positions(self):
        self.pos_table.setRowCount(0)
        if self.cur_plan_id is None: return
        positions = PaperWatchlistStore.list_positions(self.cur_plan_id)
        T = self.T
        today = date.today()

        rows_data = []
        total_cost = total_value = 0.0
        for p in positions:
            cur_price = self.env["get_realtime_price"](p["symbol"])
            if cur_price is None:
                try:
                    df = self.env["fetch_stock_data"](p["symbol"], "5d")
                    cur_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
                except Exception:
                    cur_price = None
            cost = p["qty"] * p["buy_price"]
            total_cost += cost
            try:
                bd = datetime.strptime(p["buy_date"], "%Y-%m-%d").date()
                held_days = (today - bd).days
            except Exception:
                held_days = 0
            price_ok = cur_price is not None
            if price_ok:
                value = p["qty"] * cur_price
                pnl = value - cost
                pnl_pct = (pnl / cost * 100.0) if cost else 0.0
            else:
                value = cost; pnl = 0.0; pnl_pct = 0.0  # 取价失败按持平处理，避免排序出错；单元格仍显示"取价失败"
            total_value += value
            rows_data.append({
                "id": p["id"], "symbol": p["symbol"], "name": p["name"] or "",
                "qty": p["qty"], "buy_price": p["buy_price"], "buy_date": p["buy_date"],
                "currency": p["currency"] or "$", "cur_price": cur_price, "cost": cost,
                "value": value, "pnl": pnl, "pnl_pct": pnl_pct, "held_days": held_days,
                "price_ok": price_ok,
            })
        for r in rows_data:
            r["pos_pct"] = (r["value"] / total_value * 100.0) if total_value else 0.0

        mode = self.sort_combo.currentText() if hasattr(self, "sort_combo") else SORT_MODES[0]
        if mode == SORT_MODES[0]:      # 涨幅%（高→低）
            rows_data.sort(key=lambda r: r["pnl_pct"], reverse=True)
        elif mode == SORT_MODES[1]:    # 涨幅%（低→高）
            rows_data.sort(key=lambda r: r["pnl_pct"])
        elif mode == SORT_MODES[2]:    # 仓位占比（大→小）
            rows_data.sort(key=lambda r: r["pos_pct"], reverse=True)
        elif mode == SORT_MODES[3]:    # 盈亏金额（高→低）
            rows_data.sort(key=lambda r: r["pnl"], reverse=True)
        elif mode == SORT_MODES[4]:    # 股票代码（A→Z）
            rows_data.sort(key=lambda r: r["symbol"])

        RANK_ICONS = ["🥇", "🥈", "🥉"]
        for i, r in enumerate(rows_data):
            row = self.pos_table.rowCount(); self.pos_table.insertRow(row)
            cur = r["currency"]
            color = T.GREEN if r["pnl_pct"] >= 0 else T.RED

            rank_item = QTableWidgetItem(RANK_ICONS[i] if i < 3 else str(i + 1))
            rank_item.setData(Qt.UserRole, r["id"])
            if r["price_ok"]: rank_item.setForeground(QColor(color))
            self.pos_table.setItem(row, 0, rank_item)
            self.pos_table.setItem(row, 1, QTableWidgetItem(r["symbol"]))
            self.pos_table.setItem(row, 2, QTableWidgetItem(r["name"]))
            self.pos_table.setItem(row, 3, QTableWidgetItem(f"{r['qty']:g}"))
            self.pos_table.setItem(row, 4, QTableWidgetItem(f"{cur}{r['buy_price']:.2f}"))
            self.pos_table.setItem(row, 5, QTableWidgetItem(r["buy_date"]))

            if r["price_ok"]:
                self.pos_table.setItem(row, 6, QTableWidgetItem(f"{cur}{r['cur_price']:.2f}"))
                self.pos_table.setItem(row, 7, QTableWidgetItem(f"{r['pos_pct']:.1f}%"))
                self.pos_table.setItem(row, 8, QTableWidgetItem(f"{cur}{r['value']:,.2f}"))
                pnl_item = QTableWidgetItem(f"{cur}{r['pnl']:+,.2f}"); pnl_item.setForeground(QColor(color))
                self.pos_table.setItem(row, 9, pnl_item)
                pct_item = QTableWidgetItem(f"{r['pnl_pct']:+.2f}%"); pct_item.setForeground(QColor(color))
                self.pos_table.setItem(row, 10, pct_item)
            else:
                for col in (6, 7, 8, 9, 10):
                    self.pos_table.setItem(row, col, QTableWidgetItem("取价失败"))
            self.pos_table.setItem(row, 11, QTableWidgetItem(str(r["held_days"])))

        self.pos_table.resizeColumnsToContents()
        self.pos_table.horizontalHeader().setStretchLastSection(True)

        if positions:
            total_pnl = total_value - total_cost
            total_pct = (total_pnl / total_cost * 100.0) if total_cost else 0.0
            color = T.GREEN if total_pnl >= 0 else T.RED
            self.summary_lbl.setText(
                f"总成本 <b>${total_cost:,.2f}</b>　"
                f"总现值 <b>${total_value:,.2f}</b>　"
                f"浮动盈亏 <b style='color:{color}'>${total_pnl:+,.2f} ({total_pct:+.2f}%)</b>　"
                f"共{len(positions)}笔持仓")
        else:
            self.summary_lbl.setText("这套方案还没有持仓 — 在上面添加，或用「一键建仓」")
        self._pending_snapshot = (total_cost, total_value) if positions else None

    def _refresh_prices(self):
        if self.cur_plan_id is None:
            QMessageBox.information(self, "提示", "请先选中一套方案"); return
        self._reload_positions()
        if getattr(self, "_pending_snapshot", None):
            total_cost, total_value = self._pending_snapshot
            PaperWatchlistStore.record_snapshot(self.cur_plan_id, total_cost, total_value)
        self._refresh_chart()
        self._update_plan_pnl_badge()

    # ── 收益走势图（含 ^GSPC 同期基准对比）──────────────────
    def _refresh_chart(self):
        fig = self.chart_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        T = self.T
        get_rc = self.env["_get_mpl_rc"]; apply_font = self.env["_apply_font_to_figure"]
        mpl_style = self.env.get("_mpl_style")

        if self.cur_plan_id is None:
            ax.text(0.5, 0.5, "请选择一套方案", ha="center", va="center", color=T.TEXT_2, transform=ax.transAxes)
            self.chart_canvas.draw(); return

        snaps = PaperWatchlistStore.list_snapshots(self.cur_plan_id)
        if len(snaps) < 2:
            ax.text(0.5, 0.5, "快照不足2条，暂时画不出走势\n多点几次「刷新价格/记录快照」（建议隔天/隔周点一次）",
                    ha="center", va="center", color=T.TEXT_2, transform=ax.transAxes, fontsize=10)
            if mpl_style: mpl_style(ax, title="收益走势（快照累积中）")
            apply_font(fig); self.chart_canvas.draw(); return

        with mpl.rc_context(get_rc()):
            dates = pd.to_datetime([s["snap_date"] for s in snaps])
            pnl_pct = [s["pnl_pct"] for s in snaps]
            ax.plot(dates, pnl_pct, marker="o", color=T.GOLD, lw=1.8, label="本方案收益%")
            ax.axhline(0, color=T.TEXT_3, lw=0.8)

            # 同期基准：^GSPC 归一化到同一起点，换算成百分比涨跌方便叠加对比
            try:
                fetch = self.env["fetch_stock_data"]
                bench = fetch("^GSPC", "2y")
                if bench is not None and not bench.empty:
                    bser = bench["Close"]
                    bser = bser[bser.index >= dates.min() - pd.Timedelta(days=3)]
                    bser = bser[bser.index <= dates.max() + pd.Timedelta(days=1)]
                    if len(bser) >= 2:
                        base = float(bser.iloc[0])
                        bench_pct = (bser / base - 1.0) * 100.0
                        ax.plot(bser.index, bench_pct.values, color=T.CYAN, lw=1.2,
                               ls="--", label="^GSPC同期基准%")
            except Exception as e:
                logger.warning(f"paper watchlist benchmark: {e}")

            ax.legend(loc="best", fontsize=8, facecolor=T.BG2, labelcolor=T.TEXT_1, edgecolor=T.BORDER)
            if mpl_style:
                mpl_style(ax, title="模拟方案累计收益% vs 大盘基准", ylabel="累计收益%")
            fig.autofmt_xdate()
        apply_font(fig)
        self.chart_canvas.draw()

    # ── AI趋势分析 ───────────────────────────────────────────
    def _run_ai_analysis(self):
        if self.cur_plan_id is None:
            QMessageBox.information(self, "提示", "请先选中一套方案"); return
        if self._ai_thread is not None and self._ai_thread.isRunning():
            QMessageBox.information(self, "提示", "上一轮AI分析还在跑，请稍候"); return

        positions = PaperWatchlistStore.list_positions(self.cur_plan_id)
        if not positions:
            QMessageBox.information(self, "提示", "这套方案还没有持仓，先加几笔再跑AI分析"); return

        missing = [k for k in ("TechnicalIndicators", "ProbabilityEngine") if k not in self.env]
        if missing:
            QMessageBox.warning(self, "缺少依赖",
                                f"AI分析需要主程序提供 {missing}，当前安装环境里没有，无法运行。")
            return

        agg = _aggregate_positions_by_symbol(positions)
        symbols = [(sym, meta) for sym, meta in agg.items()]

        self.ai_table.setRowCount(0)
        self._ai_results = {}
        self.ai_run_btn.setEnabled(False)
        self.ai_progress.setVisible(True); self.ai_progress.setValue(0)
        self.ai_status_lbl.setText(f"正在分析 {len(symbols)} 只标的…")

        th = AIAnalysisThread(symbols, self.env)
        th.progress.connect(self._on_ai_progress)
        th.result_ready.connect(self._on_ai_result)
        th.finished_all.connect(self._on_ai_finished)
        th.error_msg.connect(self._on_ai_error)
        self._ai_thread = th
        th.start()

    def _on_ai_progress(self, pct: int, sym: str):
        self.ai_progress.setValue(pct)
        self.ai_status_lbl.setText(f"分析中… {sym} ({pct}%)")

    def _on_ai_result(self, result: dict):
        self._ai_results[result["symbol"]] = result
        T = self.T
        r = self.ai_table.rowCount(); self.ai_table.insertRow(r)
        cur = result["currency"]
        item0 = QTableWidgetItem(result["symbol"])
        self.ai_table.setItem(r, 0, item0)
        self.ai_table.setItem(r, 1, QTableWidgetItem(result["name"]))
        self.ai_table.setItem(r, 2, QTableWidgetItem(f"{cur}{result['cur_price']:.2f}"))

        pnl_pct = result.get("pnl_pct")
        if pnl_pct is not None:
            pnl_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_item.setForeground(QColor(T.GREEN if pnl_pct >= 0 else T.RED))
        else:
            pnl_item = QTableWidgetItem("—")
        self.ai_table.setItem(r, 3, pnl_item)

        self.ai_table.setItem(r, 4, QTableWidgetItem(result["regime"]))
        for col, h in ((5, 5), (6, 20), (7, 60)):
            p = result["probs"][h] * 100
            it = QTableWidgetItem(f"{p:.0f}%")
            it.setForeground(QColor(T.GREEN if p >= 50 else T.RED))
            self.ai_table.setItem(r, col, it)

        sig_item = QTableWidgetItem(result["signal_text"])
        sig_item.setForeground(QColor(getattr(T, result["signal_color"])))
        self.ai_table.setItem(r, 8, sig_item)
        self.ai_table.setItem(r, 9, QTableWidgetItem(result["advice"]))
        self.ai_table.resizeColumnsToContents()
        self.ai_table.horizontalHeader().setStretchLastSection(True)

    def _on_ai_finished(self, ok: int):
        self.ai_run_btn.setEnabled(True)
        self.ai_progress.setVisible(False)
        self.ai_status_lbl.setText(f"完成，成功分析 {ok} 只（数据不足或取价失败的标的会跳过）")
        self._ai_thread = None

    def _on_ai_error(self, msg: str):
        logger.warning(f"AI analysis error: {msg}")

    def _open_ai_detail(self):
        row = self.ai_table.currentRow()
        if row < 0: return
        sym_item = self.ai_table.item(row, 0)
        if sym_item is None: return
        result = self._ai_results.get(sym_item.text())
        if result is None: return
        AIDetailDialog(self.env, result, parent=self).exec_()

    # ── 组合诊断 ─────────────────────────────────────────────
    def _generate_diagnosis(self):
        if self.cur_plan_id is None:
            QMessageBox.information(self, "提示", "请先选中一套方案"); return
        positions = PaperWatchlistStore.list_positions(self.cur_plan_id)
        if not positions:
            QMessageBox.information(self, "提示", "这套方案还没有持仓"); return
        agg = _aggregate_positions_by_symbol(positions)
        snaps = PaperWatchlistStore.list_snapshots(self.cur_plan_id)

        bench_pct = None
        if len(snaps) >= 2:
            try:
                fetch = self.env["fetch_stock_data"]
                bench = fetch("^GSPC", "2y")
                if bench is not None and not bench.empty:
                    dates = pd.to_datetime([s["snap_date"] for s in snaps])
                    bser = bench["Close"]
                    bser = bser[bser.index >= dates.min() - pd.Timedelta(days=3)]
                    if len(bser) >= 2:
                        base = float(bser.iloc[0]); last = float(bser.iloc[-1])
                        bench_pct = (last - base) / base * 100.0
            except Exception as e:
                logger.warning(f"diagnosis benchmark fetch: {e}")

        T = self.T
        colors = {"green": T.GREEN, "red": T.RED, "gold": T.GOLD, "text2": T.TEXT_2}
        plan_name = ""
        cur_item = self.plan_list.currentItem()
        if cur_item: plan_name = cur_item.data(Qt.UserRole).get("name", "")
        html = _build_diagnosis_report(plan_name, agg, self._ai_results, snaps, bench_pct, colors)
        self.diag_text.setHtml(html)
        self._diag_context_cache = (plan_name, agg, snaps, bench_pct)  # 供大模型润色复用
        self.diag_status_lbl.setText("已生成（离线模板）")

    def _copy_diagnosis(self):
        text = self.diag_text.toPlainText()
        if not text:
            QMessageBox.information(self, "提示", "还没有报告内容，先点「生成诊断报告」"); return
        QApplication.clipboard().setText(text)
        self.diag_status_lbl.setText("已复制到剪贴板")

    def _open_llm_settings(self):
        LLMSettingsDialog(parent=self).exec_()

    def _run_llm_narrative(self):
        if getattr(self, "_diag_context_cache", None) is None:
            QMessageBox.information(self, "提示", "先点「生成诊断报告」再润色"); return
        if self._llm_thread is not None and self._llm_thread.isRunning():
            return
        base_url = PaperWatchlistStore.get_config("llm_base_url")
        api_key = PaperWatchlistStore.get_config("llm_api_key")
        model = PaperWatchlistStore.get_config("llm_model")
        if not base_url or not api_key:
            QMessageBox.information(self, "还没配置",
                                    "先点「配置AI大模型」填好接口地址和API Key（不填也不影响离线报告）。")
            return

        plan_name, agg, snaps, bench_pct = self._diag_context_cache
        total_cost = sum(a["qty"] * a["avg_cost"] for a in agg.values())
        total_value = sum(a["qty"] * (self._ai_results[s]["cur_price"] if s in self._ai_results else a["avg_cost"])
                          for s, a in agg.items())
        context_text = _diagnosis_report_plaintext(self._ai_results, plan_name, agg, total_cost, total_value, bench_pct)

        self.diag_status_lbl.setText("大模型生成中…（网络请求，可能要几秒到十几秒）")
        th = LLMNarrativeThread(base_url, api_key, model, context_text)
        th.result_ready.connect(self._on_llm_result)
        th.error_msg.connect(self._on_llm_error)
        self._llm_thread = th
        th.start()

    def _on_llm_result(self, text: str):
        cur = self.diag_text.toHtml()
        T = self.T
        self.diag_text.append(
            f"<hr><p><b style='color:{T.GOLD}'>AI大模型点评：</b></p><p>{text}</p>"
            f"<p style='color:{T.TEXT_2};font-size:8pt;'>⚠ 以上为大模型生成的文字点评，同样基于上面的历史数据统计，不构成投资建议。</p>")
        self.diag_status_lbl.setText("大模型点评已生成")
        self._llm_thread = None

    def _on_llm_error(self, msg: str):
        self.diag_status_lbl.setText(f"大模型调用失败：{msg}（离线报告不受影响）")
        self._llm_thread = None

    # ── 历史走势预测 ─────────────────────────────────────────
    def _hist_symbol_input(self) -> str:
        raw = self.hist_sym_cb.currentText().strip()
        return raw.split()[0].upper() if raw else ""

    def _on_hist_period_clicked(self, label: str):
        self._hist_period_label = label
        self._draw_history_chart()

    def _refresh_history(self):
        sym = self._hist_symbol_input()
        if not sym:
            QMessageBox.information(self, "提示", "请输入代码"); return
        self.hist_status_lbl.setText(f"正在加载 {sym} 历史数据…")
        QApplication.processEvents()
        fetch = self.env["fetch_stock_data"]
        df = None
        try:
            df = fetch(sym, "max")
        except Exception as e:
            logger.warning(f"hist fetch max {sym}: {e}")
        if df is None or df.empty:
            try:
                df = fetch(sym, "10y")
            except Exception as e:
                logger.warning(f"hist fetch 10y {sym}: {e}")
        if df is None or df.empty:
            self.hist_info_lbl.setText(f"{sym} 没有取到历史数据（代码是否正确？）")
            self.hist_status_lbl.setText("")
            return
        self._hist_df_cache[sym] = df
        self._hist_cur_sym = sym
        self.hist_status_lbl.setText("")
        self._draw_history_chart()

    def _draw_history_chart(self):
        T = self.T
        fig = self.hist_canvas.figure
        fig.clear()
        sym = self._hist_cur_sym
        df = self._hist_df_cache.get(sym)
        if df is None or df.empty:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "选个代码点「加载」", ha="center", va="center",
                    color=T.TEXT_2, transform=ax.transAxes)
            self.hist_canvas.draw(); return

        days = dict(HIST_PERIODS).get(self._hist_period_label)
        sliced = _slice_period(df, days)
        if sliced is None or sliced.empty or len(sliced) < 2:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "该周期数据不足", ha="center", va="center",
                    color=T.TEXT_2, transform=ax.transAxes)
            self.hist_canvas.draw(); return

        cur_price = float(sliced["Close"].iloc[-1])
        start_price = float(sliced["Close"].iloc[0])
        period_high = float(sliced["High"].max())
        period_low = float(sliced["Low"].min())
        chg_pct = ((cur_price - start_price) / start_price * 100.0) if start_price else 0.0
        cur_sym = self.env["_currency"](sym)
        color = T.GREEN if chg_pct >= 0 else T.RED
        self.hist_info_lbl.setText(
            f"<b>{sym}</b>　现价 <b>{cur_sym}{cur_price:.2f}</b>　"
            f"区间({self._hist_period_label})涨跌 <b style='color:{color}'>{chg_pct:+.2f}%</b>　"
            f"区间最高 {cur_sym}{period_high:.2f}　最低 {cur_sym}{period_low:.2f}")
        self._hist_period_stats = {"start_price": start_price, "cur_price": cur_price,
                                   "high": period_high, "low": period_low}

        get_rc = self.env.get("_get_mpl_rc"); apply_font = self.env.get("_apply_font_to_figure")
        make_style = self.env.get("_make_mpf_style")
        try:
            with mpl.rc_context(get_rc() if get_rc else {}):
                if HAS_MPF and make_style is not None:
                    ohlcv = sliced[["Open", "High", "Low", "Close", "Volume"]].copy()
                    for c in ohlcv.columns:
                        ohlcv[c] = ohlcv[c].squeeze().astype(float)
                    ohlcv.dropna(inplace=True)
                    style = make_style()
                    mav_args = (5, 20) if len(ohlcv) < 60 else (5, 20, 60)
                    fig2, _axes = mpf.plot(
                        ohlcv, type="candle", volume=True, mav=mav_args, returnfig=True,
                        title=f"  {sym} ({self._hist_period_label})", style=style, figsize=(9, 4.0))
                    if apply_font: apply_font(fig2)
                    self.hist_canvas.figure = fig2
                    fig2.canvas = self.hist_canvas
                    # 跟主程序K线图一样的画布拉伸修复：窗口放大后图跟着填满，不留死白
                    try:
                        cw = max(self.hist_canvas.width(), 200); ch = max(self.hist_canvas.height(), 200)
                        dpi = fig2.get_dpi() or 100
                        fig2.set_size_inches(cw / dpi, ch / dpi)
                        try: fig2.tight_layout(pad=1.2)
                        except Exception: pass
                    except Exception as e:
                        logger.warning(f"hist canvas fit: {e}")
                    self.hist_canvas.draw_idle()
                    return
                else:
                    ax = fig.add_subplot(111)
                    ax.plot(sliced.index, sliced["Close"], color=T.GOLD, lw=1.4)
                    ax.axhline(start_price, color=T.TEXT_3, lw=0.7, ls="--")
                    ax.set_facecolor(T.MPL_AXES if hasattr(T, "MPL_AXES") else T.BG1)
                    ax.set_title(f"{sym} ({self._hist_period_label})", color=T.TEXT_H, fontsize=9)
        except Exception as e:
            logger.error(f"hist chart draw: {e}")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, f"画图失败: {e}", ha="center", va="center",
                    color=T.RED, transform=ax.transAxes)
        if apply_font: apply_font(fig)
        self.hist_canvas.draw()

    def _use_hist_for_position(self):
        sym = self._hist_cur_sym or self._hist_symbol_input()
        if not sym:
            QMessageBox.information(self, "提示", "先加载一个代码"); return
        stats = getattr(self, "_hist_period_stats", None)
        cur_price = stats["cur_price"] if stats else self.env["get_realtime_price"](sym)
        if cur_price is None:
            QMessageBox.warning(self, "提示", f"{sym} 取不到价格"); return
        name = ""
        for s, n in PRESET_STOCKS:
            if s == sym: name = n; break
        self.sym_cb.setCurrentText(f"{sym} {name}".strip())
        self.price_spin.setValue(round(cur_price, 2))
        self.tabs.setCurrentIndex(0)
        self.qty_spin.setFocus()

    def _run_hist_ai_predict(self):
        sym = self._hist_cur_sym
        if not sym:
            QMessageBox.information(self, "提示", "先加载一个代码再跑AI预测"); return
        missing = [k for k in ("TechnicalIndicators", "ProbabilityEngine") if k not in self.env]
        if missing:
            QMessageBox.warning(self, "缺少依赖", f"AI预测需要主程序提供 {missing}，当前没有，无法运行。")
            return
        if self._hist_ai_thread is not None and self._hist_ai_thread.isRunning():
            return
        name = ""
        for s, n in PRESET_STOCKS:
            if s == sym: name = n; break
        meta = {"name": name, "qty": 0.0, "avg_cost": 0.0}  # 还没建仓，纯看AI观点，不带成本解读
        self.hist_ai_btn.setEnabled(False)
        self.hist_status_lbl.setText(f"AI分析 {sym} 中…（训练模型要几秒到十几秒）")
        self.hist_ai_text.clear()
        th = AIAnalysisThread([(sym, meta)], self.env)
        th.result_ready.connect(self._on_hist_ai_result)
        th.finished_all.connect(self._on_hist_ai_finished)
        th.error_msg.connect(self._on_hist_ai_error)
        self._hist_ai_thread = th
        th.start()

    def _on_hist_ai_result(self, result: dict):
        sym = result["symbol"]
        self._hist_ai_result_cache[sym] = result
        cur = result["currency"]
        p5, p20, p60 = (result["probs"][h] * 100 for h in (5, 20, 60))
        text = (
            f"【{sym} AI趋势预测】市场状态：{result['regime']}\n"
            f"上涨概率：5日 {p5:.0f}%　20日 {p20:.0f}%　60日 {p60:.0f}%\n"
            f"综合信号：{result['signal_text']}\n"
        )
        if result.get("mc"):
            mc = result["mc"]
            text += f"20日蒙特卡洛价格带：P5 {mc['low_pct']:+.1f}% ~ P95 {mc['high_pct']:+.1f}%（中位数{mc['median_pct']:+.1f}%）\n"
        self.hist_ai_text.setPlainText(text)

        base_url = PaperWatchlistStore.get_config("llm_base_url")
        api_key = PaperWatchlistStore.get_config("llm_api_key")
        model = PaperWatchlistStore.get_config("llm_model")
        if base_url and api_key:
            stats = getattr(self, "_hist_period_stats", {}) or {}
            context = _single_symbol_context_text(
                sym, result.get("name") or "", self._hist_period_label,
                stats.get("start_price", result["cur_price"]), result["cur_price"],
                stats.get("high", result["cur_price"]), stats.get("low", result["cur_price"]),
                result)
            self.hist_status_lbl.setText("大模型点评生成中…")
            th = LLMNarrativeThread(base_url, api_key, model, context,
                                    system_prompt=SINGLE_SYMBOL_SYSTEM_PROMPT)
            th.result_ready.connect(self._on_hist_llm_result)
            th.error_msg.connect(self._on_hist_llm_error)
            self._hist_llm_thread = th
            th.start()

    def _on_hist_ai_finished(self, ok: int):
        self.hist_ai_btn.setEnabled(True)
        self._hist_ai_thread = None
        if not getattr(self, "_hist_llm_thread", None):
            self.hist_status_lbl.setText("AI预测完成" if ok else "AI预测失败（数据不足或取价失败）")

    def _on_hist_ai_error(self, msg: str):
        logger.warning(f"hist AI predict error: {msg}")

    def _on_hist_llm_result(self, text: str):
        self.hist_ai_text.append(f"\n【AI大模型点评】\n{text}\n\n⚠ 基于历史数据的概率参考，不构成投资建议。")
        self.hist_status_lbl.setText("大模型点评已生成")
        self._hist_llm_thread = None

    def _on_hist_llm_error(self, msg: str):
        self.hist_status_lbl.setText(f"大模型调用失败：{msg}（AI概率结果不受影响）")
        self._hist_llm_thread = None

    def closeEvent(self, event):
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self._ai_thread.stop()
            self._ai_thread.wait(3000)
        if getattr(self, "_llm_thread", None) is not None and self._llm_thread.isRunning():
            self._llm_thread.wait(1000)
        if getattr(self, "_hist_ai_thread", None) is not None and self._hist_ai_thread.isRunning():
            self._hist_ai_thread.stop()
            self._hist_ai_thread.wait(3000)
        if getattr(self, "_hist_llm_thread", None) is not None and self._hist_llm_thread.isRunning():
            self._hist_llm_thread.wait(1000)
        super().closeEvent(event)


# ══════════════════════════════════════════════════════════════════
# 一键安装入口
# ══════════════════════════════════════════════════════════════════
def install_paper_watchlist(env: dict):
    """env = 主程序 globals()。主窗口按钮行新增「模拟自选」按钮。"""
    required = ["QuantApp", "fetch_stock_data", "get_realtime_price", "T",
                "_currency", "_make_app_icon", "_get_mpl_rc", "_apply_font_to_figure"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_paper_watchlist: 主程序缺少 {missing}")

    PaperWatchlistStore.init_db()
    QuantApp_cls = env["QuantApp"]; T = env["T"]
    _orig_init = QuantApp_cls.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            from PyQt5.QtWidgets import QPushButton, QHBoxLayout
            btn = QPushButton("模拟自选")
            btn.setStyleSheet(btn.styleSheet() + f"color:{T.GOLD};border-color:{T.GOLD};")
            anchor = (getattr(self, "regime_btn", None) or getattr(self, "pair_btn", None)
                     or getattr(self, "coint_btn", None))
            inserted = False
            row = getattr(self, "btn_row", None)
            if isinstance(row, QHBoxLayout):
                idx = row.indexOf(anchor) if anchor is not None else -1
                if idx != -1:
                    row.insertWidget(idx + 1, btn)
                else:
                    row.insertWidget(max(0, row.count() - 1), btn)
                inserted = True
            if not inserted:
                self.layout().addWidget(btn)
            btn.clicked.connect(lambda: PaperWatchlistDialog(env, parent=self).exec_())
            self.paper_watchlist_btn = btn
        except Exception as e:
            logger.warning(f"paper watchlist button install: {e}")

    QuantApp_cls.__init__ = _new_init
    logger.info("[paper_watchlist] 已安装：主窗口「模拟自选」按钮")
    return PaperWatchlistDialog


# ══════════════════════════════════════════════════════════════════
# 独立自检（offscreen，无需联网/无需真实主程序）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import sys as _sys

    PaperWatchlistStore._DB_PATH = "quantpro_paper_selftest.db"
    if os.path.exists(PaperWatchlistStore._DB_PATH):
        os.remove(PaperWatchlistStore._DB_PATH)
    PaperWatchlistStore.init_db()

    # 1) 建方案 + 加仓 + 快照 是否正常落地
    pid = PaperWatchlistStore.create_plan("五巨头等权", note="自检")
    PaperWatchlistStore.add_position(pid, "CVX", "雪佛龙", 10, 150.0, "2026-06-01")
    PaperWatchlistStore.add_position(pid, "MSFT", "微软", 5, 420.0, "2026-06-01")
    positions = PaperWatchlistStore.list_positions(pid)
    assert len(positions) == 2, "持仓写入/读取失败"

    total_cost = sum(p["qty"] * p["buy_price"] for p in positions)
    PaperWatchlistStore.record_snapshot(pid, total_cost, total_cost * 1.05)
    PaperWatchlistStore.record_snapshot(pid, total_cost, total_cost * 1.08)  # 同日覆盖，不应变成2条
    snaps = PaperWatchlistStore.list_snapshots(pid)
    assert len(snaps) == 1 and abs(snaps[0]["pnl_pct"] - 8.0) < 1e-6, "快照upsert/涨幅计算有误"

    # 2) 编辑持仓（改数量/买入价/买入日期）是否生效
    PaperWatchlistStore.update_position(positions[0]["id"], 20, 160.0, "2026-06-15")
    updated = PaperWatchlistStore.list_positions(pid)
    edited = next(p for p in updated if p["id"] == positions[0]["id"])
    assert edited["qty"] == 20 and edited["buy_price"] == 160.0 and edited["buy_date"] == "2026-06-15", \
        "编辑持仓失败"
    print("编辑持仓自检通过 ✓")

    # 3) 删除持仓 / 方案
    PaperWatchlistStore.delete_position(positions[0]["id"])
    assert len(PaperWatchlistStore.list_positions(pid)) == 1, "删除持仓失败"
    PaperWatchlistStore.delete_plan(pid)
    assert len(PaperWatchlistStore.list_plans()) == 0, "删除方案失败"
    print("数据层自检通过 ✓ 方案/持仓/快照 增删改查均正常")

    # 3) AI分析纯函数 + AIAnalysisThread._analyze 流程自检（用轻量mock引擎，不跑真实XGB）
    assert _signal_label(0.70)[0] == "强烈看多"
    assert _signal_label(0.58)[0] == "看多"
    assert _signal_label(0.50)[0] == "中性"
    assert _signal_label(0.40)[0] == "看空"
    assert _signal_label(0.20)[0] == "强烈看空"
    assert "持有" in _position_advice("看多", 5.0)
    assert "评估" in _position_advice("看空", -5.0)
    agg_test = _aggregate_positions_by_symbol([
        {"symbol": "AAA", "name": "测试", "qty": 10, "buy_price": 100.0},
        {"symbol": "AAA", "name": "测试", "qty": 10, "buy_price": 120.0},
    ])
    assert abs(agg_test["AAA"]["avg_cost"] - 110.0) < 1e-9, "加权平均成本计算有误"
    print("AI纯函数自检通过 ✓ 信号分级/持仓解读/加权成本均正常")

    class _MockTI:
        @staticmethod
        def compute_all(df):
            out = pd.DataFrame(index=df.index)
            out["close"] = df["Close"]
            return out

    class _MockPE:
        def __init__(self, ind, close):
            self.close = close
        def detect_regime(self): return "neutral"
        def ensemble(self, fwd=5, sentiment_agg=None, symbol=None):
            base = 0.62 if fwd == 20 else 0.55
            return base, base, base, base, 0.45, 0.35, 0.20
        def feature_importance(self, fwd=20):
            return [("rsi", 0.30), ("mom20", 0.20), ("ma_dev20", 0.15)]
        def monte_carlo(self, days=20, n=3000, use_fat_tail=True):
            return {"up_prob": 0.6, "low_pct": -8.0, "high_pct": 10.0, "median_pct": 1.0,
                    "var95": -8.0, "cvar95": -10.0, "finals": np.array([1, 2]), "cur": 100.0}

    class _MockSCE:
        @classmethod
        def save_prediction(cls, *a, **kw): pass
        @classmethod
        def get_accuracy_report(cls, symbol=None): return {}

    _rng_ai = np.random.default_rng(7)
    _idx_ai = pd.bdate_range("2024-01-01", periods=400)
    _df_ai = pd.DataFrame({"Close": pd.Series(100 * np.cumprod(1 + _rng_ai.normal(0.0003, 0.01, 400)),
                                              index=_idx_ai)})

    ai_env = {
        "fetch_stock_data": lambda sym, period="6mo": _df_ai,
        "get_realtime_price": lambda sym: 105.0,
        "_currency": lambda s: "$",
        "TechnicalIndicators": _MockTI,
        "ProbabilityEngine": _MockPE,
        "SelfCorrectionEngine": _MockSCE,
    }
    th = AIAnalysisThread([("AAA", {"name": "测试标的", "qty": 10, "avg_cost": 100.0})], ai_env)
    result = th._analyze("AAA", {"name": "测试标的", "qty": 10, "avg_cost": 100.0})
    assert result is not None, "AI分析流程返回空"
    assert set(result["probs"].keys()) == {5, 20, 60}, "概率字典horizon缺失"
    assert result["signal_text"] in ("强烈看多", "看多", "中性", "看空", "强烈看空")
    assert result["feat_imp"], "特征重要性为空"
    assert result["mc"] is not None, "蒙特卡洛结果为空"
    assert abs(result["pnl_pct"] - 5.0) < 1e-6, "持仓盈亏%计算有误"
    print(f"AI分析流程自检通过 ✓ 综合信号={result['signal_text']} 解读={result['advice']}")

    # 4) 组合诊断报告 + 配置存取 自检
    PaperWatchlistStore.set_config("llm_base_url", "https://api.example.com/v1")
    assert PaperWatchlistStore.get_config("llm_base_url") == "https://api.example.com/v1", "配置写入失败"
    PaperWatchlistStore.set_config("llm_base_url", "https://api.example.com/v2")  # upsert覆盖
    assert PaperWatchlistStore.get_config("llm_base_url") == "https://api.example.com/v2", "配置覆盖(upsert)失败"
    assert PaperWatchlistStore.get_config("no_such_key", "默认值") == "默认值", "默认值兜底失败"

    _agg = {"AAA": {"name": "测试标的", "qty": 10.0, "avg_cost": 100.0}}
    _ai_results_fake = {"AAA": result}
    _snaps_fake = [
        {"snap_date": "2026-06-01", "total_cost": 1000.0, "total_value": 1000.0, "pnl_pct": 0.0},
        {"snap_date": "2026-07-01", "total_cost": 1000.0, "total_value": 1050.0, "pnl_pct": 5.0},
    ]
    _colors = {"green": "#10b981", "red": "#ef4444", "gold": "#f59e0b", "text2": "#8ba3be"}
    html = _build_diagnosis_report("测试方案", _agg, _ai_results_fake, _snaps_fake, 3.0, _colors)
    assert "测试方案" in html and "AAA" in html and "跑赢" in html, "诊断报告HTML内容不完整"
    plain = _diagnosis_report_plaintext(_ai_results_fake, "测试方案", _agg, 1000.0, 1050.0, 3.0)
    assert "AAA" in plain and "20日↑概率" in plain, "诊断报告纯文本内容不完整"
    print("组合诊断报告自检通过 ✓ HTML/纯文本生成、配置存取(含upsert/默认值)均正常")

    # 5) 历史走势Tab纯函数自检：周期切片 + 大模型上下文文本
    _hist_idx = pd.bdate_range("2020-01-01", periods=2000)
    _hist_df = pd.DataFrame({
        "Open": np.linspace(50, 150, 2000), "High": np.linspace(51, 151, 2000),
        "Low": np.linspace(49, 149, 2000), "Close": np.linspace(50, 150, 2000),
        "Volume": np.full(2000, 1_000_000.0),
    }, index=_hist_idx)
    for _label, _days in HIST_PERIODS:
        sliced = _slice_period(_hist_df, _days)
        if _days is None:
            assert len(sliced) == len(_hist_df), f"{_label}(全部)切片应等于全量"
        else:
            assert len(sliced) == min(_days + 1, len(_hist_df)), f"{_label}切片长度不对"
    tiny_df = _hist_df.iloc[-3:]
    assert len(_slice_period(tiny_df, 252)) == len(tiny_df), "数据不够时应返回全部而不报错"

    ctx = _single_symbol_context_text("AAA", "测试标的", "1年", 100.0, 120.0, 130.0, 90.0, result)
    assert "AAA" in ctx and "20日" in ctx and "区间起始价" in ctx, "单标的大模型上下文文本不完整"
    ctx_no_ai = _single_symbol_context_text("AAA", "测试标的", "1年", 100.0, 120.0, 130.0, 90.0, None)
    assert "尚未跑AI趋势预测" in ctx_no_ai, "无AI结果时的兜底文案缺失"
    print("历史走势Tab纯函数自检通过 ✓ 周期切片(含全部/数据不足兜底)、大模型上下文文本均正常")

    # 3) 窗口冒烟（合成价格，不联网）
    class _T:
        BG0='#070d14'; BG1='#0d1826'; BG2='#111f30'; BORDER='#1e3452'
        GOLD='#f59e0b'; ACCENT='#0ea5e9'; CYAN='#06b6d4'; GREEN='#10b981'; RED='#ef4444'
        TEXT_H='#e2eaf3'; TEXT_1='#c8d8e8'; TEXT_2='#8ba3be'; TEXT_3='#4e6a82'
        MPL_BG='#070d14'; MPL_AXES='#0d1826'

    def _style(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(_T.MPL_AXES); ax.set_title(title, color=_T.TEXT_H, fontsize=9)

    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-01-01", periods=400)
    _close_ser = pd.Series(4200 * np.cumprod(1 + rng.normal(0.0004, 0.008, 400)), index=idx)
    bench_df = pd.DataFrame({
        "Open": _close_ser * 0.998, "High": _close_ser * 1.01,
        "Low": _close_ser * 0.99, "Close": _close_ser,
        "Volume": pd.Series(1_000_000.0, index=idx),
    })

    def _fetch(sym, period="6mo"):
        return bench_df

    def _rt(sym):
        return 155.0

    env = {
        "QuantApp": None, "fetch_stock_data": _fetch, "get_realtime_price": _rt, "T": _T,
        "GLOBAL_STYLE": "", "_currency": lambda s: "$", "_make_app_icon": lambda: None,
        "_get_mpl_rc": lambda: {}, "_apply_font_to_figure": lambda f: None, "_mpl_style": _style,
    }
    app = QApplication(_sys.argv)
    PaperWatchlistStore._DB_PATH = "quantpro_paper_selftest2.db"
    if os.path.exists(PaperWatchlistStore._DB_PATH):
        os.remove(PaperWatchlistStore._DB_PATH)
    dlg = PaperWatchlistDialog(env)
    dlg._new_plan  # 存在性检查（真正弹QInputDialog不在自检里跑）
    pid2 = PaperWatchlistStore.create_plan("窗口自检方案")
    PaperWatchlistStore.add_position(pid2, "AAA", "测试标的", 10, 100.0, "2025-01-01")
    dlg._reload_plans(select_id=pid2)
    assert dlg.pos_table.rowCount() == 1, "持仓表未正确渲染"
    dlg._refresh_prices()
    dlg.tabs.setCurrentIndex(1)

    # 组合诊断/历史走势 Tab冒烟：5个Tab都在、离线报告能生成、没配置大模型时不崩溃
    assert dlg.tabs.count() == 5, "Tab数量不对（应为：持仓明细/历史走势预测/收益走势/AI趋势分析/组合诊断）"
    assert dlg.tabs.tabText(1) == "历史走势预测"
    assert dlg.tabs.tabText(4) == "组合诊断"
    import PyQt5.QtWidgets as _qw
    _qw.QMessageBox.information = staticmethod(lambda *a, **k: None)  # 避免弹窗阻塞自检
    _qw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    dlg._generate_diagnosis()
    assert "窗口自检方案" in dlg.diag_text.toPlainText(), "离线诊断报告未正确生成"
    dlg._run_llm_narrative()  # 未配置API，应走提示分支而不崩溃

    # 历史走势预测Tab：加载 → 切周期 → 用价格建仓 → AI预测(env缺ML依赖，应走警告分支不崩溃)
    dlg.hist_sym_cb.setCurrentText("AAA 测试标的")
    dlg._refresh_history()
    assert dlg._hist_cur_sym == "AAA", "历史数据加载后当前代码未更新"
    assert "AAA" in dlg.hist_info_lbl.text(), "历史区间信息未正确显示"
    dlg._on_hist_period_clicked("5年")
    assert dlg._hist_period_label == "5年", "周期切换未生效"
    dlg.price_spin.setValue(0)
    dlg._use_hist_for_position()
    assert dlg.tabs.currentIndex() == 0, "用此价格建仓后应跳回持仓明细Tab"
    assert dlg.price_spin.value() > 0, "用此价格建仓后买入价未回填"
    dlg._run_hist_ai_predict()  # env没有TechnicalIndicators/ProbabilityEngine，应弹警告而不崩溃
    print(f"窗口冒烟: 持仓行数={dlg.pos_table.rowCount()} 方案数={len(PaperWatchlistStore.list_plans())} "
         f"Tab数={dlg.tabs.count()} 历史走势当前代码={dlg._hist_cur_sym}")

    # 4) 持仓排序 + 方案涨幅徽章 自检（多标的、不同涨跌幅，验证默认排序/切换排序/排名徽章/方案徽章）
    _prices = {"WINNER": 150.0, "MID": 105.0, "LOSER": 80.0}  # 对应买入价100 → +50%/+5%/-20%

    def _rt2(sym):
        return _prices.get(sym)

    env2 = dict(env); env2["get_realtime_price"] = _rt2
    m_dlg_db = "quantpro_paper_sorttest.db"
    if os.path.exists(m_dlg_db): os.remove(m_dlg_db)
    PaperWatchlistStore._DB_PATH = m_dlg_db
    dlg2 = PaperWatchlistDialog(env2)
    pid3 = PaperWatchlistStore.create_plan("排序自检方案")
    PaperWatchlistStore.add_position(pid3, "LOSER", "", 10, 100.0, "2025-01-01")
    PaperWatchlistStore.add_position(pid3, "WINNER", "", 10, 100.0, "2025-01-01")
    PaperWatchlistStore.add_position(pid3, "MID", "", 10, 100.0, "2025-01-01")
    dlg2._reload_plans(select_id=pid3)

    # 默认「涨幅%（高→低）」：WINNER应排第一(🥇)，LOSER应排最后
    assert dlg2.sort_combo.currentText() == SORT_MODES[0], "默认排序方式应为涨幅%（高→低）"
    assert dlg2.pos_table.item(0, 1).text() == "WINNER", "默认降序排序：涨幅最高的应排第一行"
    assert dlg2.pos_table.item(0, 0).text() == "🥇", "第一名应显示金牌图标"
    assert dlg2.pos_table.item(1, 0).text() == "🥈", "第二名应显示银牌图标"
    assert dlg2.pos_table.item(2, 0).text() == "🥉", "第三名应显示铜牌图标"
    assert dlg2.pos_table.item(2, 1).text() == "LOSER", "亏损最多的应排最后一行"

    # 切换成「涨幅%（低→高）」：顺序应完全反过来
    dlg2.sort_combo.setCurrentIndex(1)
    assert dlg2.pos_table.item(0, 1).text() == "LOSER", "切换到升序后，跌得最多的应排第一"
    assert dlg2.pos_table.item(2, 1).text() == "WINNER", "切换到升序后，涨得最多的应排最后"

    # 切换成「股票代码（A→Z）」：应按字母排
    dlg2.sort_combo.setCurrentIndex(4)
    syms_order = [dlg2.pos_table.item(i, 1).text() for i in range(3)]
    assert syms_order == sorted(syms_order), "按代码排序未生效"

    # 仓位占比%列：三笔等金额买入，理论上各占1/3左右
    dlg2.sort_combo.setCurrentIndex(0)
    pos_pcts = [dlg2.pos_table.item(i, 7).text() for i in range(3)]
    assert all(x.endswith("%") for x in pos_pcts), "仓位占比%列未正确渲染"

    # 左侧方案列表：涨幅徽章应显示、且方案排前面（列表里只有这一个方案）
    item0 = dlg2.plan_list.item(0)
    assert "🟢" in item0.text() or "🔴" in item0.text(), "方案列表应显示涨跌徽章"
    assert "%" in item0.text(), "方案列表应显示涨幅%"

    for f in (m_dlg_db,):
        try: os.remove(f)
        except Exception: pass
    print("持仓排序+方案涨幅徽章自检通过 ✓ 默认降序/升序切换/按代码排序/排名徽章/方案徽章均正常")



    for f in (PaperWatchlistStore._DB_PATH, "quantpro_paper_selftest.db", "quantpro_paper_selftest2.db"):
        try: os.remove(f)
        except Exception: pass
    print("全部自检通过 ✓")
