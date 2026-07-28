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
    QProgressBar,
)
from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# 五个用户点名的标的（可在弹窗里自由增删，这只是默认建议清单）
PRESET_STOCKS = [
    ("CVX", "雪佛龙"),
    ("MSFT", "微软"),
    ("INTC", "英特尔"),
    ("KO", "可口可乐"),
    ("LMT", "洛克希德马丁"),
]


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
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self.resize(1180, 720)

        PaperWatchlistStore.init_db()
        self.cur_plan_id: Optional[int] = None

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

        self.pos_table = QTableWidget(0, 10)
        self.pos_table.setHorizontalHeaderLabels(
            ["代码", "名称", "数量", "买入价", "买入日期", "现价", "现值", "浮动盈亏", "盈亏%", "持有天数"])
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

        self._ai_thread = None
        self._ai_results: Dict[str, Dict] = {}

        root.addLayout(right, 1)

        self._reload_plans()

    # ── 方案管理 ─────────────────────────────────────────────
    def _reload_plans(self, select_id: Optional[int] = None):
        self.plan_list.blockSignals(True)
        self.plan_list.clear()
        plans = PaperWatchlistStore.list_plans()
        target_row = 0
        for i, p in enumerate(plans):
            it = QListWidgetItem(f"{p['name']}   ({p['created_date']})")
            it.setData(Qt.UserRole, p)
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

    def _on_plan_changed(self, cur, _prev):
        if cur is None:
            self.cur_plan_id = None; return
        p = cur.data(Qt.UserRole)
        self.cur_plan_id = p["id"]
        self.plan_note_lbl.setText(p.get("note") or "")
        self._reload_positions()
        self._refresh_chart()
        self.ai_table.setRowCount(0)
        self._ai_results = {}
        self.ai_status_lbl.setText("")

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

    def _delete_position(self):
        row = self.pos_table.currentRow()
        if row < 0: return
        pos_id = self.pos_table.item(row, 0).data(Qt.UserRole)
        if pos_id is None: return
        PaperWatchlistStore.delete_position(pos_id)
        self._reload_positions()

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

    def _reload_positions(self):
        self.pos_table.setRowCount(0)
        if self.cur_plan_id is None: return
        positions = PaperWatchlistStore.list_positions(self.cur_plan_id)
        T = self.T
        total_cost = total_value = 0.0
        today = date.today()
        for p in positions:
            r = self.pos_table.rowCount(); self.pos_table.insertRow(r)
            cur_price = self.env["get_realtime_price"](p["symbol"])
            if cur_price is None:
                try:
                    df = self.env["fetch_stock_data"](p["symbol"], "5d")
                    cur_price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
                except Exception:
                    cur_price = None
            cost = p["qty"] * p["buy_price"]
            total_cost += cost
            cur = p["currency"] or "$"
            try:
                bd = datetime.strptime(p["buy_date"], "%Y-%m-%d").date()
                held_days = (today - bd).days
            except Exception:
                held_days = 0

            item0 = QTableWidgetItem(p["symbol"]); item0.setData(Qt.UserRole, p["id"])
            self.pos_table.setItem(r, 0, item0)
            self.pos_table.setItem(r, 1, QTableWidgetItem(p["name"] or ""))
            self.pos_table.setItem(r, 2, QTableWidgetItem(f"{p['qty']:g}"))
            self.pos_table.setItem(r, 3, QTableWidgetItem(f"{cur}{p['buy_price']:.2f}"))
            self.pos_table.setItem(r, 4, QTableWidgetItem(p["buy_date"]))

            if cur_price is not None:
                value = p["qty"] * cur_price
                total_value += value
                pnl = value - cost
                pnl_pct = (pnl / cost * 100.0) if cost else 0.0
                color = T.GREEN if pnl >= 0 else T.RED
                self.pos_table.setItem(r, 5, QTableWidgetItem(f"{cur}{cur_price:.2f}"))
                self.pos_table.setItem(r, 6, QTableWidgetItem(f"{cur}{value:,.2f}"))
                pnl_item = QTableWidgetItem(f"{cur}{pnl:+,.2f}"); pnl_item.setForeground(QColor(color))
                self.pos_table.setItem(r, 7, pnl_item)
                pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%"); pct_item.setForeground(QColor(color))
                self.pos_table.setItem(r, 8, pct_item)
            else:
                total_value += cost  # 取不到价时按成本计入，避免总值失真
                for col in (5, 6, 7, 8):
                    self.pos_table.setItem(r, col, QTableWidgetItem("取价失败"))
            self.pos_table.setItem(r, 9, QTableWidgetItem(str(held_days)))

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

    def closeEvent(self, event):
        if self._ai_thread is not None and self._ai_thread.isRunning():
            self._ai_thread.stop()
            self._ai_thread.wait(3000)
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
    bench_df = pd.DataFrame({"Close": pd.Series(4200 * np.cumprod(1 + rng.normal(0.0004, 0.008, 400)), index=idx)})

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
    print(f"窗口冒烟: 持仓行数={dlg.pos_table.rowCount()} 方案数={len(PaperWatchlistStore.list_plans())}")

    for f in (PaperWatchlistStore._DB_PATH, "quantpro_paper_selftest.db"):
        try: os.remove(f)
        except Exception: pass
    print("全部自检通过 ✓")
