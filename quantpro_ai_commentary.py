"""
QuantPro — AI 文字分析师 v1.1（修复新闻情感缺失）
══════════════════════════════════════════════════════════════════
v1.1 修复：
  ① collect_context 从多个属性名尝试提取情感数据
  ② 润色前强制重新收集上下文（拿到最新情感）
  ③ Hook _on_news：新闻加载完成后自动刷新 AI tab
  ④ AI tab 首次刷新延迟 1500ms → 2500ms
  ⑤ 情感仍缺失时在 LLM 上下文显式说明（避免LLM瞎猜）

集成（quantpro_v1_6.py 的 __main__ 里，v15 之后、QuantApp 之前）：

    from quantpro_ai_commentary import install_ai_commentary
    install_ai_commentary(globals())
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser,
    QApplication, QMessageBox, QWidget, QMenu, QAction
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# LLM 配置读取
# ══════════════════════════════════════════════════════════════════
def _get_llm_config() -> Dict[str, str]:
    try:
        from quantpro_paper_watchlist import PaperWatchlistStore
        return {
            "base_url": PaperWatchlistStore.get_config("llm_base_url"),
            "api_key":  PaperWatchlistStore.get_config("llm_api_key"),
            "model":    PaperWatchlistStore.get_config("llm_model"),
        }
    except Exception:
        return {"base_url": "", "api_key": "", "model": ""}


# ══════════════════════════════════════════════════════════════════
# 【v1.1 新增】情感数据多路径提取
# ══════════════════════════════════════════════════════════════════
def _extract_sentiment_from_dialog(dialog) -> Optional[Dict]:
    """
    从 dialog 的多个可能属性名尝试获取情感数据。
    主程序写入 _sentiment_agg；本模块可能写 _ai_last_sentiment；
    未来若主程序改名，多个候选依次尝试，任一命中即用。
    """
    if dialog is None:
        return None
    for attr in ("_sentiment_agg", "_last_sentiment_agg",
                 "_ai_last_sentiment", "sentiment_agg"):
        val = getattr(dialog, attr, None)
        if not val:
            continue
        try:
            d = dict(val)
            # 至少要有一条新闻的情感结果才认
            if (d.get("n_total", 0) > 0
                    or (d.get("n_pos", 0) + d.get("n_neg", 0)
                        + d.get("n_neu", 0)) > 0):
                return d
        except Exception:
            continue
    return None


# ══════════════════════════════════════════════════════════════════
# 上下文收集
# ══════════════════════════════════════════════════════════════════
def _safe(v, default=None):
    try:
        if v is None:
            return default
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def collect_context(env: dict, sym: str,
                    dialog=None, row: Optional[Dict] = None) -> Dict:
    ctx: Dict[str, Any] = {
        "symbol": sym, "name": "", "currency": "$",
        "price": None, "change_pct": None,
    }

    fetch = env.get("fetch_stock_data")
    get_rt = env.get("get_realtime_price")
    currency_fn = env.get("_currency", lambda s: "$")
    ctx["currency"] = currency_fn(sym)

    # ── 从扫描行拿基础信息 ──
    if row:
        ctx["name"] = row.get("col_name", "") or ""
        ctx["price"] = _safe(row.get("col_price"))
        ctx["change_pct"] = _safe(row.get("col_chg"))
        ctx["rsi"] = _safe(row.get("col_rsi"))
        ctx["ma20"] = _safe(row.get("col_ma20"))
        ctx["ma60"] = _safe(row.get("col_ma60"))
        ctx["atr"] = _safe(row.get("col_atr"))
        ctx["signal_text"] = row.get("col_signal")
        ctx["risk_label"] = row.get("col_risk")
        ctx["cs_rank"] = row.get("col_cs_rank")
        ctx["rel_str"] = _safe(row.get("col_rel_str"))

    # ── 从 dialog 拿现成分析结果 ──
    if dialog is not None:
        try:
            if not ctx["name"] and hasattr(dialog, "name"):
                ctx["name"] = dialog.name or ""

            # 概率卡片
            if hasattr(dialog, "prob_cards"):
                for key, tag in (("p5", "prob_5d"),
                                 ("p20", "prob_20d"),
                                 ("p60", "prob_60d")):
                    try:
                        txt = dialog.prob_cards[key].text().replace("%", "").strip()
                        if txt and txt != "--":
                            ctx[tag] = float(txt) / 100.0
                    except Exception:
                        pass

            # 【v1.1 修复】情感 —— 多路径提取
            senti = _extract_sentiment_from_dialog(dialog)
            if senti:
                ctx["sentiment"] = senti
                logger.debug(f"[ai_commentary] 情感数据来源已命中，"
                             f"n_total={senti.get('n_total')}")

            # 价格目标 / MC
            if hasattr(dialog, "_target_data") and dialog._target_data:
                mc_bands = dialog._target_data.get("mc_bands", {})
                if 20 in mc_bands:
                    band = mc_bands[20]
                    ctx["mc_20d"] = {
                        "low_pct": _safe(band.get("p5_pct")),
                        "median_pct": _safe(band.get("p50_pct")),
                        "high_pct": _safe(band.get("p95_pct")),
                        "up_prob": _safe(band.get("up_prob")),
                    }

            if hasattr(dialog, "_ind") and dialog._ind is not None:
                ctx["_ind_df"] = dialog._ind
        except Exception as e:
            logger.warning(f"[ai_commentary] 从 dialog 取数异常: {e}")

    # ── 用 _ind 补齐技术指标 ──
    ind = ctx.pop("_ind_df", None)
    if ind is not None and not ind.empty:
        last = ind.iloc[-1]
        ctx.setdefault("rsi", _safe(last.get("rsi")))
        ctx.setdefault("ma20", _safe(last.get("ma20")))
        ctx.setdefault("ma60", _safe(last.get("ma60")))
        ctx.setdefault("ma200", _safe(last.get("ma200")))
        ctx.setdefault("atr", _safe(last.get("atr")))
        ctx.setdefault("price", _safe(last.get("close")))

    # ── 缺基础数据则现算 ──
    if fetch is not None and (ctx["price"] is None or ctx.get("rsi") is None):
        try:
            df = fetch(sym, "6mo")
            if df is not None and not df.empty:
                TI = env.get("TechnicalIndicators")
                if TI is not None:
                    ind2 = TI.compute_all(df)
                    if ind2 is not None and not ind2.empty:
                        last = ind2.iloc[-1]
                        ctx.setdefault("rsi", _safe(last.get("rsi")))
                        ctx.setdefault("ma20", _safe(last.get("ma20")))
                        ctx.setdefault("ma60", _safe(last.get("ma60")))
                        ctx.setdefault("ma200", _safe(last.get("ma200")))
                        ctx.setdefault("atr", _safe(last.get("atr")))
                        ctx.setdefault("price", _safe(last.get("close")))
        except Exception as e:
            logger.warning(f"[ai_commentary] 现算基础指标失败: {e}")

    # 实时价覆盖
    if get_rt is not None:
        try:
            rp = get_rt(sym)
            if rp is not None and rp > 0:
                ctx["price"] = float(rp)
        except Exception:
            pass

    # ── 量价诊断 ──
    try:
        from quantpro_volume_analysis import compute_volume_features
        if fetch is not None:
            df = fetch(sym, "6mo")
            if df is not None and not df.empty:
                vp = compute_volume_features(df)
                if "error" not in vp:
                    ctx["volume_price"] = vp.get("summary", {})
    except Exception as e:
        logger.debug(f"[ai_commentary] 量价诊断缺失: {e}")

    # ── EVT 尾部风险 ──
    try:
        EVT = env.get("EVTRiskEngine")
        if EVT is not None and fetch is not None:
            df = fetch(sym, "2y")
            if df is not None and not df.empty:
                closes = df["Close"]
                if isinstance(closes, pd.DataFrame):
                    closes = closes.iloc[:, 0]
                rets = closes.astype(float).pct_change().dropna()
                if len(rets) >= 100:
                    evt = EVT.tail_risk_report(rets)
                    ctx["evt"] = {
                        "var99": _safe(evt.get("var_99", {}).get("var")),
                        "cvar99": _safe(evt.get("var_99", {}).get("cvar")),
                        "var95": _safe(evt.get("var_95", {}).get("var")),
                        "cvar95": _safe(evt.get("var_95", {}).get("cvar")),
                        "xi": _safe(evt.get("var_99", {}).get("xi")),
                        "verdict": evt.get("verdict", ""),
                    }
    except Exception as e:
        logger.debug(f"[ai_commentary] EVT 缺失: {e}")

    # ── Regime ──
    try:
        PE = env.get("ProbabilityEngine")
        if PE is not None and fetch is not None:
            df = fetch(sym, "1y")
            if df is not None and not df.empty:
                TI = env.get("TechnicalIndicators")
                if TI is not None:
                    ind3 = TI.compute_all(df)
                    if ind3 is not None and not ind3.empty:
                        pe = PE(ind3, ind3["close"])
                        ctx["regime"] = pe.detect_regime()
    except Exception as e:
        logger.debug(f"[ai_commentary] regime 缺失: {e}")

    # ── Fama-French ──
    try:
        FF = env.get("FamaFrenchFactorModel")
        if FF is not None and fetch is not None:
            df = fetch(sym, "2y")
            if df is not None and not df.empty:
                closes = df["Close"]
                if isinstance(closes, pd.DataFrame):
                    closes = closes.iloc[:, 0]
                rets = closes.astype(float).pct_change()
                factors = FF.build_factor_returns(fetch, "2y")
                if factors is not None and not factors.empty:
                    ctx["ff"] = FF.regress(rets, factors)
    except Exception as e:
        logger.debug(f"[ai_commentary] FF 缺失: {e}")

    # ── 微观结构 ──
    try:
        MS = env.get("MicrostructureFeatures")
        if MS is not None and fetch is not None:
            df = fetch(sym, "6mo")
            if df is not None and not df.empty and len(df) >= 60:
                ms = MS.compute_all(df)
                if ms is not None and not ms.empty:
                    valid = ms.dropna()
                    if not valid.empty:
                        last = valid.iloc[-1]
                        ctx["micro"] = {
                            "amihud": _safe(last.get("amihud")),
                            "roll_spread": _safe(last.get("roll_spread")),
                            "gk_vol": _safe(last.get("gk_vol")),
                            "kyle_lambda": _safe(last.get("kyle_lambda")),
                        }
    except Exception as e:
        logger.debug(f"[ai_commentary] microstructure 缺失: {e}")

    return ctx


# ══════════════════════════════════════════════════════════════════
# 离线研报模板
# ══════════════════════════════════════════════════════════════════
def _pct(v, digits=2):
    if v is None:
        return "--"
    return f"{v:+.{digits}f}%"


def _prob_pct(p):
    if p is None:
        return "--"
    return f"{p * 100:.0f}%"


def build_commentary(ctx: Dict) -> str:
    sym = ctx.get("symbol", "?")
    name = ctx.get("name", "") or ""
    cur = ctx.get("currency", "$")
    price = ctx.get("price")
    chg = ctx.get("change_pct")

    lines: List[str] = []
    head = f"{sym}"
    if name and name != sym:
        head += f"（{name}）"
    if price is not None:
        head += f"  现价 {cur}{price:.2f}"
        if chg is not None:
            head += f"（{chg:+.2f}%）"
    lines.append(f"【{head}】")

    # ① 综合结论
    lines.append("")
    lines.append("◆ 综合结论")
    p20 = ctx.get("prob_20d")
    sig = ctx.get("signal_text")
    regime = ctx.get("regime")
    regime_cn = {"high_vol": "高波动", "low_vol": "低波动", "neutral": "中性",
                 "calm_bull": "温和上行", "calm_bear": "温和下行",
                 "volatile": "震荡", "crisis": "危机"}.get(regime, regime or "未知")
    bits = []
    if p20 is not None:
        if p20 >= 0.62:
            bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（偏多）")
        elif p20 <= 0.38:
            bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（偏空）")
        else:
            bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（中性）")
    if sig:
        bits.append(f"扫描信号：{sig}")
    if regime:
        bits.append(f"市场环境：{regime_cn}")
    lines.append("  " + "；".join(bits) + "。" if bits else "  数据不足。")

    # ② 技术
    lines.append("")
    lines.append("◆ 价格位置与技术")
    tb = []
    for tag, label in (("ma20", "MA20"), ("ma60", "MA60"), ("ma200", "MA200")):
        ma = ctx.get(tag)
        if price and ma:
            tb.append(f"相对 {label} {_pct((price - ma) / ma * 100)}")
    rsi = ctx.get("rsi")
    if rsi is not None:
        note = ("超买" if rsi >= 70 else "超卖" if rsi <= 30
                else "偏强" if rsi >= 55 else "偏弱" if rsi <= 45 else "中性")
        tb.append(f"RSI={rsi:.1f}（{note}）")
    atr = ctx.get("atr")
    if atr is not None and price:
        tb.append(f"ATR≈{cur}{atr:.2f}（{atr/price*100:.2f}%）")
    lines.append("  " + "；".join(tb) + "。" if tb else "  技术指标数据不足。")

    # ③ 概率
    lines.append("")
    lines.append("◆ 概率展望")
    lines.append(f"  5日 {_prob_pct(ctx.get('prob_5d'))}　|　"
                 f"20日 {_prob_pct(ctx.get('prob_20d'))}　|　"
                 f"60日 {_prob_pct(ctx.get('prob_60d'))}")
    mc = ctx.get("mc_20d")
    if mc:
        lines.append(f"  20日MC：P5 {_pct(mc.get('low_pct'))} ~ "
                     f"P95 {_pct(mc.get('high_pct'))}，"
                     f"中位数 {_pct(mc.get('median_pct'))}")

    # ④ 量价
    vp = ctx.get("volume_price")
    if vp:
        lines.append("")
        lines.append("◆ 量价结构")
        bits = []
        if vp.get("obv_vs_ma"):
            bits.append(f"OBV {vp['obv_vs_ma']}")
        rc = vp.get("roll_corr_60d")
        if rc is not None and np.isfinite(rc):
            note = ("量价同向" if rc > 0.3 else "量价背离" if rc < -0.3 else "量价弱相关")
            bits.append(f"60日相关 {rc:+.2f}（{note}）")
        vi = vp.get("vol_imbalance_20d")
        if vi is not None and np.isfinite(vi):
            note = "净流入" if vi > 0.1 else "净流出" if vi < -0.1 else "均衡"
            bits.append(f"资金 {vi:+.2f}（{note}）")
        am = vp.get("amihud_illiq")
        if am is not None and np.isfinite(am):
            bits.append(f"Amihud={am:.3f}")
        gk = vp.get("gk_vol_annual")
        if gk is not None and np.isfinite(gk):
            bits.append(f"GK波动 {gk*100:.1f}%")
        if bits:
            lines.append("  " + "；".join(bits) + "。")
        if vp.get("regime_hint"):
            lines.append(f"  研判：{vp['regime_hint']}。")

    # ⑤ 风险
    lines.append("")
    lines.append("◆ 风险提示")
    evt = ctx.get("evt")
    rb = []
    if evt:
        for k, label, scale in (("var99", "VaR99", 100), ("cvar99", "CVaR99", 100),
                                ("xi", "尾部指数 ξ", 1)):
            v = evt.get(k)
            if v is not None:
                if scale == 100:
                    rb.append(f"{label} {v*scale:.2f}%")
                else:
                    rb.append(f"{label}={v:.3f}")
    micro = ctx.get("micro")
    if micro:
        rs = micro.get("roll_spread")
        if rs is not None and np.isfinite(rs):
            rb.append(f"Roll价差 {rs*100:.3f}%")
    lines.append("  " + "；".join(rb) + "。" if rb else "  尾部风险数据不足。")
    if evt and evt.get("verdict"):
        lines.append(f"  研判：{evt['verdict']}。")

    # ⑥ 情感
    senti = ctx.get("sentiment")
    if senti:
        lines.append("")
        lines.append("◆ 新闻情感")
        label = senti.get("label", "neutral")
        label_cn = {"positive": "偏正面", "negative": "偏负面",
                    "neutral": "中性"}.get(label, label)
        lines.append(
            f"  正面 {senti.get('n_pos', 0)} / 负面 {senti.get('n_neg', 0)} / "
            f"中性 {senti.get('n_neu', 0)}，综合 {senti.get('overall_score', 0):+.3f}"
            f"（{label_cn}）；情感动量 {senti.get('momentum', 0):+.3f}。"
        )
        rev = senti.get("reversal_signal", 0.0)
        if rev > 0:
            lines.append("  ⚡ 触发情感触底反转信号。")
        elif rev < 0:
            lines.append("  ⚠ 触发情感顶部反转信号。")

    # ⑦ 因子暴露
    ff = ctx.get("ff")
    if ff and ff.get("exposures"):
        lines.append("")
        lines.append("◆ 因子暴露")
        expo = ff.get("exposures", {})
        parts = [f"{k} {v:+.2f}" for k, v in expo.items() if abs(v) > 0.05]
        if parts:
            lines.append("  " + "  ".join(parts))
        lines.append(f"  年化alpha {ff.get('annual_alpha', 0)*100:+.2f}%，"
                     f"R²={ff.get('r_squared', 0):.2f}，"
                     f"主导风格：{ff.get('dominant_style', '无明确暴露')}。")

    lines.append("")
    lines.append("─" * 56)
    lines.append("⚠ 基于历史数据统计与概率模型生成，不构成投资建议。")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# LLM 润色线程
# ══════════════════════════════════════════════════════════════════
class AICommentaryThread(QThread):
    result_ready = pyqtSignal(str)
    error_msg = pyqtSignal(str)

    _SYSTEM_PROMPT = (
        "你是一名谨慎的证券分析助理。用户会给你一份关于某只股票的结构化数据"
        "摘要（价格、概率预测、技术指标、量价结构、尾部风险、新闻情感、因子"
        "暴露等）。请用中文写一段 250-400 字的点评：\n"
        "1. 先用一句话总结当前的多空格局；\n"
        "2. 分维度解读：技术面 / 概率面 / 量价 / 风险 / 情感；\n"
        "3. 指出 2-3 个最值得关注的风险点或矛盾信号；\n"
        "4. 不要给出具体买卖点位、仓位大小或收益承诺；\n"
        "5. 结尾提醒这是基于历史数据的概率参考，不构成投资建议。\n"
        "语气克制、专业、有信息密度，避免空话套话。"
    )

    def __init__(self, base_url: str, api_key: str, model: str, context_text: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.context_text = context_text

    def run(self):
        try:
            import requests
        except ImportError:
            self.error_msg.emit("未安装 requests"); return
        if not self.base_url or not self.api_key:
            self.error_msg.emit("未配置 API"); return
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": self.model or "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": self._SYSTEM_PROMPT},
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
                self.error_msg.emit("返回空内容"); return
            self.result_ready.emit(text)
        except Exception as e:
            self.error_msg.emit(str(e))


def _build_llm_context_text(ctx: Dict) -> str:
    """【v1.1 修复】情感缺失时显式说明，避免 LLM 瞎猜。"""
    lines = [f"标的：{ctx.get('symbol')}　名称：{ctx.get('name') or '--'}"]
    if ctx.get("price") is not None:
        lines.append(f"现价：{ctx['currency']}{ctx['price']:.2f}　"
                     f"当日涨跌：{_pct(ctx.get('change_pct'))}")
    for tag, label in (("prob_5d", "5日上涨概率"),
                       ("prob_20d", "20日上涨概率"),
                       ("prob_60d", "60日上涨概率")):
        if ctx.get(tag) is not None:
            lines.append(f"{label}：{ctx[tag]*100:.0f}%")
    if ctx.get("signal_text"):
        lines.append(f"扫描信号：{ctx['signal_text']}")
    if ctx.get("regime"):
        lines.append(f"市场环境：{ctx['regime']}")

    mc = ctx.get("mc_20d")
    if mc:
        lines.append(f"20日MC价格带：P5 {_pct(mc.get('low_pct'))} ~ "
                     f"P95 {_pct(mc.get('high_pct'))}，"
                     f"中位数 {_pct(mc.get('median_pct'))}")

    for tag, label in (("rsi", "RSI"), ("ma20", "MA20"), ("ma60", "MA60"),
                       ("ma200", "MA200"), ("atr", "ATR")):
        if ctx.get(tag) is not None:
            lines.append(f"{label}：{ctx[tag]:.2f}")

    if ctx.get("volume_price"):
        lines.append("量价结构：")
        for k, v in ctx["volume_price"].items():
            lines.append(f"  {k}: {v}")

    if ctx.get("evt"):
        lines.append("尾部风险：")
        for k, v in ctx["evt"].items():
            lines.append(f"  {k}: {v}")

    # 【v1.1 修复】显式告知 LLM 情感状态
    if ctx.get("sentiment"):
        lines.append("新闻情感（有数据）：")
        for k, v in ctx["sentiment"].items():
            if k != "compounds":
                lines.append(f"  {k}: {v}")
    else:
        lines.append("新闻情感：（本次未取到情感数据，请勿编造或过度解读情感面，"
                     "直接略过这一维度）")

    if ctx.get("ff") and ctx["ff"].get("exposures"):
        lines.append("FF因子暴露：")
        for k, v in ctx["ff"]["exposures"].items():
            lines.append(f"  {k}: {v}")
        lines.append(f"  年化alpha: {ctx['ff'].get('annual_alpha')}")
        lines.append(f"  R²: {ctx['ff'].get('r_squared')}")

    if ctx.get("micro"):
        lines.append("微观结构：")
        for k, v in ctx["micro"].items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# 独立弹窗
# ══════════════════════════════════════════════════════════════════
class AIAnalystDialog(QDialog):
    def __init__(self, env: dict, sym: str, row: Optional[Dict] = None,
                 dialog=None, parent=None):
        super().__init__(parent)
        self.env = env
        T = env["T"]
        self.setWindowTitle(f"AI 分析师 — {sym}")
        try:
            self.setWindowIcon(env["_make_app_icon"]())
        except Exception:
            pass
        try:
            f = self.windowFlags()
            f |= Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint
            f &= ~Qt.WindowContextHelpButtonHint
            self.setWindowFlags(f)
        except Exception:
            pass
        self.setStyleSheet(env.get("GLOBAL_STYLE", ""))
        self.resize(820, 680)

        self._ctx: Dict = {}
        self._llm_thread: Optional[AICommentaryThread] = None
        self._dialog_ref = dialog  # 保留原dialog引用（用于重取情感）

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        hdr = QLabel(f"<b>AI 分析师报告</b>　·　{sym}")
        hdr.setStyleSheet(f"color:{T.GOLD};font-size:11pt;border:none;")
        lay.addWidget(hdr)

        btn_row = QHBoxLayout()
        gen_btn = QPushButton("重新生成")
        gen_btn.setStyleSheet(f"background:{T.ACCENT};color:white;")
        gen_btn.clicked.connect(self._reload)
        llm_btn = QPushButton("AI 大模型润色")
        llm_btn.setStyleSheet(f"color:{T.PURPLE if hasattr(T,'PURPLE') else T.GOLD};"
                              f"border-color:{T.PURPLE if hasattr(T,'PURPLE') else T.GOLD};")
        llm_btn.clicked.connect(self._run_llm)
        self._llm_btn = llm_btn
        copy_btn = QPushButton("复制")
        copy_btn.clicked.connect(self._copy)
        cfg_btn = QPushButton("配置AI")
        cfg_btn.clicked.connect(self._open_settings)
        for b in (gen_btn, llm_btn, cfg_btn, copy_btn):
            btn_row.addWidget(b)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.status = QLabel("生成中…")
        self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
        lay.addWidget(self.status)

        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setStyleSheet(
            f"QTextBrowser{{background:{T.BG2};border:1px solid {T.BORDER};"
            f"color:{T.TEXT_1};font-size:10pt;padding:12px;border-radius:8px;"
            f"font-family:'Consolas','Courier New','Microsoft YaHei',monospace;}}"
        )
        lay.addWidget(self.text, 1)

        QTimer.singleShot(80, self._reload)

    def _reload(self):
        self.status.setText("分析中…")
        QApplication.processEvents()
        try:
            self._ctx = collect_context(
                self.env, self._sym_from_title(),
                dialog=self._dialog_ref, row=None)
        except Exception as e:
            logger.warning(f"[ai_commentary] collect_context 失败: {e}")
            self._ctx = {"symbol": self._sym_from_title()}
        try:
            report = build_commentary(self._ctx)
            self.text.setPlainText(report)
            has_senti = "✓" if self._ctx.get("sentiment") else "×"
            self.status.setText(
                f"已生成（情感数据：{has_senti}，共 {len(report)} 字符）")
        except Exception as e:
            self.text.setPlainText(f"生成失败: {e}")
            self.status.setText("失败")

    def _sym_from_title(self) -> str:
        return self.windowTitle().replace("AI 分析师 —", "").strip()

    def _run_llm(self):
        if self._llm_thread is not None and self._llm_thread.isRunning():
            return
        cfg = _get_llm_config()
        if not cfg["base_url"] or not cfg["api_key"]:
            QMessageBox.information(
                self, "未配置",
                "先点「配置AI」填好接口地址和 API Key。\n"
                "（离线模板已可用，AI润色只是让它读起来更像人写的）")
            return

        # 【v1.1 修复】润色前重新收集上下文（关键！）
        try:
            fresh = collect_context(self.env, self._sym_from_title(),
                                    dialog=self._dialog_ref, row=None)
            # 合并：新数据覆盖旧数据
            for k, v in fresh.items():
                if v is not None:
                    self._ctx[k] = v
            # 立即刷新显示（让用户看到情感已补上）
            self.text.setPlainText(build_commentary(self._ctx))
        except Exception as e:
            logger.warning(f"[ai_commentary] 润色前重收集失败: {e}")

        ctx_text = _build_llm_context_text(self._ctx)
        self.status.setText("AI 润色中…")
        self._llm_btn.setEnabled(False)
        th = AICommentaryThread(cfg["base_url"], cfg["api_key"],
                                cfg["model"], ctx_text)
        th.result_ready.connect(self._on_llm_result)
        th.error_msg.connect(self._on_llm_error)
        self._llm_thread = th
        th.start()

    def _on_llm_result(self, text: str):
        cur = self.text.toPlainText()
        self.text.setPlainText(
            cur + "\n\n" + "═" * 56 + "\n【AI 大模型点评】\n" + text +
            "\n\n⚠ 以上为大模型生成的文字点评，不构成投资建议。")
        self.status.setText("AI 润色完成")
        self._llm_btn.setEnabled(True)
        self._llm_thread = None

    def _on_llm_error(self, msg: str):
        self.status.setText(f"AI 调用失败：{msg}（离线报告不受影响）")
        self._llm_btn.setEnabled(True)
        self._llm_thread = None

    def _copy(self):
        t = self.text.toPlainText()
        if t:
            QApplication.clipboard().setText(t)
            self.status.setText("已复制到剪贴板")

    def _open_settings(self):
        try:
            from quantpro_paper_watchlist import LLMSettingsDialog
            LLMSettingsDialog(parent=self).exec_()
        except ImportError:
            QMessageBox.information(
                self, "提示", "未找到 AI 配置界面。")


def show_ai_report(env: dict, sym: str, row: Optional[Dict] = None,
                   dialog=None, parent=None):
    dlg = AIAnalystDialog(env, sym, row=row, dialog=dialog, parent=parent)
    dlg.exec_()


# ══════════════════════════════════════════════════════════════════
# 给概率弹窗注入「AI点评」tab
# ══════════════════════════════════════════════════════════════════
def _install_ai_tab(ProbabilityDialog, env: dict):
    _orig_build_ui = ProbabilityDialog._build_ui

    def _build_ui_with_ai(self):
        _orig_build_ui(self)
        try:
            _build_ai_tab(self, env)
        except Exception as e:
            logger.warning(f"[ai_commentary] 添加AI点评tab失败: {e}")

    ProbabilityDialog._build_ui = _build_ui_with_ai

    _orig_run = ProbabilityDialog._run

    def _run_with_ai(self):
        _orig_run(self)
        try:
            # 【v1.1 修复】延迟 1500ms → 2500ms，给新闻加载留时间
            QTimer.singleShot(2500, lambda: _refresh_ai_tab(self, env))
        except Exception as e:
            logger.warning(f"[ai_commentary] AI tab 刷新调度失败: {e}")

    ProbabilityDialog._run = _run_with_ai


def _build_ai_tab(self, env: dict):
    T = env["T"]
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(8, 8, 8, 8)
    lay.setSpacing(8)

    note = QLabel(
        "把本次分析得到的数字（概率/Regime/量价/尾部风险/情感/因子暴露）"
        "自动拼成一份中文研报。\n"
        "离线模板即时可出；若配置了 OpenAI 兼容 API，可点「AI 润色」。")
    note.setWordWrap(True)
    note.setStyleSheet(
        f"color:{T.TEXT_2};font-size:8.5pt;background:{T.BG2};"
        f"border:1px solid {T.BORDER};border-radius:6px;padding:8px;")
    lay.addWidget(note)

    btn_row = QHBoxLayout()
    regen_btn = QPushButton("重新生成")
    regen_btn.setStyleSheet(f"background:{T.ACCENT};color:white;")
    regen_btn.clicked.connect(lambda: _refresh_ai_tab(self, env))
    llm_btn = QPushButton("AI 润色")
    llm_btn.setStyleSheet(f"color:{T.PURPLE if hasattr(T,'PURPLE') else T.GOLD};"
                          f"border-color:{T.PURPLE if hasattr(T,'PURPLE') else T.GOLD};")
    llm_btn.clicked.connect(lambda: _run_llm_in_tab(self, env))
    copy_btn = QPushButton("复制")
    copy_btn.clicked.connect(lambda: _copy_ai_tab(self))
    cfg_btn = QPushButton("配置AI")
    cfg_btn.clicked.connect(lambda: _open_llm_settings(self))
    for b in (regen_btn, llm_btn, cfg_btn, copy_btn):
        btn_row.addWidget(b)
    btn_row.addStretch()
    lay.addLayout(btn_row)

    self._ai_status = QLabel("等待分析完成后生成…")
    self._ai_status.setStyleSheet(f"color:{T.TEXT_2};font-size:8.5pt;border:none;")
    lay.addWidget(self._ai_status)

    self._ai_text = QTextBrowser()
    self._ai_text.setStyleSheet(
        f"QTextBrowser{{background:{T.BG2};border:1px solid {T.BORDER};"
        f"color:{T.TEXT_1};font-size:9.5pt;padding:10px;border-radius:6px;"
        f"font-family:'Consolas','Courier New','Microsoft YaHei',monospace;}}"
    )
    lay.addWidget(self._ai_text, 1)

    self._ai_llm_thread = None
    self._ai_ctx_cache = None

    if hasattr(self, "_add_scroll_tab"):
        self._add_scroll_tab(w, "AI点评")
    else:
        self.tabs.addTab(w, "AI点评")


def _refresh_ai_tab(self, env: dict):
    if not hasattr(self, "_ai_text"):
        return
    self._ai_status.setText("收集上下文…")
    QApplication.processEvents()
    try:
        ctx = collect_context(env, self.sym, dialog=self)
        self._ai_ctx_cache = ctx
        report = build_commentary(ctx)
        self._ai_text.setPlainText(report)
        has_senti = "✓" if ctx.get("sentiment") else "×"
        n_fields = sum(1 for k, v in ctx.items() if v is not None and k != "symbol")
        self._ai_status.setText(
            f"已生成（收集到 {n_fields} 类数据，情感 {has_senti}）")
    except Exception as e:
        logger.warning(f"[ai_commentary] 生成失败: {e}")
        self._ai_text.setPlainText(f"生成失败: {e}")
        self._ai_status.setText("失败")


def _run_llm_in_tab(self, env: dict):
    """【v1.1 修复】润色前重新收集上下文。"""
    if getattr(self, "_ai_llm_thread", None) is not None \
            and self._ai_llm_thread.isRunning():
        return
    cfg = _get_llm_config()
    if not cfg["base_url"] or not cfg["api_key"]:
        QMessageBox.information(
            self, "未配置",
            "先点「配置AI」填好接口地址和 API Key。\n"
            "（离线模板已可用，AI润色只是让它读起来更像人写的）")
        return

    # 关键修复：不管之前有没有缓存，润色前都重新收集一次
    try:
        fresh = collect_context(env, self.sym, dialog=self)
        self._ai_ctx_cache = fresh
        # 同步刷新显示
        self._ai_text.setPlainText(build_commentary(fresh))
        has_senti = "✓" if fresh.get("sentiment") else "×"
        self._ai_status.setText(f"已刷新上下文（情感 {has_senti}），润色中…")
    except Exception as e:
        logger.warning(f"[ai_commentary] 润色前重收集失败: {e}")
        if getattr(self, "_ai_ctx_cache", None) is None:
            QMessageBox.information(self, "提示", "先点「重新生成」")
            return
        self._ai_status.setText("AI 润色中…")

    ctx_text = _build_llm_context_text(self._ai_ctx_cache)
    th = AICommentaryThread(cfg["base_url"], cfg["api_key"],
                            cfg["model"], ctx_text)
    th.result_ready.connect(lambda text: _on_ai_tab_llm_result(self, text, env))
    th.error_msg.connect(lambda msg: _on_ai_tab_llm_error(self, msg))
    self._ai_llm_thread = th
    th.start()


def _on_ai_tab_llm_result(self, text: str, env: dict):
    cur = self._ai_text.toPlainText()
    self._ai_text.setPlainText(
        cur + "\n\n" + "═" * 56 + "\n【AI 大模型点评】\n" + text +
        "\n\n⚠ 以上为大模型生成的文字点评，不构成投资建议。")
    self._ai_status.setText("AI 润色完成")
    self._ai_llm_thread = None


def _on_ai_tab_llm_error(self, msg: str):
    self._ai_status.setText(f"AI 调用失败：{msg}")
    self._ai_llm_thread = None


def _copy_ai_tab(self):
    if hasattr(self, "_ai_text"):
        t = self._ai_text.toPlainText()
        if t:
            QApplication.clipboard().setText(t)
            self._ai_status.setText("已复制到剪贴板")


def _open_llm_settings(self):
    try:
        from quantpro_paper_watchlist import LLMSettingsDialog
        LLMSettingsDialog(parent=self).exec_()
    except ImportError:
        QMessageBox.information(self, "提示", "未找到 AI 配置界面。")


# ══════════════════════════════════════════════════════════════════
# 扫描列表右键菜单
# ══════════════════════════════════════════════════════════════════
def _install_scan_right_click(QuantApp, env: dict):
    _orig_init = QuantApp.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            self.table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.table.customContextMenuRequested.connect(
                lambda pos: _show_scan_menu(self, env, pos))
        except Exception as e:
            logger.warning(f"[ai_commentary] 右键菜单安装失败: {e}")

    QuantApp.__init__ = _new_init


def _show_scan_menu(app_self, env: dict, pos):
    try:
        idx = app_self.table.indexAt(pos)
        if not idx.isValid():
            return
        src = app_self.proxy.mapToSource(idx)
        row = app_self.scan_model.row_dict(src.row())
        if not row:
            return
        sym = row.get("col_code", "")
        if not sym:
            return
        menu = QMenu(app_self.table)
        act = QAction(f"生成 AI 研报（{sym}）", menu)
        act.triggered.connect(
            lambda: show_ai_report(env, sym, row=row, parent=app_self))
        menu.addAction(act)
        menu.exec_(app_self.table.viewport().mapToGlobal(pos))
    except Exception as e:
        logger.warning(f"[ai_commentary] 右键菜单: {e}")


# ══════════════════════════════════════════════════════════════════
# 一键安装
# ══════════════════════════════════════════════════════════════════
def install_ai_commentary(env: dict):
    required = ["ProbabilityDialog", "QuantApp", "fetch_stock_data",
                "get_realtime_price", "T", "_make_app_icon", "_currency"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_ai_commentary: 主程序缺少 {missing}")

    PD = env["ProbabilityDialog"]
    if getattr(PD, "_ai_commentary_installed", False):
        logger.info("[ai_commentary] 已安装（跳过重复注入）")
        return PD

    _install_ai_tab(PD, env)
    _install_scan_right_click(env["QuantApp"], env)

    # 【v1.1 修复】Hook _on_news：新闻加载完成后自动刷新 AI tab
    _orig_on_news = PD._on_news

    def _on_news_chain(self, items):
        _orig_on_news(self, items)
        # 新闻回来了，如果 AI tab 已存在，延迟 300ms 刷新一次
        # （让原 _on_news 里的情感聚合先完成）
        try:
            if hasattr(self, "_ai_text"):
                def _delayed():
                    try:
                        _refresh_ai_tab(self, env)
                    except Exception as e:
                        logger.debug(f"[ai_commentary] news hook 刷新: {e}")
                QTimer.singleShot(300, _delayed)
        except Exception as e:
            logger.debug(f"[ai_commentary] news hook 安装: {e}")

    PD._on_news = _on_news_chain

    PD._ai_commentary_installed = True
    env["show_ai_report"] = lambda sym, row=None, dialog=None, parent=None: \
        show_ai_report(env, sym, row, dialog, parent)

    logger.info("[ai_commentary v1.1] 已安装：AI 文字分析师"
                "（含新闻情感修复 + 自动刷新）")
    return PD


# ══════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("AI 文字分析师 v1.1 自检")
    print("=" * 64)

    # 1) 情感多路径提取
    class FakeDialog:
        pass

    d1 = FakeDialog()
    d1._sentiment_agg = {"n_total": 5, "n_pos": 3, "n_neg": 1, "n_neu": 1,
                         "overall_score": 0.15, "momentum": 0.05,
                         "reversal_signal": 0.0, "label": "positive"}
    s = _extract_sentiment_from_dialog(d1)
    assert s and s["n_total"] == 5, f"路径1失败: {s}"
    print("[1] 情感从 _sentiment_agg 提取 ✓")

    d2 = FakeDialog()
    d2._ai_last_sentiment = {"n_total": 3, "n_pos": 2, "n_neg": 0, "n_neu": 1}
    s = _extract_sentiment_from_dialog(d2)
    assert s and s["n_total"] == 3, f"路径2失败: {s}"
    print("[2] 情感从 _ai_last_sentiment 提取 ✓")

    d3 = FakeDialog()  # 全部为空
    s = _extract_sentiment_from_dialog(d3)
    assert s is None, "空情况应返回 None"
    print("[3] 无情感时返回 None ✓")

    d4 = FakeDialog()
    d4._sentiment_agg = {"n_total": 0, "n_pos": 0, "n_neg": 0, "n_neu": 0}
    s = _extract_sentiment_from_dialog(d4)
    assert s is None, "全零情感应视为无效"
    print("[4] 空情感视为无效 ✓")

    # 2) LLM 上下文含情感数据
    ctx = {
        "symbol": "AAPL", "name": "苹果", "currency": "$",
        "price": 182.35, "change_pct": 0.87,
        "prob_20d": 0.62,
        "sentiment": {"n_total": 9, "n_pos": 3, "n_neg": 1, "n_neu": 5,
                      "overall_score": 0.14, "momentum": 0.05,
                      "reversal_signal": 0.0, "label": "positive",
                      "compounds": [0.1, 0.2]},
    }
    text = _build_llm_context_text(ctx)
    assert "新闻情感（有数据）" in text, f"有数据时应标明: {text}"
    assert "n_pos: 3" in text
    assert "compounds" not in text, "compounds 列表不应传给LLM"
    print("[5] LLM 上下文含情感数据 ✓")

    # 3) 无情感时显式说明
    ctx2 = {"symbol": "XYZ", "currency": "$"}
    text2 = _build_llm_context_text(ctx2)
    assert "未取到情感数据" in text2, f"无数据时应显式说明: {text2}"
    print("[6] 无情感时显式说明 ✓")

    # 4) 研报模板包含情感章节
    report = build_commentary(ctx)
    assert "◆ 新闻情感" in report
    assert "偏正面" in report
    print("[7] 研报模板含情感章节 ✓")

    print("\n全部自检通过 ✓")