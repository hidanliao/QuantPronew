"""
QuantPro — AI 文字分析师 v1.0
══════════════════════════════════════════════════════════════════
把散落在主程序/各补丁里的数字（概率、Regime、量价、EVT尾部、
Monte Carlo区间、FF因子暴露、情感、微观结构）汇成一份中文研报。

设计原则：
  ① 离线优先：无任何 API 也能出报告（模板+规则拼接）
  ② LLM 可选：若用户配了 OpenAI 兼容接口（复用 paper_watchlist
     的配置 key），异步追加一段润色
  ③ 三处挂载：
       - 概率预测弹窗：新增「AI点评」tab
       - 扫描列表：右键 → 「生成AI研报」
       - 对外暴露 show_ai_report(env, sym, row) 供其它模块调用
  ④ 数字可信：所有结论都附带具体数值，无"我觉得""可能"

集成（quantpro_v1_6.py 的 __main__ 里，v15 之后、QuantApp 之前）：

    from quantpro_ai_commentary import install_ai_commentary
    install_ai_commentary(globals())

依赖：仅用主程序已有的 PyQt5 / matplotlib / pandas / numpy
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import threading
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextBrowser,
    QApplication, QMessageBox, QFrame, QWidget, QMenu, QAction
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 配置读取（复用 paper_watchlist 的存储，无则降级为只读空）
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
# 上下文收集：把各模块的数字汇总成一个 dict
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
    """
    收集一只股票的所有可用分析数据。优先从已打开的 dialog 里取，
    否则用 fetch_stock_data 现算关键指标。所有字段都可能是 None，
    模板层会做条件渲染。
    """
    ctx: Dict[str, Any] = {
        "symbol": sym,
        "name": "",
        "currency": "$",
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

    # ── 从对话框拿现成的分析结果 ──
    if dialog is not None:
        try:
            if not ctx["name"] and hasattr(dialog, "name"):
                ctx["name"] = dialog.name or ""
            # 概率卡片
            if hasattr(dialog, "prob_cards"):
                for key, tag in (("p5", "prob_5d"), ("p20", "prob_20d"), ("p60", "prob_60d")):
                    try:
                        txt = dialog.prob_cards[key].text().replace("%", "").strip()
                        if txt and txt != "--":
                            ctx[tag] = float(txt) / 100.0
                    except Exception:
                        pass
            # 情感
            if hasattr(dialog, "_sentiment_agg") and dialog._sentiment_agg:
                ctx["sentiment"] = dict(dialog._sentiment_agg)
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
            # 指标 DataFrame（用来算技术细节）
            if hasattr(dialog, "_ind") and dialog._ind is not None:
                ind = dialog._ind
                ctx["_ind_df"] = ind
        except Exception as e:
            logger.warning(f"[ai_commentary] 从 dialog 取数异常: {e}")

    # ── 用 _ind 补齐 RSI/MA 等 ──
    ind = ctx.pop("_ind_df", None)
    if ind is not None and not ind.empty:
        last = ind.iloc[-1]
        ctx.setdefault("rsi", _safe(last.get("rsi")))
        ctx.setdefault("ma20", _safe(last.get("ma20")))
        ctx.setdefault("ma60", _safe(last.get("ma60")))
        ctx.setdefault("ma200", _safe(last.get("ma200")))
        ctx.setdefault("atr", _safe(last.get("atr")))
        ctx.setdefault("price", _safe(last.get("close")))

    # ── 若还没有基础数据，现算 ──
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
                    last = ms.dropna().iloc[-1] if not ms.dropna().empty else None
                    if last is not None:
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
# 离线模板：把上下文拼成中文研报
# ══════════════════════════════════════════════════════════════════
def _pct(v, digits=2):
    if v is None:
        return "--"
    return f"{v:+.{digits}f}%"


def _num(v, digits=2):
    if v is None:
        return "--"
    return f"{v:.{digits}f}"


def _prob_pct(p):
    if p is None:
        return "--"
    return f"{p * 100:.0f}%"


def build_commentary(ctx: Dict) -> str:
    """纯函数：把上下文 dict 渲染成中文研报字符串（无 HTML）。"""
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

    # ── ① 综合结论 ──
    lines.append("")
    lines.append("◆ 综合结论")
    p20 = ctx.get("prob_20d")
    sig = ctx.get("signal_text")
    regime = ctx.get("regime")
    regime_cn = {"high_vol": "高波动", "low_vol": "低波动",
                 "neutral": "中性",
                 "calm_bull": "温和上行", "calm_bear": "温和下行",
                 "volatile": "震荡", "crisis": "危机"}.get(regime, regime or "未知")

    verdict_bits = []
    if p20 is not None:
        if p20 >= 0.62:
            verdict_bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（偏多）")
        elif p20 <= 0.38:
            verdict_bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（偏空）")
        else:
            verdict_bits.append(f"AI 20日上涨概率 {p20*100:.0f}%（中性）")
    if sig:
        verdict_bits.append(f"扫描信号：{sig}")
    if regime:
        verdict_bits.append(f"市场环境：{regime_cn}")

    if verdict_bits:
        lines.append("  " + "；".join(verdict_bits) + "。")
    else:
        lines.append("  数据不足，无法给出综合判断。")

    # ── ② 价格位置 ──
    lines.append("")
    lines.append("◆ 价格位置与技术")
    tech_bits = []
    ma20 = ctx.get("ma20")
    ma60 = ctx.get("ma60")
    ma200 = ctx.get("ma200")
    rsi = ctx.get("rsi")
    atr = ctx.get("atr")
    if price is not None and ma20:
        dev = (price - ma20) / ma20 * 100
        tech_bits.append(f"相对 MA20 {_pct(dev)}")
    if price is not None and ma60:
        dev = (price - ma60) / ma60 * 100
        tech_bits.append(f"相对 MA60 {_pct(dev)}")
    if price is not None and ma200:
        dev = (price - ma200) / ma200 * 100
        tech_bits.append(f"相对 MA200 {_pct(dev)}")
    if rsi is not None:
        if rsi >= 70:
            rsi_note = "超买"
        elif rsi <= 30:
            rsi_note = "超卖"
        elif rsi >= 55:
            rsi_note = "偏强"
        elif rsi <= 45:
            rsi_note = "偏弱"
        else:
            rsi_note = "中性"
        tech_bits.append(f"RSI={rsi:.1f}（{rsi_note}）")
    if atr is not None and price:
        atr_pct = atr / price * 100
        tech_bits.append(f"日均波动 ATR≈{cur}{atr:.2f}（{atr_pct:.2f}%）")
    if tech_bits:
        lines.append("  " + "；".join(tech_bits) + "。")
    else:
        lines.append("  技术指标数据不足。")

    # ── ③ 概率展望 ──
    lines.append("")
    lines.append("◆ 概率展望（AI集成：Purged CV + 校准）")
    p5 = ctx.get("prob_5d")
    p60 = ctx.get("prob_60d")
    prob_line = (f"  5日 上涨 {_prob_pct(p5)}　|　20日 上涨 {_prob_pct(p20)}"
                 f"　|　60日 上涨 {_prob_pct(p60)}")
    lines.append(prob_line)
    mc = ctx.get("mc_20d")
    if mc:
        lines.append(
            f"  20日蒙特卡洛价格带：P5 {_pct(mc.get('low_pct'))} ~ "
            f"P95 {_pct(mc.get('high_pct'))}，中位数 {_pct(mc.get('median_pct'))}"
        )

    # ── ④ 量价结构 ──
    vp = ctx.get("volume_price")
    if vp:
        lines.append("")
        lines.append("◆ 量价结构")
        vp_bits = []
        obv_note = vp.get("obv_vs_ma")
        if obv_note:
            vp_bits.append(f"OBV {obv_note}")
        rc = vp.get("roll_corr_60d")
        if rc is not None and np.isfinite(rc):
            rc_note = ("量价同向、健康趋势" if rc > 0.3
                       else "量价背离、警惕反转" if rc < -0.3
                       else "量价弱相关、震荡")
            vp_bits.append(f"60日量价相关 {rc:+.2f}（{rc_note}）")
        vi = vp.get("vol_imbalance_20d")
        if vi is not None and np.isfinite(vi):
            vi_note = "净流入" if vi > 0.1 else ("净流出" if vi < -0.1 else "均衡")
            vp_bits.append(f"20日资金 {vi:+.2f}（{vi_note}）")
        vw = vp.get("vwap_dev_pct")
        if vw is not None and np.isfinite(vw):
            vw_note = "高于均价" if vw > 0 else "低于均价"
            vp_bits.append(f"VWAP 偏离 {vw:+.2f}%（{vw_note}）")
        am = vp.get("amihud_illiq")
        if am is not None and np.isfinite(am):
            am_note = "流动性充裕" if am < 0.1 else ("流动性偏紧" if am > 0.5 else "流动性一般")
            vp_bits.append(f"Amihud={am:.3f}（{am_note}）")
        gk = vp.get("gk_vol_annual")
        if gk is not None and np.isfinite(gk):
            vp_bits.append(f"GK年化波动 {gk*100:.1f}%")
        nd = vp.get("n_divergence_90d", 0)
        if nd:
            vp_bits.append(f"近90日 {nd} 次背离信号")
        if vp_bits:
            lines.append("  " + "；".join(vp_bits) + "。")
        hint = vp.get("regime_hint")
        if hint:
            lines.append(f"  研判：{hint}。")

    # ── ⑤ 风险提示 ──
    lines.append("")
    lines.append("◆ 风险提示")
    evt = ctx.get("evt")
    risk_bits = []
    if evt:
        v99 = evt.get("var99")
        c99 = evt.get("cvar99")
        xi = evt.get("xi")
        if v99 is not None:
            risk_bits.append(f"EVT-VaR99 {v99*100:.2f}%")
        if c99 is not None:
            risk_bits.append(f"EVT-CVaR99 {c99*100:.2f}%")
        if xi is not None:
            xi_note = ("厚尾显著" if xi > 0.15
                       else "接近指数尾" if xi > -0.1
                       else "有界尾")
            risk_bits.append(f"尾部指数 ξ={xi:.3f}（{xi_note}）")
    micro = ctx.get("micro")
    if micro:
        rs = micro.get("roll_spread")
        kl = micro.get("kyle_lambda")
        if rs is not None and np.isfinite(rs):
            risk_bits.append(f"Roll 价差 {rs*100:.3f}%")
        if kl is not None and np.isfinite(kl):
            risk_bits.append(f"Kyle λ={kl:.2e}")
    if risk_bits:
        lines.append("  " + "；".join(risk_bits) + "。")
        if evt and evt.get("verdict"):
            lines.append(f"  研判：{evt['verdict']}。")
    else:
        lines.append("  尾部风险数据不足。")

    # ── ⑥ 情感面 ──
    senti = ctx.get("sentiment")
    if senti:
        lines.append("")
        lines.append("◆ 新闻情感")
        n_pos = senti.get("n_pos", 0)
        n_neg = senti.get("n_neg", 0)
        n_neu = senti.get("n_neu", 0)
        score = senti.get("overall_score", 0.0)
        mom = senti.get("momentum", 0.0)
        label = senti.get("label", "neutral")
        label_cn = {"positive": "偏正面", "negative": "偏负面", "neutral": "中性"}.get(label, label)
        rev = senti.get("reversal_signal", 0.0)
        lines.append(
            f"  正面 {n_pos} 条 / 负面 {n_neg} 条 / 中性 {n_neu} 条，"
            f"综合得分 {score:+.3f}（{label_cn}）；情感动量 {mom:+.3f}。"
        )
        if rev > 0:
            lines.append("  ⚡ 触发情感触底反转信号。")
        elif rev < 0:
            lines.append("  ⚠ 触发情感顶部反转信号。")

    # ── ⑦ 因子暴露 ──
    ff = ctx.get("ff")
    if ff and ff.get("exposures"):
        lines.append("")
        lines.append("◆ 因子暴露（Fama-French 5因子 + 动量）")
        expo = ff.get("exposures", {})
        parts = [f"{k} {v:+.2f}" for k, v in expo.items() if abs(v) > 0.05]
        if parts:
            lines.append("  " + "  ".join(parts))
        lines.append(
            f"  年化 alpha {ff.get('annual_alpha', 0)*100:+.2f}%，"
            f"R²={ff.get('r_squared', 0):.2f}，"
            f"主导风格：{ff.get('dominant_style', '无明确暴露')}。"
        )

    # ── 免责 ──
    lines.append("")
    lines.append("─" * 56)
    lines.append("⚠ 以上基于历史数据统计与概率模型生成，不构成投资建议。")
    lines.append("   模型不保证未来表现，请结合基本面和自身风险承受能力独立判断。")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# LLM 润色线程
# ══════════════════════════════════════════════════════════════════
class AICommentaryThread(QThread):
    result_ready = pyqtSignal(str)
    error_msg = pyqtSignal(str)

    _SYSTEM_PROMPT = (
        "你是一名谨慎的证券分析助理。用户会给你一份关于某只股票的"
        "结构化数据摘要（价格、概率预测、技术指标、量价结构、尾部风险、"
        "新闻情感、因子暴露等）。请用中文写一段 250-400 字的点评：\n"
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
    """把上下文 dict 转成给 LLM 看的紧凑文本（不带模板里的装饰）。"""
    lines = [f"标的：{ctx.get('symbol')}　名称：{ctx.get('name') or '--'}"]
    if ctx.get("price") is not None:
        lines.append(f"现价：{ctx['currency']}{ctx['price']:.2f}　"
                     f"当日涨跌：{_pct(ctx.get('change_pct'))}")
    for tag, label in (("prob_5d", "5日上涨概率"), ("prob_20d", "20日上涨概率"),
                       ("prob_60d", "60日上涨概率")):
        if ctx.get(tag) is not None:
            lines.append(f"{label}：{ctx[tag]*100:.0f}%")
    if ctx.get("signal_text"):
        lines.append(f"扫描信号：{ctx['signal_text']}")
    if ctx.get("regime"):
        lines.append(f"市场环境：{ctx['regime']}")
    mc = ctx.get("mc_20d")
    if mc:
        lines.append(
            f"20日MC价格带：P5 {_pct(mc.get('low_pct'))} ~ "
            f"P95 {_pct(mc.get('high_pct'))}，中位数 {_pct(mc.get('median_pct'))}")
    for tag, label in (("rsi", "RSI"), ("ma20", "MA20"), ("ma60", "MA60"),
                       ("ma200", "MA200"), ("atr", "ATR")):
        if ctx.get(tag) is not None:
            lines.append(f"{label}：{ctx[tag]:.2f}")
    vp = ctx.get("volume_price")
    if vp:
        lines.append("量价结构：")
        for k, v in vp.items():
            lines.append(f"  {k}: {v}")
    evt = ctx.get("evt")
    if evt:
        lines.append("尾部风险：")
        for k, v in evt.items():
            lines.append(f"  {k}: {v}")
    if ctx.get("sentiment"):
        lines.append("新闻情感：")
        for k, v in ctx["sentiment"].items():
            if k != "compounds":
                lines.append(f"  {k}: {v}")
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
        btn_row.addWidget(gen_btn)
        btn_row.addWidget(llm_btn)
        btn_row.addWidget(cfg_btn)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self.status = QLabel("生成中…")
        self.status.setStyleSheet(f"color:{T.TEXT_2};font-size:9pt;border:none;")
        lay.addWidget(self.status)

        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        self.text.setStyleSheet(
            f"QTextBrowser{{background:{T.BG2};border:1px solid {T.BORDER};"
            f"color:{T.TEXT_1};font-size:10pt;padding:12px;"
            f"border-radius:8px;font-family:'Consolas','Courier New','Microsoft YaHei',monospace;}}"
        )
        lay.addWidget(self.text, 1)

        QTimer.singleShot(80, self._reload)

    def _reload(self):
        self.status.setText("分析中…")
        QApplication.processEvents()
        try:
            self._ctx = collect_context(self.env, self._sym_from_title(), row=None,
                                        dialog=None)
        except Exception as e:
            logger.warning(f"[ai_commentary] collect_context 失败: {e}")
            self._ctx = {"symbol": self._sym_from_title()}
        try:
            report = build_commentary(self._ctx)
            self.text.setPlainText(report)
            self.status.setText(f"已生成（离线模板，共 {len(report)} 字符）")
        except Exception as e:
            self.text.setPlainText(f"生成失败: {e}")
            self.status.setText("失败")

    def _sym_from_title(self) -> str:
        t = self.windowTitle()
        return t.replace("AI 分析师 —", "").strip()

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
        ctx_text = _build_llm_context_text(self._ctx)
        self.status.setText("AI 润色中…（网络请求，几秒到十几秒）")
        self._llm_btn.setEnabled(False)
        th = AICommentaryThread(cfg["base_url"], cfg["api_key"],
                                cfg["model"], ctx_text)
        th.result_ready.connect(self._on_llm_result)
        th.error_msg.connect(self._on_llm_error)
        self._llm_thread = th
        th.start()

    def _on_llm_result(self, text: str):
        T = self.env["T"]
        cur = self.text.toPlainText()
        self.text.setPlainText(
            cur + "\n\n" + "═" * 56 + "\n【AI 大模型点评】\n" + text +
            "\n\n⚠ 以上为大模型生成的文字点评，同样基于上面的历史数据统计，"
            "不构成投资建议。"
        )
        self.status.setText("AI 润色完成")
        self._llm_btn.setEnabled(True)
        self._llm_thread = None

    def _on_llm_error(self, msg: str):
        self.status.setText(f"AI 调用失败：{msg}（离线报告不受影响）")
        self._llm_btn.setEnabled(True)
        self._llm_thread = None

    def _copy(self):
        text = self.text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.status.setText("已复制到剪贴板")

    def _open_settings(self):
        try:
            from quantpro_paper_watchlist import LLMSettingsDialog
            LLMSettingsDialog(parent=self).exec_()
        except ImportError:
            QMessageBox.information(
                self, "提示",
                "未找到 AI 配置界面（需先安装 quantpro_paper_watchlist.py）。")


def show_ai_report(env: dict, sym: str, row: Optional[Dict] = None,
                   dialog=None, parent=None):
    """对外接口：弹出一个 AI 分析师窗口。"""
    dlg = AIAnalystDialog(env, sym, row=row, dialog=dialog, parent=parent)
    dlg.exec_()


# ══════════════════════════════════════════════════════════════════
# 给概率弹窗注入「AI点评」tab
# ══════════════════════════════════════════════════════════════════
def _install_ai_tab(ProbabilityDialog, env: dict):
    """给 ProbabilityDialog 追加「AI点评」tab。"""

    _orig_build_ui = ProbabilityDialog._build_ui

    def _build_ui_with_ai(self):
        _orig_build_ui(self)
        try:
            _build_ai_tab(self, env)
        except Exception as e:
            logger.warning(f"[ai_commentary] 添加AI点评tab失败: {e}")

    ProbabilityDialog._build_ui = _build_ui_with_ai

    # 原 _run 跑完后，刷新 AI tab
    _orig_run = ProbabilityDialog._run

    def _run_with_ai(self):
        _orig_run(self)
        try:
            QTimer.singleShot(1500, lambda: _refresh_ai_tab(self, env))
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
        "离线模板即时可出；若配置了 OpenAI 兼容 API，可点「AI 润色」让它"
        "读起来更自然。")
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
    btn_row.addWidget(regen_btn)
    btn_row.addWidget(llm_btn)
    btn_row.addWidget(cfg_btn)
    btn_row.addWidget(copy_btn)
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
        n_fields = sum(1 for k, v in ctx.items() if v is not None and k != "symbol")
        self._ai_status.setText(f"已生成（离线模板，收集到 {n_fields} 类数据）")
    except Exception as e:
        logger.warning(f"[ai_commentary] 生成失败: {e}")
        self._ai_text.setPlainText(f"生成失败: {e}")
        self._ai_status.setText("失败")


def _run_llm_in_tab(self, env: dict):
    if getattr(self, "_ai_llm_thread", None) is not None \
            and self._ai_llm_thread.isRunning():
        return
    if getattr(self, "_ai_ctx_cache", None) is None:
        QMessageBox.information(self, "提示", "先点「重新生成」")
        return
    cfg = _get_llm_config()
    if not cfg["base_url"] or not cfg["api_key"]:
        QMessageBox.information(
            self, "未配置",
            "先点「配置AI」填好接口地址和 API Key。\n"
            "（离线模板已可用，AI润色只是让它读起来更像人写的）")
        return
    ctx_text = _build_llm_context_text(self._ai_ctx_cache)
    self._ai_status.setText("AI 润色中…")
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
        "\n\n⚠ 以上为大模型生成的文字点评，不构成投资建议。"
    )
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
        QMessageBox.information(
            self, "提示",
            "未找到 AI 配置界面（需先安装 quantpro_paper_watchlist.py）。")


# ══════════════════════════════════════════════════════════════════
# 给扫描列表加右键菜单
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
            logger.warning(f"[ai_commentary] 安装扫描列表右键菜单失败: {e}")

    QuantApp.__init__ = _new_init


def _show_scan_menu(app_self, env: dict, pos):
    try:
        idx = app_self.table.indexAt(pos)
        if not idx.isValid():
            return
        # 通过 proxy 映射到源
        src = app_self.proxy.mapToSource(idx)
        row = app_self.scan_model.row_dict(src.row())
        if not row:
            return
        sym = row.get("col_code", "")
        if not sym:
            return
        menu = QMenu(app_self.table)
        act_report = QAction(f"生成 AI 研报（{sym}）", menu)
        act_report.triggered.connect(
            lambda: show_ai_report(env, sym, row=row, parent=app_self))
        menu.addAction(act_report)

        act_llm = QAction("AI 研报 + 大模型润色", menu)
        act_llm.triggered.connect(
            lambda: _show_report_with_llm(env, sym, row, app_self))
        menu.addAction(act_llm)

        menu.exec_(app_self.table.viewport().mapToGlobal(pos))
    except Exception as e:
        logger.warning(f"[ai_commentary] 右键菜单: {e}")


def _show_report_with_llm(env: dict, sym: str, row: dict, parent):
    dlg = AIAnalystDialog(env, sym, row=row, parent=parent)
    QTimer.singleShot(500, dlg._run_llm)
    dlg.exec_()


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

    PD._ai_commentary_installed = True

    # 把便捷入口挂到 env，方便其他模块调用
    env["show_ai_report"] = lambda sym, row=None, dialog=None, parent=None: \
        show_ai_report(env, sym, row, dialog, parent)

    logger.info("[ai_commentary] 已安装：AI 文字分析师"
                "（概率弹窗新增「AI点评」tab + 扫描列表右键菜单）")
    return PD


# ══════════════════════════════════════════════════════════════════
# 自检（合成数据，无需联网/无需主程序）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 64)
    print("AI 文字分析师 自检")
    print("=" * 64)

    # 构造一个信息丰富的上下文
    ctx = {
        "symbol": "AAPL", "name": "苹果", "currency": "$",
        "price": 182.35, "change_pct": 0.87,
        "rsi": 58.3, "ma20": 180.12, "ma60": 177.45, "ma200": 170.10,
        "atr": 3.28,
        "prob_5d": 0.58, "prob_20d": 0.62, "prob_60d": 0.55,
        "signal_text": "买入",
        "regime": "low_vol",
        "mc_20d": {"low_pct": -8.2, "median_pct": 1.2,
                   "high_pct": 9.5, "up_prob": 0.62},
        "volume_price": {
            "obv_vs_ma": "↑强势",
            "roll_corr_60d": 0.35,
            "vol_imbalance_20d": 0.18,
            "vwap_dev_pct": 1.5,
            "amihud_illiq": 0.042,
            "gk_vol_annual": 0.223,
            "n_divergence_90d": 2,
            "regime_hint": "量价同向（健康趋势）",
        },
        "evt": {"var99": -0.052, "cvar99": -0.071, "var95": -0.031,
                "cvar95": -0.045, "xi": 0.18,
                "verdict": "厚尾显著，正态假设会低估风险"},
        "micro": {"roll_spread": 0.0015, "kyle_lambda": 2.1e-7},
        "sentiment": {"n_pos": 3, "n_neg": 1, "n_neu": 5,
                      "overall_score": 0.14, "momentum": 0.05,
                      "reversal_signal": 0.0, "label": "positive"},
        "ff": {"exposures": {"MKT": 0.92, "SMB": -0.15, "HML": 0.08,
                             "RMW": 0.30, "CMA": -0.05, "MOM": 0.25},
               "annual_alpha": 0.032, "r_squared": 0.78,
               "dominant_style": "市场暴露为主"},
    }

    report = build_commentary(ctx)
    print(report)
    print()

    # 关键字段检查
    assert "AAPL" in report
    assert "62%" in report       # 概率
    assert "低波动" in report     # regime
    assert "厚尾" in report       # EVT
    assert "量价" in report
    assert "苹果" in report
    print("[✓] 完整数据渲染成功")

    # 空数据降级
    empty = {"symbol": "XYZ", "currency": "$"}
    r2 = build_commentary(empty)
    assert "XYZ" in r2
    assert "数据不足" in r2
    print("[✓] 空数据降级成功")

    # 部分数据
    partial = {"symbol": "GOOG", "currency": "$",
               "price": 140.0, "rsi": 72.0,
               "prob_20d": 0.45}
    r3 = build_commentary(partial)
    assert "GOOG" in r3
    assert "超买" in r3
    print("[✓] 部分数据渲染成功")

    # LLM context 文本
    ctx_text = _build_llm_context_text(ctx)
    assert "AAPL" in ctx_text
    assert "5日上涨概率" in ctx_text
    print("[✓] LLM 上下文文本生成成功")

    print()
    print("全部自检通过 ✓")
    print()
    print("集成方式（quantpro_v1_6.py 的 __main__ 里，v15 之后）：")
    print("    from quantpro_ai_commentary import install_ai_commentary")
    print("    install_ai_commentary(globals())")