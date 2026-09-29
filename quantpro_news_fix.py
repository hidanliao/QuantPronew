"""
QuantPro — 新闻模块健壮化补丁 v1.1
══════════════════════════════════════════════════════════════════
问题：
  ① yfinance 近半年改了 news 返回结构（content 包装 / canonicalUrl
     可能是 None / news 被 {"news": [...]} 包一层），导致新闻Tab
     一片空白，日志里什么都看不到（异常被宽 except 吞掉）。
  ② 概率面板 status_lbl 里的情感调整字段会被重复追加，
     多次加载新闻后显示成 "prob_done  |  情感调整  |  情感调整 ..."

修复：
  ① NewsThread.run 用新 fetch_news_robust：三级数据源 + 字段安全提取
  ② 空结果时 QTextBrowser 显示排查方向卡片（不是白屏）
  ③ 修补 _refresh_prob_with_sentiment，把情感调整字段做成幂等替换
  ④ 全程打印 [news_fix] 前缀的 INFO/WARNING 日志，可追踪

集成（quantpro_v1_6.py 的 __main__ 里、创建 QuantApp 之前）：

    from quantpro_news_fix import install_news_fix
    install_news_fix(globals())

依赖：requests, yfinance（主程序已有）
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote as _urlquote

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# 工具：安全字段提取 / 时间归一化
# ══════════════════════════════════════════════════════════════════
def _safe_get(d: Any, *keys, default=None):
    """沿路径安全取 dict 里的值，任何一段是 None/缺失/类型异常都不炸。"""
    cur = d
    for k in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k, None)
        elif isinstance(cur, (list, tuple)) and isinstance(k, int):
            try:
                cur = cur[k]
            except (IndexError, TypeError):
                return default
        else:
            return default
    return default if cur is None else cur


def _normalize_time(ts_raw: Any) -> str:
    """把各种时间表示统一成 'YYYY-MM-DD HH:MM'。"""
    if ts_raw is None or ts_raw == "":
        return ""
    # Unix 时间戳（int / float）
    if isinstance(ts_raw, (int, float)):
        try:
            return datetime.fromtimestamp(int(ts_raw)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""
    if isinstance(ts_raw, str):
        s = ts_raw.strip()
        if not s:
            return ""
        # ISO 8601（含 Z 结尾）
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        # RFC 822（Google News RSS 格式）
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(s)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
        # 兜底：正则挖 YYYY-MM-DD
        m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
        return s[:16]
    return ""


def _extract_news_item(n: Any) -> Optional[Dict]:
    """
    从单条 yfinance news 记录里提取 {title, link, publisher, time}。
    兼容 3 种结构 + 字段为 None/类型异常。
    """
    if not isinstance(n, dict):
        return None

    # ── 结构 B：新版 content 包装 ──
    content = n.get("content")
    if isinstance(content, dict):
        title = _safe_get(content, "title") or ""
        if title:
            link = (_safe_get(content, "canonicalUrl", "url")
                    or _safe_get(content, "clickThroughUrl", "url")
                    or _safe_get(n, "link")
                    or "")
            prov = content.get("provider")
            if isinstance(prov, dict):
                publisher = prov.get("displayName") or prov.get("name") or ""
            elif isinstance(prov, str):
                publisher = prov
            else:
                publisher = _safe_get(n, "publisher") or ""
            ts_raw = content.get("pubDate") or content.get("displayTime") or ""
            return {"title": title, "link": link,
                    "publisher": publisher, "time": _normalize_time(ts_raw)}

    # ── 结构 A：旧版顶层字段 ──
    title = n.get("title") or ""
    if title:
        link = n.get("link") or n.get("url") or ""
        publisher = n.get("publisher") or ""
        ts_raw = n.get("providerPublishTime") or n.get("pubDate") or ""
        return {"title": title, "link": link,
                "publisher": publisher, "time": _normalize_time(ts_raw)}

    return None


# ══════════════════════════════════════════════════════════════════
# 数据源 1：yfinance
# ══════════════════════════════════════════════════════════════════
def _fetch_news_yfinance(sym: str, limit: int = 15) -> List[Dict]:
    try:
        tk = yf.Ticker(sym)
        raw = getattr(tk, "news", None)
    except Exception as e:
        logger.warning(f"[news_fix] yf.Ticker({sym}).news 抛异常: {e}")
        return []

    if raw is None:
        logger.info(f"[news_fix] yf news 返回 None（{sym}）")
        return []

    # 有时被包一层 {"news": [...]}
    if isinstance(raw, dict):
        raw = raw.get("news") or raw.get("items") or []

    if not isinstance(raw, (list, tuple)):
        logger.info(f"[news_fix] yf news 返回类型异常: {type(raw).__name__}")
        return []

    logger.info(f"[news_fix] yf news 返回 {len(raw)} 条原始记录（{sym}）")

    items: List[Dict] = []
    for n in raw[:limit]:
        try:
            parsed = _extract_news_item(n)
            if parsed and parsed.get("title"):
                items.append(parsed)
        except Exception as e:
            logger.warning(f"[news_fix] 解析单条 news 失败: {e}")

    logger.info(f"[news_fix] yf 解析后得到 {len(items)} 条有效新闻")
    return items


# ══════════════════════════════════════════════════════════════════
# 数据源 2：Yahoo Finance Query API（直接 HTTP）
# ══════════════════════════════════════════════════════════════════
_YF_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
_YF_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0 Safari/537.36"),
    "Accept": "application/json,text/plain,*/*",
}


def _fetch_news_yahoo_api(sym: str, limit: int = 15) -> List[Dict]:
    try:
        params = {
            "q": sym,
            "quotesCount": 0,
            "newsCount": limit,
            "enableFuzzyQuery": "false",
            "quotesQueryId": "tss_match_phrase_query",
        }
        r = requests.get(_YF_SEARCH_URL, params=params,
                         headers=_YF_HEADERS, timeout=8)
        if r.status_code != 200:
            logger.info(f"[news_fix] Yahoo API HTTP {r.status_code}")
            return []
        raw = (r.json() or {}).get("news") or []
        logger.info(f"[news_fix] Yahoo API 返回 {len(raw)} 条原始记录")
        items: List[Dict] = []
        for n in raw[:limit]:
            if not isinstance(n, dict):
                continue
            title = n.get("title") or ""
            if not title:
                continue
            items.append({
                "title": title,
                "link": n.get("link") or "",
                "publisher": n.get("publisher") or "",
                "time": _normalize_time(n.get("providerPublishTime")
                                        or n.get("pubDate") or ""),
            })
        logger.info(f"[news_fix] Yahoo API 解析后 {len(items)} 条有效新闻")
        return items
    except Exception as e:
        logger.warning(f"[news_fix] Yahoo API 调用失败: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# 数据源 3：Google News RSS（最后保底）
# ══════════════════════════════════════════════════════════════════
def _fetch_news_google_rss(sym: str, limit: int = 15) -> List[Dict]:
    try:
        import xml.etree.ElementTree as ET
        q = _urlquote(sym + " stock")
        url = (f"https://news.google.com/rss/search?"
               f"q={q}&hl=en-US&gl=US&ceid=US:en")
        r = requests.get(url, headers=_YF_HEADERS, timeout=8)
        if r.status_code != 200:
            logger.info(f"[news_fix] Google RSS HTTP {r.status_code}")
            return []
        root = ET.fromstring(r.content)
        items: List[Dict] = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            publisher = (source_el.text if source_el is not None
                         else "") or "Google News"
            if title:
                items.append({
                    "title": title, "link": link,
                    "publisher": publisher,
                    "time": _normalize_time(pub),
                })
            if len(items) >= limit:
                break
        logger.info(f"[news_fix] Google RSS 解析后 {len(items)} 条有效新闻")
        return items
    except Exception as e:
        logger.warning(f"[news_fix] Google RSS 失败: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# 统一入口：三级数据源串联
# ══════════════════════════════════════════════════════════════════
def fetch_news_robust(sym: str, limit: int = 15) -> List[Dict]:
    """依次 yfinance → Yahoo API → Google RSS，任一有结果即用。"""
    for name, fn in (("yfinance", _fetch_news_yfinance),
                     ("yahoo_api", _fetch_news_yahoo_api),
                     ("google_rss", _fetch_news_google_rss)):
        try:
            items = fn(sym, limit)
        except Exception as e:
            logger.warning(f"[news_fix] {name} 抛出: {e}")
            items = []
        if items:
            logger.info(f"[news_fix] {sym}: 使用 {name} 源，共 {len(items)} 条")
            return items
        logger.info(f"[news_fix] {sym}: {name} 源无结果，尝试下一个")
    logger.warning(f"[news_fix] {sym}: 所有新闻源均失败")
    return []


# ══════════════════════════════════════════════════════════════════
# 一键安装
# ══════════════════════════════════════════════════════════════════
def install_news_fix(env: dict):
    required = ["NewsThread", "ProbabilityDialog"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_news_fix: 主程序缺少 {missing}")

    NewsThread_cls = env["NewsThread"]
    PD = env["ProbabilityDialog"]
    sentiment_engine = env.get("NewsSentimentEngine")   # 可选

    if getattr(PD, "_news_fix_installed", False):
        logger.info("[news_fix] 已安装（跳过重复注入）")
        return PD

    # ── ① 替换 NewsThread.run ──
    def _news_thread_run(self):
        try:
            items = fetch_news_robust(self.sym, limit=15)
            if items and sentiment_engine is not None:
                try:
                    use_finbert = bool(getattr(self, "use_finbert", False))
                    items = sentiment_engine.score_news_batch(
                        items, use_finbert=use_finbert)
                except Exception as e:
                    logger.warning(f"[news_fix] 情感打分失败（不影响显示）: {e}")
            if not items:
                self.error_msg.emit(
                    "所有新闻源均未返回结果。可能原因："
                    "① 该代码本身无近期新闻；"
                    "② yfinance/Yahoo/Google 反爬升级；"
                    "③ 网络代理阻断。详见终端 [news_fix] 日志。")
                return
            self.news_ready.emit(items)
        except Exception as e:
            logger.error(f"[news_fix] NewsThread.run 异常: {e}", exc_info=True)
            self.error_msg.emit(str(e))

    NewsThread_cls.run = _news_thread_run

    # ── ② 包装 _on_news：空结果时给诊断卡片 ──
    _orig_on_news = PD._on_news

    def _on_news_with_diag(self, items):
        try:
            _orig_on_news(self, items)
        except Exception as e:
            logger.error(f"[news_fix] _on_news 原逻辑异常: {e}", exc_info=True)
            try:
                self.news_status.setText(f"渲染失败: {e}")
            except Exception:
                pass
            return

        if items:
            return  # 有内容就不用补诊断

        # 空结果：写诊断卡片，避免白屏
        try:
            T = env.get("T")
            text_color = getattr(T, "TEXT_2", "#6d7d70") if T else "#6d7d70"
            warn_color = getattr(T, "YELLOW", "#d97706") if T else "#d97706"
            html = (
                f"<div style='padding:16px;'>"
                f"<p style='color:{warn_color};font-size:11pt;font-weight:600;'>"
                f"⚠ 未获取到 {self.sym} 的新闻</p>"
                f"<p style='color:{text_color};font-size:9.5pt;line-height:1.7;'>"
                f"可能的排查方向：<br>"
                f"1. 该代码本身没有近期新闻（冷门股、ETF 常见）<br>"
                f"2. yfinance 的 news 接口返回结构变更（本补丁已加兼容）<br>"
                f"3. Yahoo / Google 反爬升级 —— 请检查终端日志里 "
                f"<code style='color:{warn_color};'>[news_fix]</code> "
                f"开头的 INFO/WARNING<br>"
                f"4. 网络代理/DNS 阻断 —— 浏览器能否打开 "
                f"<a href='https://query1.finance.yahoo.com/v1/finance/search?q=AAPL' "
                f"style='color:{warn_color};'>Yahoo Finance API</a>？"
                f"</p></div>"
            )
            self.news_text.setHtml(html)
        except Exception as e:
            logger.warning(f"[news_fix] 诊断渲染失败: {e}")

    PD._on_news = _on_news_with_diag

    # ── ③ 修补 _refresh_prob_with_sentiment 的情感字段幂等替换 ──
    _orig_refresh = PD._refresh_prob_with_sentiment

    def _refresh_with_sentiment_fixed(self):
        if self._ind is None or self._sentiment_agg is None:
            return
        try:
            engine = env["ProbabilityEngine"](self._ind, self._close)
            p5 = engine.ensemble(5, self._sentiment_agg)
            pct = int(p5 * 100)
            T = env["T"]
            col = T.GREEN if pct >= 60 else (T.RED if pct <= 40 else T.YELLOW)
            self.prob_cards["p5"].setText(f"{pct}%")
            self.prob_cards["p5"].setStyleSheet(
                f"color:{col};font-size:26pt;font-weight:700;border:none;")

            # 关键修复：把上次写进去的情感字段先剥掉，再拼新的
            sent_engine = env.get("NewsSentimentEngine")
            adj = sent_engine.sentiment_to_prob_adjustment(self._sentiment_agg) \
                if sent_engine else 0.0
            cur_text = self.status_lbl.text()
            # 用正则剥掉任意已有 "  |  情感调整:±X.X%" 片段
            base = re.sub(r"\s*\|\s*情感调整:[+\-]?\d+\.?\d*%", "", cur_text)
            self.status_lbl.setText(f"{base}  |  情感调整:{adj:+.1%}")
        except Exception as e:
            logger.warning(f"[news_fix] _refresh_prob_with_sentiment 异常: {e}")

    PD._refresh_prob_with_sentiment = _refresh_with_sentiment_fixed

    PD._news_fix_installed = True
    logger.info("[news_fix] 已安装：多源新闻抓取（yf/YahooAPI/GoogleRSS）"
                "+ 诊断卡片 + 情感字段幂等修复")
    return PD


# ══════════════════════════════════════════════════════════════════
# 自检（不联网，合成数据）
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("新闻模块健壮化补丁自检")
    print("=" * 60)

    # 1) 旧版 yfinance 结构
    old_fmt = {"title": "Apple beats earnings",
               "link": "https://example.com/a",
               "publisher": "Reuters",
               "providerPublishTime": 1735689600}
    p = _extract_news_item(old_fmt)
    assert p and p["title"] == "Apple beats earnings", f"旧版解析失败: {p}"
    assert p["publisher"] == "Reuters"
    assert "2025-01-01" in p["time"], f"时间戳转换错: {p['time']}"
    print("[1] 旧版结构解析 ✓")

    # 2) 新版 yfinance（content 包装）
    new_fmt = {"id": "abc-123",
               "content": {"title": "NVIDIA surges on AI demand",
                           "canonicalUrl": {"url": "https://example.com/b"},
                           "provider": {"displayName": "Bloomberg"},
                           "pubDate": "2025-01-15T13:30:00Z"}}
    p = _extract_news_item(new_fmt)
    assert p and p["title"] == "NVIDIA surges on AI demand", f"新版解析失败: {p}"
    assert p["link"] == "https://example.com/b"
    assert p["publisher"] == "Bloomberg"
    assert p["time"].startswith("2025-01-15"), f"ISO时间解析错: {p['time']}"
    print("[2] 新版结构解析 ✓")

    # 3) 脏数据：canonicalUrl=None, provider=str, pubDate=None
    dirty = {"content": {"title": "Tesla drops 5%",
                         "canonicalUrl": None,
                         "provider": "CNBC",
                         "pubDate": None}}
    p = _extract_news_item(dirty)
    assert p and p["title"] == "Tesla drops 5%", f"脏数据解析失败: {p}"
    assert p["publisher"] == "CNBC", f"provider str 兼容失败: {p}"
    assert p["link"] == "", "canonicalUrl=None 应返回空链接"
    print("[3] 脏数据字段兼容 ✓")

    # 4) 各种时间格式
    assert _normalize_time(1735689600).startswith("2025-01-01")
    assert _normalize_time("2025-02-03T10:00:00Z").startswith("2025-02-03")
    assert _normalize_time("Mon, 03 Feb 2025 10:00:00 GMT").startswith("2025-02-03")
    assert _normalize_time("") == ""
    assert _normalize_time(None) == ""
    assert _normalize_time(0) == "" or _normalize_time(0)  # 不报错即可
    print("[4] 时间格式归一化 ✓")

    # 5) 空/异常结构
    assert _extract_news_item(None) is None
    assert _extract_news_item({}) is None
    assert _extract_news_item({"content": {}}) is None
    assert _extract_news_item({"foo": "bar"}) is None
    assert _extract_news_item("not a dict") is None
    print("[5] 空/异常结构返回 None ✓")

    # 6) _safe_get 各种边界
    assert _safe_get({"a": {"b": {"c": 1}}}, "a", "b", "c") == 1
    assert _safe_get({"a": None}, "a", "b") is None
    assert _safe_get({"a": {"b": None}}, "a", "b", "c") is None
    assert _safe_get(None, "a") is None
    assert _safe_get("not_a_dict", "a") is None
    print("[6] _safe_get 边界 ✓")

    # 7) 情感调整字段幂等替换的正则
    pattern = re.compile(r"\s*\|\s*情感调整:[+\-]?\d+\.?\d*%")
    base = "完成 5日:62% 20日:58% 60日:55%"
    once = f"{pattern.sub('', base)}  |  情感调整:+2.5%"
    twice = f"{pattern.sub('', once)}  |  情感调整:+3.1%"
    assert twice.count("情感调整") == 1, f"幂等替换失败: {twice}"
    assert "情感调整:+3.1%" in twice
    assert "情感调整:+2.5%" not in twice
    print("[7] 情感字段幂等替换 ✓")

    print("\n全部自检通过 ✓")
    print("\n联网验证命令：")
    print("  python -c \"from quantpro_news_fix import fetch_news_robust; "
          "import json; "
          "print(json.dumps(fetch_news_robust('AAPL')[:3], indent=2, "
          "ensure_ascii=False))\"")