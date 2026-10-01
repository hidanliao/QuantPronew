"""
QuantPro — 美股 Top100 扫描提速补丁  v1.0
══════════════════════════════════════════════════════════════════
慢的根因（逐只股票、串行/重复的网络请求）：
  ① AnalysisThread._analyze 每只股票：history(10y) + get_realtime_price(fast_info，
     内部还会拉一次 1y 行情) + 再 new 一个 Ticker 读 previous_close(又一次) +
     可能回退到 .info(最慢的 quoteSummary)  → 100 只 ≈ 300~400 次请求，
     10 线程同时打 Yahoo 还容易被限流(429)，触发 fetch 里的 sleep(1) 重试
  ② _on_scan_done 又对 100 只股票串行重新下载 1y（缓存 key 是 _10y，根本不命中）
  ③ 扫描开头 10 个线程同时冷启动去拉 SPY 1mo（惊群）
  ④ 扫描完成后 logo 线程串行、每只最多 3 次请求(4~5s 超时)
  ⑤ 【第二次扫描依旧慢的真正元凶】v15 把 ProbabilityEngine.detect_regime 换成
     AdvancedRegimeDetector.current_regime，里面的 BOCPD 是 O(n²) 纯 Python 双重循环，
     每一步还逐点调 scipy.stats.t.pdf（~62 万次调用）：单只股票约 18 秒，100 只≈半小时，
     与网络无关、每次扫描都重算。而 detect_regime 只取 regime 标签，
     BOCPD 算出的 cp_recent 根本没被使用 → 白算。
     另外 v15 用一个全局 _global_detector，10 个扫描线程同时 fit() 会互相覆盖
     _labels/_cluster_model（线程不安全）。

本补丁：扫描前用「单次线程内的批量 yf.download」一次性预热 10y 缓存，
       _analyze 里的 fetch_stock_data 全部命中缓存。
       （yf.download 的全局状态问题只在多线程并发调用时出现；这里加锁保证同一时刻
         只有一个批量下载在跑，并逐只校验，失败的自动退回原来的单只下载。）

集成（quantpro_v1_6.py 末尾，其它补丁之前或之后均可）：
    from quantpro_top100_speedup import install_top100_speedup
    install_top100_speedup(globals())
══════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

_BATCH_LOCK = threading.Lock()   # 同一时刻只允许一个 yf.download 在跑
_CHUNK = 25                      # 每批 25 只：单批失败影响小，也便于刷新进度


def _split_frame(raw: pd.DataFrame, syms: List[str]) -> Dict[str, pd.DataFrame]:
    """把 yf.download(group_by='ticker') 的结果拆成 {sym: OHLCV DataFrame}。"""
    out: Dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not isinstance(raw.columns, pd.MultiIndex):        # 只有 1 只时是平表
        if len(syms) == 1 and all(c in raw.columns for c in cols):
            out[syms[0]] = raw[cols].dropna(subset=['Close'])
        return out
    lvl0 = set(raw.columns.get_level_values(0))
    for s in syms:
        if s not in lvl0:
            continue
        sub = raw[s]
        if not all(c in sub.columns for c in cols):
            continue
        sub = sub[cols].dropna(subset=['Close'])
        if not sub.empty:
            out[s] = sub
    return out


def prewarm_cache(env: dict, syms: List[str], period: str = "10y", progress_cb=None,
                  force: bool = False) -> int:
    """批量预热 env['_data_cache']，返回命中数量。失败的股票留给 fetch_stock_data 兜底。
    force=True：忽略已有缓存、全部重新下载（扫描用——主程序的 _data_cache 没有过期时间，
    不强制刷新的话第二次扫描会一直用旧K线，价格就不更新了）。下载失败的保留旧缓存。"""
    _normalize = env["_normalize"]
    cache, lock = env["_data_cache"], env["_cache_lock"]
    todo = []
    for s in dict.fromkeys(_normalize(x) for x in syms if x):
        with lock:
            if force or f"{s}_{period}" not in cache:
                todo.append(s)
    got = 0
    with _BATCH_LOCK:
        for i in range(0, len(todo), _CHUNK):
            chunk = todo[i:i + _CHUNK]
            try:
                raw = yf.download(chunk, period=period, auto_adjust=True, group_by="ticker",
                                  threads=True, progress=False)
                parts = _split_frame(raw, chunk)
            except Exception as e:
                logger.warning(f"[top100_speedup] 批量下载失败 {chunk[:3]}…: {e}")
                parts = {}
            for s, df in parts.items():
                try:
                    if getattr(df.index, "tz", None) is not None:
                        df.index = df.index.tz_localize(None)
                except Exception:
                    pass
                if len(df) >= 2:
                    with lock:
                        cache[f"{s}_{period}"] = df
                    got += 1
            if progress_cb:
                progress_cb(min(i + _CHUNK, len(todo)), len(todo))
    logger.info(f"[top100_speedup] 批量预热 {period}: {got}/{len(todo)}")
    return got


_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
_CHART_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")


def _fetch_one_quote(sess, sym: str):
    """Yahoo chart 接口（range=1d）的 meta：regularMarketPrice=现价，chartPreviousClose=昨收。
    每只 1 次轻量请求，不需要 crumb。失败返回 None。"""
    for host in _CHART_HOSTS:
        try:
            r = sess.get(f"https://{host}/v8/finance/chart/{sym}",
                         params={"range": "1d", "interval": "1d"},
                         headers=_UA, timeout=(3, 6))
            if r.status_code != 200:
                continue
            meta = r.json()["chart"]["result"][0]["meta"]
            px = meta.get("regularMarketPrice")
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            if px and prev and px > 0 and prev > 0:
                return float(px), float(prev)
        except Exception:
            continue
    return None


def fetch_live_quotes(syms: List[str], workers: int = 12) -> Dict[str, tuple]:
    """并行拉实时报价：{sym: (现价, 昨收)}。拿不到的不在结果里（调用方退回最后一根K线）。"""
    out: Dict[str, tuple] = {}
    with requests.Session() as sess:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s, q in zip(syms, ex.map(lambda x: _fetch_one_quote(sess, x), syms)):
                if q:
                    out[s] = q
    return out


def refresh_scan_quotes(env: dict, syms: List[str]) -> None:
    """扫描前：拉实时报价写入 env['_SCAN_QUOTES']，并用它校验批量下载的K线没被串数据。
    现价与K线最后收盘价偏差>25%的股票，丢弃批量结果、改用原来的单只 Ticker.history 重下。"""
    _normalize = env["_normalize"]
    fetch = env["fetch_stock_data"]
    cache, lock = env["_data_cache"], env["_cache_lock"]
    quotes = env.setdefault("_SCAN_QUOTES", {})
    quotes.clear()
    ss = list(dict.fromkeys(_normalize(x) for x in syms if x))
    live = fetch_live_quotes(ss)
    bad = []
    for s, (px, prev) in live.items():
        with lock:
            df = cache.get(f"{s}_10y")
        last = float(df["Close"].iloc[-1]) if df is not None and len(df) else None
        if last and abs(px - last) / last > 0.25:
            bad.append(s)
        else:
            quotes[s] = (px, prev)
    for s in bad:                          # 数据疑似串了：单只重下并复核
        with lock:
            cache.pop(f"{s}_10y", None)
        df = fetch(s, "10y")
        last = float(df["Close"].iloc[-1]) if df is not None and len(df) else None
        px, prev = live[s]
        if last and abs(px - last) / last <= 0.25:
            quotes[s] = (px, prev)
        else:
            logger.warning(f"[top100_speedup] {s} 现价{px}与K线{last}偏差过大，该股改用K线价")
    logger.info(f"[top100_speedup] 实时报价 {len(quotes)}/{len(ss)}，校验重下 {len(bad)}")


def _install_fast_regime(env: dict):
    """替换 ProbabilityEngine.detect_regime：每次调用用独立检测器（线程安全），
    并跳过结果不会被使用的 BOCPD。regime 标签的判定逻辑与 v15 原实现完全一致。"""
    import numpy as np
    from scipy import stats as _st
    ARD = env.get("AdvancedRegimeDetector")
    PE = env.get("ProbabilityEngine")
    if ARD is None or PE is None or getattr(PE, "_regime_fast_patched", False):
        return
    try:
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return
    _prev_detect = PE.detect_regime          # v15 增强版（异常时回退用）

    def _fast_detect_regime(self):
        try:
            close = self.close
            det = ARD()                       # 独立实例：不与其它线程共享状态
            det.fit(close)
            if det._cluster_model is None:
                return "neutral"
            feats = det._build_features(close, None)
            if feats.empty:
                return "neutral"
            X = StandardScaler().fit_transform(feats.values)
            dists = np.linalg.norm(X - X[-1].reshape(1, -1), axis=1)
            knn_labels = det._labels[np.argsort(dists)[:20]]
            cid = int(_st.mode(knn_labels, keepdims=False).mode)
            reg = det._label_map.get(cid, "neutral")
            if reg in ("crisis", "volatile"): return "high_vol"
            if reg in ("calm_bull", "calm_bear"): return "low_vol"
            return "neutral"
        except Exception as e:
            logger.warning(f"[top100_speedup] fast regime 回退: {e}")
            return _prev_detect(self)

    PE.detect_regime = _fast_detect_regime
    PE._regime_fast_patched = True
    logger.info("[top100_speedup] 已替换 detect_regime：跳过无用的 BOCPD + 线程安全")


def install_top100_speedup(env: dict):
    for k in ("AnalysisThread", "_data_cache", "_cache_lock", "_normalize", "fetch_stock_data"):
        if k not in env:
            raise RuntimeError(f"install_top100_speedup: 主程序缺少 {k}")
    AT = env["AnalysisThread"]
    if getattr(AT, "_speedup_patched", False):
        return AT
    fetch = env["fetch_stock_data"]
    _orig_run = AT.run

    def _run(self):
        syms = [s for s, _ in self.syms]
        try:
            self.progress.emit(0, "批量下载行情")
            # 每次扫描都强制刷新行情（批量下载 100 只约 2~3 秒），保证现价/涨跌/指标是最新的
            prewarm_cache(env, syms + ["SPY"], "10y", force=True)
            with env["_cache_lock"]:
                env["_data_cache"].pop("SPY_1mo", None)   # SPY 1mo 同样要刷新
            fetch("SPY", "1mo")          # 先单独缓存，避免 10 个线程同时冷启动去拉
            refresh_scan_quotes(env, syms)   # 实时现价/昨收 + 校验批量数据
        except Exception as e:
            logger.warning(f"[top100_speedup] 预热异常，退回逐只下载: {e}")
        _orig_run(self)

    AT.run = _run
    AT._speedup_patched = True
    _install_fast_regime(env)
    logger.info("[top100_speedup] 已安装：扫描前批量预热行情缓存")
    return AT


def _diagnose(syms: List[str]):
    """python quantpro_top100_speedup.py AAPL NVDA …  ：对比几种价格来源，定位"价格不对"。"""
    print(f"{'代码':<7}{'批量K线末根':>14}{'日期':>12}{'Ticker.history':>16}{'chart现价':>12}{'昨收':>10}{'fast_info':>12}")
    raw = yf.download(syms, period="5d", auto_adjust=True, group_by="ticker", progress=False)
    parts = _split_frame(raw, syms)
    live = fetch_live_quotes(syms)
    for s in syms:
        b = parts.get(s)
        bl = f"{float(b['Close'].iloc[-1]):.2f}" if b is not None and len(b) else "-"
        bd = str(b.index[-1].date()) if b is not None and len(b) else "-"
        try:
            h = yf.Ticker(s).history(period="5d", auto_adjust=True)
            hl = f"{float(h['Close'].iloc[-1]):.2f}"
        except Exception:
            hl = "-"
        q = live.get(s)
        try:
            fi = f"{float(yf.Ticker(s).fast_info.last_price):.2f}"
        except Exception:
            fi = "-"
        print(f"{s:<7}{bl:>14}{bd:>12}{hl:>16}{(f'{q[0]:.2f}' if q else '-'):>12}"
              f"{(f'{q[1]:.2f}' if q else '-'):>10}{fi:>12}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1:
    _diagnose([a.upper() for a in __import__("sys").argv[1:]])
elif __name__ == "__main__":
    import numpy as np
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = pd.MultiIndex.from_product([["AAA", "BBB", "CCC"], ["Open", "High", "Low", "Close", "Volume"]])
    rng = np.random.default_rng(1)
    raw = pd.DataFrame(rng.random((30, 15)) + 10, index=idx, columns=cols)
    raw[("CCC", "Close")] = np.nan                       # CCC 无数据
    parts = _split_frame(raw, ["AAA", "BBB", "CCC", "DDD"])
    assert set(parts) == {"AAA", "BBB"}, parts.keys()
    assert not parts["AAA"].equals(parts["BBB"])         # 各股数据独立，不串
    one = raw["AAA"].copy()
    assert set(_split_frame(one, ["AAA"])) == {"AAA"}
    print("自检通过 ✓ 多股拆分/缺失剔除/单股平表 正常")
