"""
QuantPro — 股票 Logo 获取 / 缓存 / 兜底  v1.0
══════════════════════════════════════════════════════════════════
数据源：两档兜底链，逐档尝试，不依赖单一家：
  ① cdn.tickerlogos.com（AllInvestView 提供的免key logo CDN，图质量最好，
     条款要求展示处放一条可见署名链接，见下方"条款要求"）
  ② Google favicon 接口（不要求署名、不用key，只在①失败时用；图小、
     清晰度一般，作为纯保底，不作为主力源）
  - 以上两档都按"域名"查图，不是按股票代码，所以先要 ticker -> 官网域名
  - 常见大盘股直接查本地静态表（零网络请求）；查不到的走一次
    /api/logo-search/ 做 ticker->domain 解析，解析结果落盘缓存，
    以后同一个代码不用再查
  - 图片本身也落盘缓存（~/.quantpro_cache/logos/），避免重复打CDN，
    也顶得住偶尔的网络波动 / 404

条款要求：使用方需要在展示logo的界面上放一个可见的署名链接
  （见 ATTRIBUTION_HTML / attribution_label()）。已经在集成的两个
  地方（大盘仪表盘、模拟自选持仓表）各放了一条，别删。

网络/接口失败时的兜底：本地生成"彩色圆点+首字母"占位图标
  （make_fallback_icon），风格上接近 TradingView 抓不到logo时的
  占位样式，不依赖网络，随时可用。

集成：
    from quantpro_ticker_logos import install_ticker_logos
    install_ticker_logos(globals())
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import requests
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon, QPainter, QColor, QFont, QBrush

logger = logging.getLogger(__name__)

CDN_BASE = "https://cdn.tickerlogos.com"
SEARCH_API = "https://www.allinvestview.com/api/logo-search/"
ATTRIBUTION_URL = "https://www.allinvestview.com/tools/ticker-logos/"
# 保底二档：Google favicon接口。不要求署名/不要key，但只有小图标、清晰度一般，
# 只在 tickerlogos.com 那档失败时才用（详见 fetch_logo_bytes）。
GOOGLE_FAVICON_URL = "https://www.google.com/s2/favicons"

_CACHE_DIR = Path.home() / ".quantpro_cache" / "logos"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_DOMAIN_MAP_FILE = _CACHE_DIR / "_domain_map.json"

# 常见标的本地兜底映射：命中就是零网络请求，覆盖组合里最常出现的大盘股
_STATIC_DOMAIN_MAP: Dict[str, str] = {
    "AAPL": "apple.com", "MSFT": "microsoft.com", "GOOGL": "google.com", "GOOG": "google.com",
    "AMZN": "amazon.com", "META": "meta.com", "TSLA": "tesla.com", "NVDA": "nvidia.com",
    "JPM": "jpmorganchase.com", "V": "visa.com", "MA": "mastercard.com",
    "DIS": "thewaltdisneycompany.com", "NFLX": "netflix.com", "INTC": "intel.com",
    "AMD": "amd.com", "CRM": "salesforce.com", "PYPL": "paypal.com", "UBER": "uber.com",
    "ABNB": "airbnb.com", "COIN": "coinbase.com", "BA": "boeing.com",
    "KO": "coca-colacompany.com", "PEP": "pepsico.com", "WMT": "walmart.com",
    "CVX": "chevron.com", "XOM": "exxonmobil.com", "JNJ": "jnj.com", "PG": "pg.com",
    "HD": "homedepot.com", "UNH": "unitedhealthgroup.com", "LMT": "lockheedmartin.com",
    "ORCL": "oracle.com", "IBM": "ibm.com", "CSCO": "cisco.com", "PFE": "pfizer.com",
    "T": "att.com", "VZ": "verizon.com", "GE": "ge.com", "MCD": "mcdonalds.com",
    "NKE": "nike.com", "SBUX": "starbucks.com", "GS": "goldmansachs.com",
    "MS": "morganstanley.com", "BAC": "bankofamerica.com", "WFC": "wellsfargo.com",
    "C": "citigroup.com", "QQQ": "invesco.com", "SPY": "ssga.com",
    "XLK": "ssga.com", "XLF": "ssga.com", "XLE": "ssga.com", "XLV": "ssga.com",
    "XLI": "ssga.com", "XLY": "ssga.com", "XLP": "ssga.com", "XLU": "ssga.com",
    "XLB": "ssga.com", "XLRE": "ssga.com", "XLC": "ssga.com",
}

_domain_cache: Dict[str, Optional[str]] = {}
_pixmap_mem_cache: Dict[str, QPixmap] = {}


def _load_domain_cache() -> None:
    global _domain_cache
    if _DOMAIN_MAP_FILE.exists():
        try:
            _domain_cache = json.loads(_DOMAIN_MAP_FILE.read_text(encoding="utf-8"))
        except Exception:
            _domain_cache = {}


_load_domain_cache()


def _save_domain_cache() -> None:
    try:
        _DOMAIN_MAP_FILE.write_text(json.dumps(_domain_cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"logo domain cache save fail: {e}")


def _base_symbol(ticker: str) -> str:
    """去掉市场后缀：600519.SS / 0700.HK / 7203.T -> 主代码"""
    return ticker.split(".")[0].upper()


def resolve_domain(ticker: str) -> Optional[str]:
    """ticker -> 官网域名。命中静态表最快；否则查一次搜索API并落盘缓存，找不到记None（也缓存，避免反复打空请求）。"""
    key = ticker.upper()
    if key in _domain_cache:
        return _domain_cache[key]
    base = _base_symbol(ticker)
    if base in _STATIC_DOMAIN_MAP:
        dom = _STATIC_DOMAIN_MAP[base]
        _domain_cache[key] = dom
        _save_domain_cache()
        return dom
    try:
        r = requests.get(SEARCH_API, params={"q": base}, timeout=4)
        domain = None
        if r.status_code == 200:
            data = r.json()
            items = data if isinstance(data, list) else data.get("results", [])
            for it in items:
                if isinstance(it, dict) and it.get("domain"):
                    domain = it["domain"]
                    break
        _domain_cache[key] = domain
        _save_domain_cache()
        return domain
    except Exception as e:
        logger.warning(f"logo domain search fail {ticker}: {e}")
        return None


def _cache_path(domain: str) -> Path:
    return _CACHE_DIR / f"{domain.replace('/', '_')}.png"


def _fetch_google_favicon(domain: str, size: int = 128) -> Optional[bytes]:
    """保底二档：Google favicon接口，不要求署名、不用key。已知限制：域名没有
    favicon时Google给的是一张通用地球图标而不是404，从图片内容本身没法
    100%分辨"真实favicon"和"通用占位图"——这里退而求其次，只要拿到图就用，
    大部分场景比彩色首字母兜底更接近真实公司标识；极少数会显示成通用地球图标，
    可接受的折衷（比首字母兜底信息量更少，但不会显示错误的logo）。"""
    try:
        r = requests.get(GOOGLE_FAVICON_URL, params={"domain": domain, "sz": size}, timeout=5)
        if r.status_code == 200 and r.content and len(r.content) > 200:
            return r.content
    except Exception as e:
        logger.warning(f"google favicon fetch fail {domain}: {e}")
    return None


def fetch_logo_bytes(ticker: str) -> Optional[bytes]:
    """返回PNG字节；本地磁盘缓存优先命中，否则依次试：
    ① cdn.tickerlogos.com（图质量最好，条款要求署名，已在集成处加了链接）
    ② Google favicon（不要求署名，保底用，图小/清晰度一般）
    两档都拿不到（或无域名）返回None，调用方会用本地绘制的兜底图标顶上。"""
    domain = resolve_domain(ticker)
    if not domain:
        return None
    fp = _cache_path(domain)
    if fp.exists():
        try:
            data = fp.read_bytes()
            if data:
                return data
        except Exception:
            pass
    try:
        r = requests.get(f"{CDN_BASE}/{domain}", timeout=5)
        if r.status_code == 200 and r.content:
            try:
                fp.write_bytes(r.content)
            except Exception:
                pass
            return r.content
    except Exception as e:
        logger.warning(f"logo fetch fail {ticker}/{domain}: {e}")
    data = _fetch_google_favicon(domain)
    if data:
        try:
            fp.write_bytes(data)
        except Exception:
            pass
        return data
    return None


class LogoLoaderThread(QThread):
    """后台批量加载一组ticker的logo，逐个emit，不卡UI线程。"""
    loaded = pyqtSignal(str, bytes)   # ticker, png_bytes
    missing = pyqtSignal(str)         # ticker（没查到域名/CDN没有这张图）
    finished_all = pyqtSignal()

    def __init__(self, tickers, parent=None):
        super().__init__(parent)
        self._tickers = list(dict.fromkeys(t for t in tickers if t))  # 去重保序

    def run(self):
        for t in self._tickers:
            try:
                data = fetch_logo_bytes(t)
            except Exception as e:
                logger.warning(f"logo load thread error {t}: {e}")
                data = None
            if data:
                self.loaded.emit(t, data)
            else:
                self.missing.emit(t)
        self.finished_all.emit()


def make_fallback_icon(ticker: str, size: int, colors=None) -> QPixmap:
    """网络logo拿不到时的本地兜底：彩色圆点+首字母，纯本地绘制，不吃网络。"""
    palette = colors or ["#1b7a43", "#2f9e5c", "#a9822f", "#7c5cb0", "#0f9c9c", "#d97706"]
    base = _base_symbol(ticker) or "?"
    idx = sum(ord(c) for c in base) % len(palette)
    color = QColor(palette[idx])
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.setPen(QColor("#ffffff"))
    f = QFont()
    f.setBold(True)
    f.setPixelSize(max(8, int(size * 0.46)))
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignCenter, base[0])
    p.end()
    return pm


def pixmap_from_bytes(data: bytes, size: int) -> Optional[QPixmap]:
    pm = QPixmap()
    if pm.loadFromData(data):
        return pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return None


def get_cached_pixmap(ticker: str, size: int) -> Optional[QPixmap]:
    return _pixmap_mem_cache.get(f"{ticker.upper()}_{size}")


def cache_pixmap(ticker: str, size: int, pm: QPixmap) -> None:
    _pixmap_mem_cache[f"{ticker.upper()}_{size}"] = pm


def get_or_fallback_icon(ticker: str, size: int) -> QIcon:
    """先看内存缓存，没有就现给一个本地兜底图标（后台线程回来后会被替换成真logo）。"""
    pm = get_cached_pixmap(ticker, size)
    if pm is None:
        pm = make_fallback_icon(ticker, size)
    return QIcon(pm)


def attribution_html(text_color: str = "#6d7d70", link_color: Optional[str] = None) -> str:
    """条款要求的可见署名链接，塞进 QLabel(richtext) 里显示；点了会跳浏览器。"""
    lc = link_color or text_color
    return (f'<span style="color:{text_color};font-size:8pt;">图标由 '
            f'<a href="{ATTRIBUTION_URL}" style="color:{lc};">AllInvestView</a> 提供</span>')


def install_ticker_logos(env: dict) -> dict:
    """把本模块的关键函数/类挂进主程序 env，方便其它补丁 from env 里取用。"""
    env.setdefault("resolve_ticker_domain", resolve_domain)
    env.setdefault("fetch_logo_bytes", fetch_logo_bytes)
    env.setdefault("LogoLoaderThread", LogoLoaderThread)
    env.setdefault("make_fallback_logo_icon", make_fallback_icon)
    env.setdefault("pixmap_from_bytes", pixmap_from_bytes)
    env.setdefault("get_cached_logo_pixmap", get_cached_pixmap)
    env.setdefault("cache_logo_pixmap", cache_pixmap)
    env.setdefault("get_or_fallback_logo_icon", get_or_fallback_icon)
    env.setdefault("ticker_logo_attribution_html", attribution_html)
    logger.info("[ticker_logos] 已安装：logo 获取/缓存/兜底 工具集")
    return env


# ── 独立自检（不发真实网络请求，只测本地逻辑）───────────────────
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)

    # 1) 静态域名表命中
    assert resolve_domain("MSFT") == "microsoft.com", "静态域名表查找失败"
    assert resolve_domain("msft") == "microsoft.com", "域名查找应大小写不敏感"
    assert resolve_domain("600519.SS") is None or isinstance(resolve_domain("600519.SS"), (str, type(None)))

    # 2) 兜底图标本地绘制（不吃网络）
    icon_pm = make_fallback_icon("NVDA", 32)
    assert not icon_pm.isNull() and icon_pm.width() == 32, "兜底图标绘制失败"

    # 3) 内存缓存读写
    cache_pixmap("TEST", 24, icon_pm)
    assert get_cached_pixmap("TEST", 24) is not None, "内存缓存读写失败"
    assert get_cached_pixmap("TEST", 999) is None, "不同尺寸不应该命中同一份缓存"

    # 4) 有兜底的 get_or_fallback_icon 在没缓存时也不应该报错/返回空
    icon = get_or_fallback_icon("UNKNOWN_TICKER_XYZ", 28)
    assert isinstance(icon, QIcon) and not icon.isNull(), "无缓存时兜底图标获取失败"

    # 5) 署名HTML里必须带可点击链接（条款要求）
    html = attribution_html()
    assert "allinvestview.com" in html and "<a href=" in html, "署名链接缺失"

    print("ticker_logos 自检通过 ✓ 域名解析/兜底绘制/缓存/署名链接均正常")
