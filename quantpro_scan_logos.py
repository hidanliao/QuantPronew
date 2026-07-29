"""
QuantPro — 主界面扫描列表 logo 图标  v1.1（线程安全版）
══════════════════════════════════════════════════════════════════
需求：主界面扫描列表代码列显示公司 logo（真实图标或彩色首字母兜底）。
修复：线程结束后自动清空引用，避免关闭窗口时 RuntimeError。
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging

try:
    import quantpro_ticker_logos as _tlogos
    HAS_LOGOS = True
except ImportError:
    _tlogos = None
    HAS_LOGOS = False

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QIcon

logger = logging.getLogger(__name__)

_LOGO_SIZE = 20


def install_scan_logos(env: dict):
    """安装主界面扫描列表 logo 功能。若缺失依赖则跳过。"""
    if not HAS_LOGOS:
        logger.warning("[scan_logos] 未找到 quantpro_ticker_logos，跳过主界面logo功能（不影响其它功能）")
        return env

    required = ["ScanModel", "QuantApp", "_COL_KEYS"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_scan_logos: 主程序缺少 {missing}")

    ScanModel = env["ScanModel"]
    QuantApp = env["QuantApp"]
    _COL_KEYS = env["_COL_KEYS"]
    if "col_code" not in _COL_KEYS:
        raise RuntimeError("install_scan_logos: 主程序 _COL_KEYS 里没有 col_code 列")
    code_col = _COL_KEYS.index("col_code")

    # ── ScanModel：代码列返回 QIcon ──────────────────────────
    _orig_data = ScanModel.data

    def _data_with_logo(self, index, role=Qt.DisplayRole):
        if role == Qt.DecorationRole and index.isValid() and index.column() == code_col:
            row = self._rows[index.row()]
            sym = row.get("col_code", "")
            if not sym:
                return None
            try:
                pm = _tlogos.get_cached_pixmap(sym, _LOGO_SIZE)
                if pm is None:
                    # 立即显示兜底图标（避免空白）
                    pm = _tlogos.make_fallback_icon(sym, _LOGO_SIZE)
                    _tlogos.cache_pixmap(sym, _LOGO_SIZE, pm)
                return QIcon(pm)
            except Exception as e:
                logger.warning(f"[scan_logos] 图标获取失败 {sym}: {e}")
                return None
        return _orig_data(self, index, role)

    ScanModel.data = _data_with_logo

    # ── 记录待加载 logo 的代码 ──────────────────────────────
    _orig_append = ScanModel.append_row

    def _append_with_logo_queue(self, d):
        _orig_append(self, d)
        sym = d.get("col_code", "")
        if sym:
            pending = getattr(self, "_pending_logo_syms", None)
            if pending is None:
                pending = set()
                self._pending_logo_syms = pending
            pending.add(sym)

    ScanModel.append_row = _append_with_logo_queue

    _orig_clear = ScanModel.clear

    def _clear_with_logo_queue(self):
        self._pending_logo_syms = set()
        _orig_clear(self)

    ScanModel.clear = _clear_with_logo_queue

    # ── 扫描完成后批量加载 logo ─────────────────────────────
    _orig_on_scan_done = QuantApp._on_scan_done

    def _on_scan_done_with_logos(self, n):
        _orig_on_scan_done(self, n)
        try:
            _start_scan_logo_loading(self, code_col)
        except Exception as e:
            logger.warning(f"[scan_logos] 批量拉取logo启动失败: {e}")

    QuantApp._on_scan_done = _on_scan_done_with_logos

    # ── 初始化时设置图标尺寸 + 状态栏署名 ──────────────────
    _orig_init = QuantApp.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        try:
            self.table.setIconSize(QSize(_LOGO_SIZE, _LOGO_SIZE))
        except Exception as e:
            logger.warning(f"[scan_logos] 设置表格图标尺寸失败: {e}")
        try:
            if hasattr(self, "status_bar"):
                attr_lbl = QLabel(_tlogos.attribution_html())
                attr_lbl.setOpenExternalLinks(True)
                self.status_bar.addPermanentWidget(attr_lbl)
        except Exception as e:
            logger.warning(f"[scan_logos] 状态栏署名链接添加失败: {e}")

    QuantApp.__init__ = _new_init

    logger.info("[scan_logos] 已安装：主界面扫描列表代码列显示公司logo")
    return env


def _start_scan_logo_loading(app_self, code_col: int):
    """扫描全部完成后，批量后台拉取 logo，逐个刷新对应行。"""
    model = getattr(app_self, "scan_model", None)
    if model is None:
        return
    pending = getattr(model, "_pending_logo_syms", None)
    if not pending:
        return
    symbols = list(pending)
    model._pending_logo_syms = set()

    # 停止之前未完成的线程
    old = getattr(app_self, "_scan_logo_thread", None)
    if old is not None:
        try:
            if old.isRunning():
                old.quit()
                old.wait(300)
        except Exception:
            pass

    th = _tlogos.LogoLoaderThread(symbols, parent=app_self)

    def _refresh_symbol(symbol: str):
        rows = getattr(model, "_rows", None)
        if not rows:
            return
        sym_u = symbol.upper()
        for i, row in enumerate(rows):
            if str(row.get("col_code", "")).upper() == sym_u:
                idx = model.index(i, code_col)
                model.dataChanged.emit(idx, idx, [Qt.DecorationRole])

    def _on_loaded(symbol, data):
        pm = _tlogos.pixmap_from_bytes(data, _LOGO_SIZE)
        if pm is None:
            _on_missing(symbol)
            return
        _tlogos.cache_pixmap(symbol, _LOGO_SIZE, pm)
        _refresh_symbol(symbol)

    def _on_missing(symbol):
        pm = _tlogos.make_fallback_icon(symbol, _LOGO_SIZE)
        _tlogos.cache_pixmap(symbol, _LOGO_SIZE, pm)
        _refresh_symbol(symbol)

    th.loaded.connect(_on_loaded)
    th.missing.connect(_on_missing)
    # 线程结束后清理引用
    th.finished_all.connect(lambda: setattr(app_self, '_scan_logo_thread', None))
    app_self._scan_logo_thread = th
    th.start()