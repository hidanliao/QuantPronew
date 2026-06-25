"""
QuantPro — 个股 K 线清爽化 + 金叉/死叉标记  v1.0
══════════════════════════════════════════════════════════════════
问题：价格目标补丁(quantpro_price_target_patch.py)给个股K线叠了太多层
      —— 3周期 MC 价格带(P5/P50/P95)、±1/1.5/2 ATR 共6条、7条 Fibonacci
      线、外置图例 —— 导致蜡烛图被埋没、显示糊乱，远不如大盘仪表盘清爽。

修复目标（用户明确要求）：
  1. K线画得像大盘仪表盘那样清爽：蜡烛 + 成交量 + MA(5/20/60)，没有满屏叠加线
  2. 能看到金叉/死叉：MA20 上穿 MA60 = 金叉(绿▲)，下穿 = 死叉(红▼)
  3. 仍保留概率Tab里的「价格目标卡片」（数字信息在那里干净列出，不挤在图上）
  4. 想看预测带的人可以用「预测带」开关手动打开（默认关）

设计：覆盖 ProbabilityDialog._kline_plot，在价格目标补丁之后安装（后装者生效）。
依赖全部从主程序 globals 取，零新依赖。

集成（quantpro_v1_6.py 末尾、patch_probability_dialog(...) 之后、入口之前）：

    from quantpro_clean_kline import install_clean_kline
    install_clean_kline(globals())
══════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import matplotlib as mpl
import mplfinance as mpf

logger = logging.getLogger(__name__)


def _detect_ma_cross(close: pd.Series, fast: int = 20, slow: int = 60):
    """
    返回金叉/死叉布尔序列（与 close 同索引）。
    金叉：fast 上穿 slow；死叉：fast 下穿 slow。
    """
    ma_f = close.rolling(fast).mean()
    ma_s = close.rolling(slow).mean()
    prev_f, prev_s = ma_f.shift(1), ma_s.shift(1)
    golden = (prev_f <= prev_s) & (ma_f > ma_s)
    death = (prev_f >= prev_s) & (ma_f < ma_s)
    return golden.fillna(False), death.fillna(False), ma_f, ma_s


def install_clean_kline(env: dict):
    """
    env = 主程序 globals()。覆盖 ProbabilityDialog._kline_plot 为清爽版。
    必须在 patch_probability_dialog(...) 之后调用，确保覆盖补丁版本。
    """
    required = ["ProbabilityDialog", "fetch_stock_data", "get_realtime_price",
                "_currency", "_make_mpf_style", "_get_mpl_rc",
                "_apply_font_to_figure", "t", "T"]
    missing = [k for k in required if k not in env]
    if missing:
        raise RuntimeError(f"install_clean_kline: 主程序缺少 {missing}")

    PD = env["ProbabilityDialog"]
    fetch_stock_data = env["fetch_stock_data"]
    get_realtime_price = env["get_realtime_price"]
    _currency = env["_currency"]
    _make_mpf_style = env["_make_mpf_style"]
    _get_mpl_rc = env["_get_mpl_rc"]
    _apply_font_to_figure = env["_apply_font_to_figure"]
    t = env["t"]
    T = env["T"]
    PriceTargetEngine = env.get("PriceTargetEngine")  # 来自价格目标补丁，可能不存在

    def _clean_kline_plot(self):
        sym = self.sym
        period = self._kline_period
        df = fetch_stock_data(sym, period)
        if df is None or df.empty:
            self.kline_info.setText(t("kline_no_data"))
            return

        ohlcv = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        for c in ohlcv.columns:
            ohlcv[c] = ohlcv[c].squeeze().astype(float)
        ohlcv.dropna(inplace=True)
        if len(ohlcv) > 1500:
            ohlcv = ohlcv.iloc[-1500:]
        if len(ohlcv) < 2:
            self.kline_info.setText(t("kline_no_data"))
            return

        cur_sym = _currency(sym)
        rp = get_realtime_price(sym)
        last_close = float(ohlcv['Close'].iloc[-1])
        if rp:
            cp, src = rp, "实时"
            chg = (cp - last_close) / last_close * 100
        else:
            cp, src = last_close, "历史"
            prev = float(ohlcv['Close'].iloc[-2])
            chg = (last_close - prev) / prev * 100 if prev else 0.0

        # 金叉/死叉检测（MA20 vs MA60）
        close = ohlcv['Close']
        golden, death, ma20, ma60 = _detect_ma_cross(close, 20, 60)
        n_gc, n_dc = int(golden.sum()), int(death.sum())

        cc = T.GREEN if chg >= 0 else T.RED
        cross_note = ""
        if n_gc or n_dc:
            cross_note = (f"　<span style='color:{T.GREEN}'>金叉×{n_gc}</span>"
                          f"　<span style='color:{T.RED}'>死叉×{n_dc}</span>")
        # 最近一次穿越
        last_cross = ""
        gd = golden[golden].index.tolist()
        dd = death[death].index.tolist()
        last_g = gd[-1] if gd else None
        last_d = dd[-1] if dd else None
        if last_g or last_d:
            if last_g and (last_d is None or last_g >= last_d):
                last_cross = f"　最近<span style='color:{T.GREEN}'>金叉</span> {last_g.strftime('%m-%d')}"
            else:
                last_cross = f"　最近<span style='color:{T.RED}'>死叉</span> {last_d.strftime('%m-%d')}"

        self.kline_info.setText(
            f"<b>{sym}</b>  {src}: <b>{cur_sym}{cp:.2f}</b>  "
            f"<b style='color:{cc}'>{chg:+.2f}%</b>  "
            f"{t('kline_bars', n=len(ohlcv))} [{period.upper()}]{cross_note}{last_cross}"
        )

        try:
            style = _make_mpf_style()
            mav_args = (5, 20) if len(ohlcv) < 60 else (5, 20, 60)

            # 金叉/死叉散点（对齐K线x轴；mplfinance 的 addplot 自动对齐索引）
            addplots = []
            # 放在K线下方/上方一点，避免压住蜡烛
            span = float((ohlcv['High'] - ohlcv['Low']).tail(60).mean() or 0)
            gc_series = pd.Series(np.nan, index=ohlcv.index)
            dc_series = pd.Series(np.nan, index=ohlcv.index)
            gc_idx = golden.reindex(ohlcv.index, fill_value=False)
            dc_idx = death.reindex(ohlcv.index, fill_value=False)
            gc_series[gc_idx] = ohlcv['Low'][gc_idx] - span * 0.8
            dc_series[dc_idx] = ohlcv['High'][dc_idx] + span * 0.8

            if gc_idx.any():
                addplots.append(mpf.make_addplot(
                    gc_series, type='scatter', marker='^', markersize=90,
                    color='#10b981', panel=0))
            if dc_idx.any():
                addplots.append(mpf.make_addplot(
                    dc_series, type='scatter', marker='v', markersize=90,
                    color='#ef4444', panel=0))

            # 可选叠加层（默认关，避免糊乱）—— 用户可用「预测带」开关打开
            show_overlay = getattr(self, "_show_pred_overlay", False)

            with mpl.rc_context(_get_mpl_rc()):
                plot_kw = dict(
                    type='candle', volume=True, mav=mav_args,
                    returnfig=True, title=f"  {sym}", style=style, figsize=(8, 5),
                )
                if addplots:
                    plot_kw['addplot'] = addplots
                fig, axes = mpf.plot(ohlcv, **plot_kw)

            # 计算价格目标数据（卡片要用）——但默认不画在图上
            if PriceTargetEngine is not None and getattr(self, "_ind", None) is not None:
                try:
                    engine = PriceTargetEngine(self._ind['close'], self._ind, cp)
                    self._target_data = engine.summary()
                except Exception as _e:
                    logger.warning(f"price target compute: {_e}")
                    self._target_data = None

                # 仅当用户主动开启「预测带」才叠加（复用补丁的绘制器）
                if show_overlay and self._target_data:
                    painter = env.get("KlineOverlayPainter")
                    if painter is not None:
                        try:
                            painter.overlay_price_targets(
                                ax=axes[0], df=ohlcv, target_data=self._target_data,
                                horizons=[5, 20, 60], show_mc=True, show_atr=True,
                                show_fib=False, currency=cur_sym)
                        except Exception as _e:
                            logger.warning(f"overlay: {_e}")

            _apply_font_to_figure(fig)
            self.kline_canvas.figure = fig
            fig.canvas = self.kline_canvas
            # 关键修复：把新建的固定尺寸 figure 拉伸到 canvas 控件的实际大小，
            # 否则窗口最大化后图仍是 8x5 英寸、钉在左上角、右下大片空白。
            try:
                cw = max(self.kline_canvas.width(), 200)
                ch = max(self.kline_canvas.height(), 200)
                dpi = fig.get_dpi() or 100
                fig.set_size_inches(cw / dpi, ch / dpi)
                try:
                    fig.tight_layout(pad=1.2)
                except Exception:
                    pass
            except Exception as _e:
                logger.warning(f"kline fit canvas: {_e}")
            self.kline_canvas.draw_idle()

            # 刷新概率Tab价格目标卡片（数字信息的干净归处）
            if getattr(self, "_target_data", None) and hasattr(self, "_refresh_price_target_widget"):
                try:
                    self._refresh_price_target_widget(cur_sym)
                except Exception as _e:
                    logger.warning(f"refresh price target widget: {_e}")

        except Exception as e:
            self.kline_info.setText(t("kline_plot_fail", e=e))
            logger.error(f"CleanKline error: {e}", exc_info=True)

    PD._kline_plot = _clean_kline_plot

    # ── 给K线tab按钮行加一个「预测带」开关（默认关）──────────────
    _orig_build = PD._build_tab_kline

    def _build_with_toggle(self):
        from PyQt5.QtWidgets import QCheckBox, QHBoxLayout, QPushButton
        from PyQt5.QtCore import QObject, QEvent
        _orig_build(self)
        self._show_pred_overlay = False
        # ── 让K线图始终填满canvas控件（修复最大化后图不放大）──────────
        # PyQt里实例级覆盖 resizeEvent 无效，必须用事件过滤器捕获 Resize。
        try:
            canvas = self.kline_canvas

            class _CanvasFitFilter(QObject):
                def eventFilter(self, obj, ev):
                    if ev.type() == QEvent.Resize:
                        try:
                            fig = obj.figure
                            if fig is not None:
                                dpi = fig.get_dpi() or 100
                                fig.set_size_inches(max(obj.width(), 200) / dpi,
                                                    max(obj.height(), 200) / dpi)
                                try: fig.tight_layout(pad=1.2)
                                except Exception: pass
                                obj.draw_idle()
                        except Exception:
                            pass
                    return False  # 不拦截，继续正常处理

            self._kline_fit_filter = _CanvasFitFilter(self)   # 持有引用防GC
            canvas.installEventFilter(self._kline_fit_filter)
        except Exception as _e:
            logger.warning(f"kline canvas resize filter: {_e}")
        # 找到K线tab的内容widget（穿透滚动区）
        inner = None
        if hasattr(self, "tab_inner"):
            inner = self.tab_inner(t("tab_kline"))
        if inner is None:
            return
        lay = inner.layout()
        if lay is None:
            return
        # 最后一个布局项是按钮行
        btn_row = lay.itemAt(lay.count() - 1)
        row_lay = btn_row.layout() if btn_row else None
        if row_lay is None:
            return
        chk = QCheckBox("预测带")
        chk.setToolTip("叠加蒙特卡洛预测价格带 + ATR 位（默认关，开启会让图变密）")
        chk.setChecked(False)
        chk.setStyleSheet(f"color:{T.TEXT_2};")

        def _toggle(state):
            self._show_pred_overlay = bool(state)
            self._kline_plot()
        chk.stateChanged.connect(_toggle)
        # 插到刷新按钮之前（行尾 stretch 之后的位置）
        row_lay.addWidget(chk)
        self._pred_overlay_chk = chk

    PD._build_tab_kline = _build_with_toggle

    logger.info("[clean_kline] 已安装：个股K线清爽化 + 金叉/死叉标记（预测带默认关）")
    return PD


# ── 独立自检（合成数据，验证金叉死叉检测正确）─────────────────────
if __name__ == "__main__":
    idx = pd.bdate_range("2024-01-01", periods=300)
    # 先涨(MA20在MA60上方)→大跌(死叉)→再涨(金叉)，确保两种穿越都出现
    base = np.concatenate([
        np.linspace(80, 120, 100),   # 上涨 → MA20 位于 MA60 上方
        np.linspace(120, 80, 100),   # 下跌 → MA20 下穿 MA60（死叉）
        np.linspace(80, 130, 100),   # 上涨 → MA20 上穿 MA60（金叉）
    ])
    noise = np.random.default_rng(0).normal(0, 0.5, 300)
    close = pd.Series(base + noise, index=idx)
    g, d, ma20, ma60 = _detect_ma_cross(close, 20, 60)
    print(f"金叉次数={int(g.sum())}, 死叉次数={int(d.sum())}")
    print("金叉日期:", [x.strftime('%Y-%m-%d') for x in g[g].index])
    print("死叉日期:", [x.strftime('%Y-%m-%d') for x in d[d].index])
    assert g.sum() >= 1 and d.sum() >= 1, "应至少各检测到一次金叉和死叉"
    print("自检通过 ✓ 金叉死叉检测正常")
