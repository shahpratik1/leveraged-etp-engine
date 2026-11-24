import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from trading_system.agents.risk_manager import DecisionCard, RiskManager
from trading_system.agents.strategy_improvement_agent import (
    ImprovementAudit,
    StrategyImprovementAgent,
)
from trading_system.backtest.engine import BacktestEngine
from trading_system.config_bus import ConfigBus, ConfigSnapshot
from trading_system.config_schema import ALLOWLISTED_SCREENER_METRICS
from trading_system.screener.laya_screener import (
    CycleFunnelReport,
    LayaScreener,
    ScreenerResult,
    TechnicalFallbackScreener,
    build_cycle_funnel_report,
    simulate_screener_gate_impact,
)
from trading_system.strategies.base import StrategySignal
from trading_system.strategies.directional_trend import DirectionalTrendStrategy


def _make_bus(tmpdir: str) -> ConfigBus:
    return ConfigBus(
        data_dir=tmpdir,
        env_file=os.path.join(tmpdir, ".env.nonexistent"),
        ignore_process_env=True,
    )


def _bullish_metrics(
    *,
    close: float = 150.0,
    vwap: float = 148.0,
    vwap_dist_pct: float = 1.35,
    sma200: float = 130.0,
    ema20: float = 148.0,
    ema50: float = 142.0,
    ema20_slope_5d: float = 0.8,
    adx: float = 21.0,
    plus_di: float = 28.0,
    minus_di: float = 14.0,
    rsi3: float = 65.0,
    rsi7: float = 60.0,
    rsi14: float = 58.0,
    macd: float = 0.5,
    macd_signal: float = 0.3,
    atr: float = 3.0,
    atr_20d_sma: float = 2.8,
    atr_spike_ratio: float = 1.07,
    realized_vol: float = 0.25,
    earnings_in_days: float = 10.0,
) -> dict:
    return {
        "close": close,
        "vwap": vwap,
        "vwap_dist_pct": vwap_dist_pct,
        "sma200": sma200,
        "ema20": ema20,
        "ema50": ema50,
        "ema20_slope_5d": ema20_slope_5d,
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "rsi3": rsi3,
        "rsi7": rsi7,
        "rsi14": rsi14,
        "macd": macd,
        "macd_signal": macd_signal,
        "atr": atr,
        "atr_20d_sma": atr_20d_sma,
        "atr_spike_ratio": atr_spike_ratio,
        "realized_vol": realized_vol,
        "earnings_in_days": earnings_in_days,
    }


def _make_trending_candles(count: int = 240) -> list:
    """Chop warmup (30 bars) then steady swing uptrend with pullbacks so Wilder ADX stays <= ~40."""
    candles = []
    price = 100.0
    for idx in range(count):
        if idx < 30:
            step = 0.4 if (idx % 2 == 0) else -0.4
        else:
            step = 1.4 if ((idx - 30) % 7) < 5 else -2.0
        open_p = price
        close_p = price + step
        high_p = max(open_p, close_p) + 0.6
        low_p = min(open_p, close_p) - 0.6
        candles.append(
            {
                "date": f"2025-01-{(idx % 28) + 1:02d}",
                "open": round(open_p, 4),
                "high": round(high_p, 4),
                "low": round(low_p, 4),
                "close": round(close_p, 4),
                "volume": 1_500_000,
            }
        )
        price = close_p
    return candles


def _make_chop_candles(count: int = 240) -> list:
    candles = []
    for idx in range(count):
        base = 100.0 + (0.6 if idx % 2 == 0 else -0.6)
        candles.append(
            {
                "date": f"2025-02-{(idx % 28) + 1:02d}",
                "open": round(base - 0.1, 4),
                "high": round(base + 0.5, 4),
                "low": round(base - 0.5, 4),
                "close": round(base + 0.1, 4),
                "volume": 900_000,
            }
        )
    return candles


class TestStrategyEpochAdaptation(unittest.TestCase):
    def test_reused_strategy_adapts_across_epochs_and_records_rejections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            strategy = DirectionalTrendStrategy()
            snap1 = bus.active_snapshot()

            metrics = _bullish_metrics(adx=21.0, ema20_slope_5d=0.6, earnings_in_days=4.0)
            sig1 = strategy.evaluate_candidate("NVDA", metrics, config_snapshot=snap1)
            self.assertIsInstance(sig1, StrategySignal)
            self.assertEqual(sig1.direction, "LONG")
            self.assertEqual(sig1.config_epoch_id, snap1.config_epoch_id)
            self.assertFalse(sig1.gate_rejections)

            snap2 = bus.commit_patch(
                {
                    "MIN_ADX_TREND": 24.0,
                    "MIN_EMA20_SLOPE_PCT": 0.9,
                    "ENTRY_EARNINGS_BLACKOUT_DAYS": 5.0,
                },
                expected_epoch_id=snap1.config_epoch_id,
                reason="Tighten directional filters",
            )
            sig2 = strategy.evaluate_candidate("NVDA", metrics, config_snapshot=snap2)
            self.assertEqual(sig2.direction, "NEUTRAL")
            self.assertEqual(sig2.config_epoch_id, snap2.config_epoch_id)
            self.assertTrue(sig2.gate_rejections)
            gate_ids = {r["gate_id"] for r in sig2.gate_rejections}
            self.assertIn("adx_trend_gate", gate_ids)
            for rej in sig2.gate_rejections:
                self.assertIn("param_keys", rej)
                self.assertIn("observed", rej)
                self.assertIn("threshold", rej)
                self.assertIn("label", rej)

    def test_conviction_and_short_weights_update_on_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            strategy = DirectionalTrendStrategy()
            snap1 = bus.active_snapshot()
            # sma200 > close so tailwind bonus does not trigger -> 0.70 + 0.22 = 0.92
            no_tailwind_metrics = _bullish_metrics(
                close=150.0, sma200=165.0, adx=22.0, ema20_slope_5d=1.2
            )
            sig1 = strategy.evaluate_candidate("AAPL", no_tailwind_metrics, config_snapshot=snap1)
            self.assertEqual(sig1.direction, "LONG")
            self.assertAlmostEqual(sig1.conviction, 0.92, places=3)

            snap2 = bus.commit_patch(
                {
                    "STRATEGY_BASE_CONVICTION": 0.60,
                    "STRATEGY_MACRO_TAILWIND_BONUS": 0.05,
                    "DIRECTIONAL_MAX_CONVICTION": 0.88,
                    "STRATEGY_SHORT_BASE_CONVICTION": 0.62,
                },
                expected_epoch_id=snap1.config_epoch_id,
            )
            with_tailwind = _bullish_metrics(close=150.0, sma200=130.0, adx=22.0, ema20_slope_5d=1.2)
            sig2 = strategy.evaluate_candidate("AAPL", with_tailwind, config_snapshot=snap2)
            # 0.60 + 0.22 + 0.05 = 0.87 <= 0.88
            self.assertAlmostEqual(sig2.conviction, 0.87, places=3)

            # Broad index qualifies for SHORT; single stock hits cash defense NEUTRAL
            short_metrics = _bullish_metrics(
                close=90.0,
                sma200=120.0,
                ema20=92.0,
                ema50=98.0,
                ema20_slope_5d=-0.8,
                adx=24.0,
                plus_di=11.0,
                minus_di=29.0,
                rsi3=40.0,
                rsi7=42.0,
                rsi14=42.0,
                macd=-0.6,
                macd_signal=-0.2,
            )
            short_sig = strategy.evaluate_candidate("QQQ", short_metrics, config_snapshot=snap2)
            self.assertEqual(short_sig.direction, "SHORT")
            self.assertAlmostEqual(short_sig.conviction, 0.86, places=3)

            single_stock_bear = strategy.evaluate_candidate("TSLA", short_metrics, config_snapshot=snap2)
            self.assertEqual(single_stock_bear.direction, "NEUTRAL")
            self.assertEqual(single_stock_bear.gate_rejections[0]["gate_id"], "dual_track_cash_defense")

    def test_dynamic_screener_metric_gate_and_missing_modes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            strategy = DirectionalTrendStrategy()
            screener = LayaScreener()

            snap1 = bus.add_dynamic_parameter(
                {
                    "key": "MAX_RSI3_ENTRY",
                    "label": "Max RSI3 Entry",
                    "type": "float",
                    "default": 70.0,
                    "min": 10.0,
                    "max": 95.0,
                    "screener_metric_gate": {
                        "metric": "rsi3",
                        "op": "<=",
                        "threshold": 70.0,
                        "on_missing": "reject",
                    },
                },
                expected_epoch_id=bus.active_epoch_id,
            )

            overbought = _bullish_metrics(rsi3=84.0)
            sig_rej = strategy.evaluate_candidate("AMD", overbought, config_snapshot=snap1)
            self.assertEqual(sig_rej.direction, "NEUTRAL")
            self.assertEqual(sig_rej.config_epoch_id, snap1.config_epoch_id)
            self.assertTrue(
                any("MAX_RSI3_ENTRY" in r.get("param_keys", []) for r in (sig_rej.gate_rejections or []))
            )

            scr_rej = screener.evaluate_single("AMD", overbought, config_snapshot=snap1)
            self.assertEqual(scr_rej.bias, "neutral")
            self.assertFalse(screener.passes_filter(scr_rej, config_snapshot=snap1))
            self.assertTrue(
                any("MAX_RSI3_ENTRY" in r.get("param_keys", []) for r in (scr_rej.gate_rejections or []))
            )

            # Non-numeric metric values must not crash; with on_missing="reject", reject gracefully
            for bad_val in ("NaN", "bad_str", float("nan"), None):
                corrupt = _bullish_metrics()
                corrupt["rsi3"] = bad_val
                corrupt_sig = strategy.evaluate_candidate("AMD", corrupt, config_snapshot=snap1)
                self.assertEqual(corrupt_sig.direction, "NEUTRAL")
                self.assertTrue(corrupt_sig.gate_rejections)
                corrupt_scr = screener.evaluate_single("AMD", corrupt, config_snapshot=snap1)
                self.assertEqual(corrupt_scr.bias, "neutral")

            # Loosening dynamic param threshold lets rsi3=84 pass on the same reused instances
            snap2 = bus.commit_patch(
                {"MAX_RSI3_ENTRY": 90.0},
                expected_epoch_id=snap1.config_epoch_id,
            )
            sig_pass = strategy.evaluate_candidate("AMD", overbought, config_snapshot=snap2)
            self.assertEqual(sig_pass.direction, "LONG")
            scr_pass = screener.evaluate_single("AMD", overbought, config_snapshot=snap2)
            self.assertTrue(screener.passes_filter(scr_pass, config_snapshot=snap2))

            # Separate bus testing on_missing="skip_gate" with non-numeric metric
            with tempfile.TemporaryDirectory() as tmp2:
                bus2 = _make_bus(tmp2)
                snap_skip = bus2.add_dynamic_parameter(
                    {
                        "key": "OPTIONAL_BB_WIDTH_CAP",
                        "label": "Optional BB Width Cap",
                        "type": "float",
                        "default": 0.10,
                        "min": 0.01,
                        "max": 1.0,
                        "screener_metric_gate": {
                            "metric": "bb_bandwidth",
                            "op": "<=",
                            "threshold": 0.10,
                            "on_missing": "skip_gate",
                        },
                    },
                    expected_epoch_id=bus2.active_epoch_id,
                )
                missing_m = _bullish_metrics()
                missing_m["bb_bandwidth"] = "not-a-number"
                sig_skip = strategy.evaluate_candidate("MSFT", missing_m, config_snapshot=snap_skip)
                self.assertEqual(sig_skip.direction, "LONG")


class TestScreenerAndFunnelReport(unittest.TestCase):
    def test_laya_and_fallback_screener_epoch_adaptation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            screener = LayaScreener()
            fallback = TechnicalFallbackScreener()
            snap1 = bus.active_snapshot()

            # Regime shifts dynamically under snapshot thresholds
            borderline_regime = _bullish_metrics(adx=26.0, realized_vol=0.55, atr_spike_ratio=1.2)
            self.assertEqual(
                LayaScreener.classify_regime(borderline_regime, config_snapshot=snap1),
                "TRENDING_BREAKOUT",
            )

            snap2 = bus.commit_patch(
                {
                    "REGIME_BREAKOUT_ADX_MIN": 30.0,
                    "HIGH_VOL_MAX_REALIZED_VOL": 0.50,
                    "MIN_ADX_TREND": 23.0,
                    "FALLBACK_SCREENER_MIN_ADX": 24.0,
                    "SCREENER_TOP_N": 1,
                },
                expected_epoch_id=snap1.config_epoch_id,
            )
            self.assertEqual(
                LayaScreener.classify_regime(borderline_regime, config_snapshot=snap2),
                "HIGH_VOL_ERRATIC",
            )

            universe = {
                "NVDA": _bullish_metrics(adx=21.0, realized_vol=0.22),
                "MSFT": _bullish_metrics(adx=28.0, realized_vol=0.20),
                "AAPL": _bullish_metrics(adx=27.0, realized_vol=0.21),
            }
            res_s1 = screener.scan_universe(universe, config_snapshot=snap1)
            res_s2 = screener.scan_universe(universe, config_snapshot=snap2)
            self.assertEqual(len(res_s1), 3)
            self.assertEqual(len(res_s2), 1)
            self.assertEqual(res_s2[0].config_epoch_id, snap2.config_epoch_id)

            fb_s1 = fallback.scan_universe({"NVDA": _bullish_metrics(adx=21.0)}, config_snapshot=snap1)
            fb_s2 = fallback.scan_universe({"NVDA": _bullish_metrics(adx=21.0)}, config_snapshot=snap2)
            self.assertEqual(len(fb_s1), 1)
            self.assertEqual(len(fb_s2), 0)

    def test_cycle_funnel_report_and_offline_simulation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            snap1 = bus.active_snapshot()

            universe = {
                "NVDA": {
                    **_bullish_metrics(adx=22.0, rsi3=60.0),
                    "secret_non_allowlisted": 999.0,
                    "bad_nan": float("nan"),
                },
                "META": {
                    **_bullish_metrics(adx=16.0, rsi3=55.0),
                    "custom_blob": "ignore-me",
                },
                "GOOG": _bullish_metrics(adx=30.0, rsi3=72.0),
            }
            screener = LayaScreener()
            results = [screener.evaluate_single(t, m, config_snapshot=snap1) for t, m in universe.items()]
            passed = screener.scan_universe(universe, config_snapshot=snap1)

            report = build_cycle_funnel_report(
                universe_metrics=universe,
                screener_results=results,
                passed_screener=passed,
                strategy_signals=[r.strategy_signal for r in results if r.strategy_signal is not None],
                debate_outcomes=[SimpleNamespace(consensus_decision="BUY")],
                gate_decisions=[SimpleNamespace(decision="BUY", gate_rejections=[])],
                executed_orders=["NVDA"],
                config_snapshot=snap1,
            )
            self.assertIsInstance(report, CycleFunnelReport)
            self.assertEqual(report.config_epoch_id, snap1.config_epoch_id)
            self.assertEqual(report.stage_counts["universe"], 3)
            self.assertEqual(report.stage_counts["screened"], 2)
            self.assertEqual(report.stage_counts["executed"], 1)

            # _compact_indicators restricted strictly to ALLOWLISTED_SCREENER_METRICS
            for ticker, row in report._compact_indicators.items():
                self.assertNotIn("secret_non_allowlisted", row, msg=ticker)
                self.assertNotIn("bad_nan", row, msg=ticker)
                self.assertNotIn("custom_blob", row, msg=ticker)
                for k in row:
                    self.assertIn(k, ALLOWLISTED_SCREENER_METRICS)

            adx_stats = report.metric_percentiles["adx"]
            self.assertEqual(adx_stats["count"], 3)
            self.assertEqual(adx_stats["min"], 16.0)
            self.assertEqual(adx_stats["median"], 22.0)
            self.assertEqual(adx_stats["max"], 30.0)
            self.assertLessEqual(adx_stats["p25"], adx_stats["median"])
            self.assertLessEqual(adx_stats["median"], adx_stats["p75"])
            self.assertTrue(report.gate_breakdown)
            json.dumps(report.to_dict())

            # Offline diff simulation on compact indicators
            looser_snap = bus.commit_patch(
                {"MIN_ADX_TREND": 14.0},
                expected_epoch_id=snap1.config_epoch_id,
            )
            sim = simulate_screener_gate_impact(report._compact_indicators, looser_snap)
            self.assertEqual(sim["config_epoch_id"], looser_snap.config_epoch_id)
            self.assertEqual(sim["universe_count"], 3)
            self.assertEqual(sim["passed_count"], 3)

    def test_cycle_funnel_report_fallback_from_results_and_sell_entry_debate(self):
        screener = LayaScreener()
        results = [
            screener.evaluate_single("NVDA", _bullish_metrics(adx=22.0)),
            screener.evaluate_single("AAPL", _bullish_metrics(adx=28.0)),
        ]
        # universe_metrics=None falls back to extracting metrics per item in screener_results
        report = build_cycle_funnel_report(
            universe_metrics=None,
            screener_results=results,
            debate_outcomes=[
                SimpleNamespace(consensus_decision="BUY_ENTRY", approved=False),
                SimpleNamespace(consensus_decision="SELL_ENTRY", approved=False),
                SimpleNamespace(consensus_decision="DO_NOT_TRADE", approved=False),
            ],
        )
        self.assertEqual(set(report._compact_indicators.keys()), {"NVDA", "AAPL"})
        self.assertEqual(report.metric_percentiles["adx"]["min"], 22.0)
        self.assertEqual(report.metric_percentiles["adx"]["max"], 28.0)
        self.assertEqual(report.stage_counts["debate"], 2)

        # Also verifies strategy_signals fallback when screener_results=None
        signals = [r.strategy_signal for r in results if r.strategy_signal is not None]
        sig_report = build_cycle_funnel_report(universe_metrics=None, strategy_signals=signals)
        self.assertEqual(set(sig_report._compact_indicators.keys()), {"NVDA", "AAPL"})
        self.assertEqual(sig_report.metric_percentiles["adx"]["count"], 2)


class TestRiskManagerDirectEpochAdaptation(unittest.TestCase):
    def test_risk_manager_direct_sizing_uses_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            rm = RiskManager()
            snap1 = bus.active_snapshot()
            metrics = _bullish_metrics(adx=21.0)
            with patch("trading_system.data.trading212_client.MarketDataProvider.get_quote", return_value={"price": 25.0, "currency": "EUR"}):
                card1 = rm.generate_decision_card(
                    "NVDA",
                    "NVD3l_EQ",
                    "LONG",
                    metrics,
                    debate_outcome=None,
                    account_equity=10000.0,
                    config_snapshot=snap1,
                )
                self.assertEqual(card1.config_epoch_id, snap1.config_epoch_id)
                self.assertEqual(card1.decision, "BUY_ENTRY")


class TestRiskManagerEpochAndStatePreservation(unittest.TestCase):
    def test_enable_5x_fallback_bug_fixed_default_false(self):
        rm = RiskManager()
        metrics = _bullish_metrics(adx=28.0)
        debate = SimpleNamespace(
            consensus_decision="BUY_ENTRY",
            conviction_score=0.88,
            synthesis="Strong trend",
        )
        # Mock snapshot missing ENABLE_5X_LEVERAGE attribute -> must default False (3x), NEVER 5x
        sparse_snapshot = SimpleNamespace(config_epoch_id="epoch-sparse")
        with patch("trading_system.config.settings.ENABLE_5X_LEVERAGE", True), patch(
            "trading_system.data.trading212_client.MarketDataProvider.get_quote",
            return_value={"price": 25.0, "currency": "USD"},
        ):
            card_default = rm.generate_decision_card(
                "NVDA",
                "NVD3l_EQ",
                "LONG",
                metrics,
                debate,
                account_equity=10000.0,
                config_snapshot=sparse_snapshot,
            )
            self.assertEqual(card_default.leverage, 3)

        # Real default ConfigSnapshot also keeps ENABLE_5X_LEVERAGE=False -> QQQ uses 3x QQQ3m_EQ
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "trading_system.data.trading212_client.MarketDataProvider.get_quote",
            return_value={"price": 25.0, "currency": "EUR"},
        ):
            bus = _make_bus(tmpdir)
            snap1 = bus.active_snapshot()
            self.assertFalse(snap1.ENABLE_5X_LEVERAGE)
            card_3x = rm.generate_decision_card(
                "QQQ",
                "QQQ3m_EQ",
                "LONG",
                metrics,
                debate,
                account_equity=10000.0,
                config_snapshot=snap1,
            )
            self.assertEqual(card_3x.leverage, 3)
            self.assertEqual(card_3x.trading212_ticker, "QQQ3m_EQ")

            snap_5x = bus.commit_patch(
                {
                    "ENABLE_5X_LEVERAGE": True,
                    "PREFER_5X_LEVERAGE": True,
                    "MAX_LEVERAGE": 5,
                },
                expected_epoch_id=snap1.config_epoch_id,
            )
            card_5x = rm.generate_decision_card(
                "QQQ",
                "QQQ3m_EQ",
                "LONG",
                metrics,
                debate,
                account_equity=10000.0,
                config_snapshot=snap_5x,
            )
            self.assertEqual(card_5x.leverage, 5)
            self.assertEqual(card_5x.trading212_ticker, "5QQEl_EQ")

    def test_risk_manager_epoch_updates_preserve_runtime_state_and_classmethod(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "trading_system.data.trading212_client.MarketDataProvider.get_quote",
            return_value={"price": 25.0, "currency": "USD"},
        ):
            bus = _make_bus(tmpdir)
            rm = RiskManager()
            snap1 = bus.active_snapshot()

            # Mutate runtime state on the reused RiskManager instance
            rm.register_loss_exit("NVDA", cooldown_seconds=120.0, config_snapshot=snap1)
            rm.check_circuit_breaker(current_equity=10000.0, session_start_equity=10000.0, config_snapshot=snap1)

            snap2 = bus.commit_patch(
                {
                    "ATR_SL_MULTIPLIER": 3.0,
                    "ETP_MIN_SL_PCT": 0.03,
                    "ETP_MAX_SL_PCT": 0.07,
                    "MAX_RISK_PER_TRADE": 0.01,
                    "MAX_BID_ASK_SPREAD_PCT": 0.005,
                    "DAILY_MAX_DRAWDOWN_PCT": 0.01,
                    "TSL_DECAY_DAY1_MULT": 2.50,
                    "TSL_DECAY_RUNNER_MULT": 2.25,
                },
                expected_epoch_id=snap1.config_epoch_id,
            )

            # Runtime cooldown state must survive epoch switch
            in_cd, remaining = rm.is_in_loss_cooldown("NVDA")
            self.assertTrue(in_cd)
            self.assertGreater(remaining, 0.0)

            # Bounds & sizing adapt to snap2
            min_sl, max_sl = rm.get_etp_sl_bounds(config_snapshot=snap2)
            self.assertAlmostEqual(min_sl, 0.03)
            self.assertAlmostEqual(max_sl, 0.07)

            debate = SimpleNamespace(consensus_decision="BUY_ENTRY", conviction_score=0.8, synthesis="OK")
            card = rm.generate_decision_card(
                "MSFT",
                "MSF3l_EQ",
                "LONG",
                _bullish_metrics(close=100.0, atr=3.0),
                debate,
                account_equity=10000.0,
                config_snapshot=snap2,
            )
            self.assertIsInstance(card, DecisionCard)
            self.assertEqual(card.config_epoch_id, snap2.config_epoch_id)

            # Tighter spread gate under snap2 rejects spread=0.01 with structured gate_rejections
            ok_spread, reason_spread = rm.validate_trade(
                card,
                account_cash=10000.0,
                existing_positions=[],
                spread_pct=0.01,
                account_equity=10000.0,
                config_snapshot=snap2,
            )
            self.assertFalse(ok_spread)
            self.assertIn("spread", reason_spread.lower())
            self.assertTrue(card.gate_rejections)
            self.assertEqual(card.gate_rejections[-1]["gate_id"], "spread_gate")

            # Tighter DAILY_MAX_DRAWDOWN_PCT=1% trips circuit breaker at 1.5% loss
            halted, _ = rm.check_circuit_breaker(current_equity=9850.0, config_snapshot=snap2)
            self.assertTrue(halted)
            self.assertTrue(rm.is_circuit_breaker_tripped)

            # @classmethod preservation with & without config_snapshot
            self.assertAlmostEqual(
                RiskManager.get_decay_atr_multiplier(session_count=1, is_runner=False, config_snapshot=snap2),
                2.50,
            )
            self.assertAlmostEqual(
                RiskManager.get_decay_atr_multiplier(session_count=1, is_runner=True, config_snapshot=snap2),
                2.25,
            )
            self.assertAlmostEqual(
                RiskManager.get_decay_atr_multiplier(session_count=1, is_runner=False, config_snapshot=None),
                2.20,
            )
            ratchet = RiskManager.calculate_progressive_trailing_stop(
                entry_price=100.0,
                current_price=102.0,
                current_sl=95.0,
                direction="LONG",
                ticker="QQQ3l_EQ",
                config_snapshot=snap2,
            )
            self.assertGreaterEqual(ratchet, 100.0)

    def test_dict_config_snapshot_and_trailing_stop_etp_min_sl_alignment(self):
        dict_snap = {
            "config_epoch_id": "epoch-dict-7",
            "TSL_DECAY_DAY1_MULT": 2.40,
            "TSL_DECAY_RUNNER_MULT": 2.35,
            "ETP_MIN_SL_PCT": 0.035,
            "ETP_MAX_SL_PCT": 0.090,
            "MIN_ADX_TREND": 30.0,
        }
        self.assertAlmostEqual(
            RiskManager.get_decay_atr_multiplier(1, False, dict_snap),
            2.40,
        )
        self.assertAlmostEqual(
            RiskManager.get_decay_atr_multiplier(1, True, dict_snap),
            2.35,
        )

        rm = RiskManager()
        self.assertEqual(rm.get_etp_sl_bounds(dict_snap), (0.035, 0.090))

        # Trailing stop honors ETP_MIN_SL_PCT when snapshot is provided vs 0.02 when None
        self.assertAlmostEqual(
            rm.calculate_trailing_stop(
                current_price=100.0,
                peak_price=100.0,
                atr=0.5,
                current_tsl=90.0,
                config_snapshot=dict_snap,
            ),
            96.50,
        )
        self.assertAlmostEqual(
            rm.calculate_trailing_stop(
                current_price=100.0,
                peak_price=100.0,
                atr=0.5,
                current_tsl=90.0,
                config_snapshot=None,
            ),
            98.00,
        )

        # Dict snapshot also works cleanly in strategy evaluation
        sig = DirectionalTrendStrategy().evaluate_candidate(
            "NVDA", _bullish_metrics(adx=25.0), config_snapshot=dict_snap
        )
        self.assertEqual(sig.direction, "NEUTRAL")
        self.assertEqual(sig.config_epoch_id, "epoch-dict-7")


class TestStrategyImprovementAndBacktestEpoch(unittest.TestCase):
    def test_strategy_improvement_agent_directional_audit_and_atomic_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            snap1 = bus.commit_patch(
                {"MIN_ADX_TREND": 24.0, "SCREENER_MAX_CATALYST_RISK": 0.25},
                expected_epoch_id=bus.active_epoch_id,
            )
            history_file = Path(tmpdir) / "audit_history.jsonl"

            agent = StrategyImprovementAgent(
                zero_yield_threshold=3,
                audit_history_path=history_file,
                max_persisted_audits=5,
            )

            low_adx_universe = {
                "SPY": _bullish_metrics(adx=15.0, realized_vol=0.14),
                "QQQ": _bullish_metrics(adx=17.0, realized_vol=0.16),
            }
            audit1 = agent.record_cycle_yield(0, low_adx_universe, config_snapshot=snap1)
            self.assertIsInstance(audit1, ImprovementAudit)
            self.assertEqual(audit1.active_strategy, "vwap")
            self.assertEqual(audit1.config_epoch_id, snap1.config_epoch_id)

            agent.record_cycle_yield(0, low_adx_universe, config_snapshot=snap1)
            audit3 = agent.record_cycle_yield(0, low_adx_universe, config_snapshot=snap1)
            self.assertEqual(audit3.health_status, "STARVED")
            self.assertTrue(audit3.parameter_recommendations)
            rec_keys = {r["param_key"] for r in audit3.parameter_recommendations}
            self.assertIn("MIN_ADX_TREND", rec_keys)
            self.assertTrue(any("#/config?key=MIN_ADX_TREND" in r for r in audit3.recommendations))

            # Verify atomic persistence & corrupt line tolerance across reload
            self.assertTrue(history_file.exists())
            with open(history_file, "a", encoding="utf-8") as fh:
                fh.write("{corrupt json line\n")

            reloaded = StrategyImprovementAgent(
                zero_yield_threshold=3,
                audit_history_path=history_file,
                max_persisted_audits=5,
            )
            self.assertEqual(len(reloaded.audit_history), 3)
            self.assertEqual(reloaded.audit_history[-1].config_epoch_id, snap1.config_epoch_id)

            # Test backward compatibility with legacy candidate_rsi3_threshold kwarg
            sim_res = reloaded.verify_parameter_adaptation(
                symbol="QQQ",
                candidate_candles=_make_trending_candles(240),
                candidate_rsi3_threshold=25.0,
                candidate_adx_threshold=16.0,
                config_snapshot=snap1,
            )
            self.assertEqual(sim_res["symbol"], "QQQ")
            self.assertEqual(sim_res["strategy"], "vwap")
            self.assertIn("candidate_win_rate", sim_res)

    def test_backtest_engine_directional_trend_uses_wilder_adx_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = _make_bus(tmpdir)
            snap_loose = bus.commit_patch(
                {"MIN_ADX_TREND": 14.0, "SCREENER_MAX_CATALYST_RISK": 0.30},
                expected_epoch_id=bus.active_epoch_id,
            )
            snap_impossible = bus.commit_patch(
                {"MIN_ADX_TREND": 50.0, "SCREENER_MAX_CATALYST_RISK": 0.30},
                expected_epoch_id=snap_loose.config_epoch_id,
            )

            candles = _make_trending_candles(240)
            engine = BacktestEngine(initial_capital=10000.0)

            res_loose = engine.run(
                "QQQ", candles, strategy_filter="directional_trend", config_snapshot=snap_loose
            )
            res_tight = engine.run(
                "QQQ", candles, strategy_filter="directional_trend", config_snapshot=snap_impossible
            )
            self.assertGreater(res_loose.total_trades, 0)
            self.assertGreaterEqual(res_loose.total_trades, res_tight.total_trades)
            self.assertEqual(res_tight.total_trades, 0)

            # Sideways chop produces 0 trades under directional_trend
            chop_res = engine.run(
                "SPY",
                _make_chop_candles(240),
                strategy_filter="directional_trend",
                config_snapshot=snap_loose,
            )
            self.assertEqual(chop_res.total_trades, 0)


if __name__ == "__main__":
    unittest.main()
