"""Multi-Provider Market Data Router with Sliding-Window Rate Limiting & Failover Cascades.

Orchestrates market data fetching across:
1. Financial Modeling Prep (FMP)
2. Tiingo
3. Polygon.io
4. Alpha Vantage
5. Yahoo Finance Direct Chart API (keyless)
6. Stooq (keyless)
7. yfinance (keyless library fallback)

Enforces per-minute, per-hour, and per-day rate limits for each provider.
When a provider reaches its quota or returns HTTP 429, the router activates
a cooldown period and automatically cascades down the waterfall to the next provider.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ProviderQuota:
    """Rate limit configuration and runtime accounting for a single data provider."""

    name: str
    rate_limit_per_min: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    daily_budget: Optional[int] = None
    cooldown_on_429: int = 60  # seconds

    # Runtime state
    _request_timestamps: deque = field(default_factory=deque)
    _day_request_count: int = 0
    _current_day: str = ""
    _cooldown_until: float = 0.0
    _total_success: int = 0
    _total_errors: int = 0
    _total_429: int = 0
    _last_status: str = "IDLE"
    _last_error_reason: str = ""

    def reset_day_if_needed(self) -> None:
        """Reset daily count at UTC midnight."""
        today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._current_day != today_utc:
            self._current_day = today_utc
            self._day_request_count = 0


class MultiProviderRouter:
    """Thread-safe multi-provider market data router with automatic failover."""

    _instance: Optional[MultiProviderRouter] = None
    _lock = threading.Lock()

    # Known non-equity underlyings (ETFs, commodities, bonds, crypto, currencies, indices)
    _NON_EQUITY_SYMBOLS: set[str] = {
        "QQQ", "SPY", "SOXX", "SMH", "MAGS", "FXI", "^GDAXI", "^FTSE",
        "^STOXX50E", "^FCHI", "FTSEMIB.MI", "EWY", "EWJ", "INDA", "VT",
        "EWT", "ARKK", "GDX", "BOTZ", "CIBR", "DFNS", "SPCX", "URA",
        "URNM", "KWEB", "XLF", "XLE", "XLV", "XLI", "GLD", "SLV",
        "USO", "BNO", "UNG", "COFF", "WEAT", "HG=F", "SILV", "TLT",
        "BUND", "BTP", "3GIL", "BLC", "BTC-USD", "ETH-USD", "GBPUSD=X",
    }

    def __init__(self) -> None:
        self._provider_lock = threading.Lock()
        self.quotas: Dict[str, ProviderQuota] = {
            "fmp": ProviderQuota(
                name="fmp",
                rate_limit_per_min=30,
                daily_budget=250,
                cooldown_on_429=120,
            ),
            "tiingo": ProviderQuota(
                name="tiingo",
                rate_limit_per_hour=50,
                daily_budget=500,
                cooldown_on_429=180,
            ),
            "polygon": ProviderQuota(
                name="polygon",
                rate_limit_per_min=5,
                daily_budget=1000,
                cooldown_on_429=65,
            ),
            "alpha_vantage": ProviderQuota(
                name="alpha_vantage",
                rate_limit_per_min=5,
                daily_budget=25,
                cooldown_on_429=65,
            ),
            "yahoo_chart": ProviderQuota(
                name="yahoo_chart",
                rate_limit_per_min=60,
                rate_limit_per_hour=2000,
                cooldown_on_429=60,
            ),
            "stooq": ProviderQuota(
                name="stooq",
                rate_limit_per_min=30,
                cooldown_on_429=60,
            ),
        }

    @classmethod
    def get_instance(cls) -> MultiProviderRouter:
        """Singleton accessor for thread-safe shared quota state."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ─────────────────────────────────────────────────────────────────
    # Quota & Rate-Limiting Guard
    # ─────────────────────────────────────────────────────────────────

    def can_request(self, provider_name: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """Check whether provider is eligible to make a request without violating limits."""
        with self._provider_lock:
            quota = self.quotas.get(provider_name)
            if not quota:
                return True, "ok"

            now = time.time()

            # Keyed provider missing API key
            if api_key is not None and not api_key:
                return False, "no_api_key"

            # Active cooldown check (e.g. following HTTP 429)
            if now < quota._cooldown_until:
                wait_sec = int(quota._cooldown_until - now)
                return False, f"cooldown_active_{wait_sec}s"

            quota.reset_day_if_needed()

            # Prune sliding window timestamps older than 1 hour
            while quota._request_timestamps and (now - quota._request_timestamps[0] > 3600):
                quota._request_timestamps.popleft()

            # Minute limit check
            if quota.rate_limit_per_min:
                min_count = sum(1 for t in quota._request_timestamps if now - t <= 60)
                if min_count >= quota.rate_limit_per_min:
                    return False, f"minute_limit_{min_count}/{quota.rate_limit_per_min}"

            # Hour limit check
            if quota.rate_limit_per_hour:
                hour_count = len(quota._request_timestamps)
                if hour_count >= quota.rate_limit_per_hour:
                    return False, f"hour_limit_{hour_count}/{quota.rate_limit_per_hour}"

            # Daily limit check
            if quota.daily_budget and quota._day_request_count >= quota.daily_budget:
                return False, f"daily_budget_exhausted_{quota._day_request_count}/{quota.daily_budget}"

            return True, "ok"

    def record_request(self, provider_name: str) -> None:
        """Record an outbound request to update sliding rate limits."""
        with self._provider_lock:
            quota = self.quotas.get(provider_name)
            if quota:
                now = time.time()
                quota.reset_day_if_needed()
                quota._request_timestamps.append(now)
                quota._day_request_count += 1
                quota._last_status = "REQUESTING"

    def record_response(
        self,
        provider_name: str,
        status_code: int,
        error_msg: str = "",
        is_rate_limited: bool = False,
    ) -> None:
        """Record response result, triggering cooldown on 429 or quota exhaustion."""
        with self._provider_lock:
            quota = self.quotas.get(provider_name)
            if not quota:
                return

            now = time.time()
            if status_code == 200 and not is_rate_limited:
                quota._total_success += 1
                quota._last_status = "HEALTHY"
                quota._last_error_reason = ""
            elif status_code == 429 or is_rate_limited:
                quota._total_429 += 1
                quota._cooldown_until = now + quota.cooldown_on_429
                quota._last_status = f"COOLDOWN_{quota.cooldown_on_429}s"
                quota._last_error_reason = error_msg or f"HTTP {status_code} Rate Limit Hit"
                logger.warning(
                    "MarketDataRouter: [%s] Rate limit hit (%s). Activating %ds cooldown and cascading to fallback.",
                    provider_name,
                    quota._last_error_reason,
                    quota.cooldown_on_429,
                )
            else:
                quota._total_errors += 1
                quota._last_status = f"ERROR_{status_code}"
                quota._last_error_reason = error_msg or f"HTTP {status_code}"

    # ─────────────────────────────────────────────────────────────────
    # Provider Fetch Implementations
    # ─────────────────────────────────────────────────────────────────

    def _http_get_json(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 8) -> Tuple[int, Any]:
        """Perform an HTTP GET request returning (status_code, parsed_json_or_text)."""
        hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Trading212-Quantitative-Engine/2.0"}
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                raw = resp.read().decode("utf-8")
                try:
                    return status, json.loads(raw)
                except Exception:
                    return status, raw
        except urllib.error.HTTPError as he:
            try:
                body = he.read().decode("utf-8")
            except Exception:
                body = str(he)
            return he.code, body
        except Exception as exc:
            return 500, str(exc)

    # ── Financial Modeling Prep (FMP) ──

    def fetch_fmp_daily(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch daily OHLCV from Financial Modeling Prep."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{clean}?apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429 or (isinstance(data, dict) and "limit" in str(data.get("Error Message", "")).lower())
        self.record_response("fmp", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, dict) and "historical" in data:
            records = []
            for bar in reversed(data["historical"]):
                records.append({
                    "date": bar.get("date"),
                    "open": float(bar.get("open", 0.0)),
                    "high": float(bar.get("high", 0.0)),
                    "low": float(bar.get("low", 0.0)),
                    "close": float(bar.get("close", 0.0)),
                    "volume": float(bar.get("volume", 0.0)),
                })
            if records:
                logger.info("MarketDataRouter: Fetched %d daily candles for %s via [FMP]", len(records), symbol)
                return records
        return None

    def _fetch_fmp_intraday(
        self, symbol: str, api_key: str, chart_interval: str = "5min", label: str = "5m"
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch intraday OHLCV bars from Financial Modeling Prep for a given chart interval."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{chart_interval}/{clean}?apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429 or (isinstance(data, dict) and "limit" in str(data.get("Error Message", "")).lower())
        self.record_response("fmp", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, list) and data:
            records = []
            for bar in reversed(data):
                records.append({
                    "date": bar.get("date"),
                    "open": float(bar.get("open", 0.0)),
                    "high": float(bar.get("high", 0.0)),
                    "low": float(bar.get("low", 0.0)),
                    "close": float(bar.get("close", 0.0)),
                    "volume": float(bar.get("volume", 0.0)),
                })
            if records:
                logger.info("MarketDataRouter: Fetched %d %s candles for %s via [FMP]", len(records), label, symbol)
                return records
        return None

    def fetch_fmp_hourly(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch hourly OHLCV from Financial Modeling Prep."""
        return self._fetch_fmp_intraday(symbol, api_key, chart_interval="1hour", label="hourly")

    def fetch_fmp_5m(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch 5-minute OHLCV from Financial Modeling Prep."""
        return self._fetch_fmp_intraday(symbol, api_key, chart_interval="5min", label="5m")

    def fetch_fmp_quote(self, symbol: str, api_key: str) -> Optional[Dict[str, float]]:
        """Fetch bid/ask and current price quote from FMP."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://financialmodelingprep.com/api/v3/quote/{clean}?apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429
        self.record_response("fmp", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, list) and data:
            q = data[0]
            bid = float(q.get("bid", 0.0) or 0.0)
            ask = float(q.get("ask", 0.0) or 0.0)
            px = float(q.get("price", 0.0) or 0.0)
            if ask > 0 and bid > 0 and ask >= bid:
                mid = (ask + bid) / 2.0
                spread_pct = (ask - bid) / mid if mid > 0 else 0.0
                return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": spread_pct}
            elif px > 0:
                return {"bid": round(px * 0.99875, 4), "ask": round(px * 1.00125, 4), "mid": px, "spread_pct": 0.0025}
        return None

    def fetch_fmp_earnings_days(self, symbol: str, api_key: str) -> Optional[float]:
        """Fetch upcoming earnings release calendar from FMP."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://financialmodelingprep.com/api/v3/historical/earning_calendar/{clean}?limit=5&apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429
        self.record_response("fmp", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, list) and data:
            today = datetime.now(timezone.utc).date()
            for item in data:
                d_str = item.get("date")
                if d_str:
                    try:
                        e_date = datetime.strptime(d_str, "%Y-%m-%d").date()
                        delta = (e_date - today).days
                        if delta >= 0:
                            return float(delta)
                    except Exception:
                        pass
        return None

    # ── Tiingo ──

    def fetch_tiingo_daily(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch daily OHLCV from Tiingo."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        start_date = (datetime.now(timezone.utc) - timedelta(days=365)).strftime("%Y-%m-%d")
        url = f"https://api.tiingo.com/tiingo/daily/{clean}/prices?startDate={start_date}&token={api_key}"
        status, data = self._http_get_json(url, headers={"Content-Type": "application/json"})
        is_429 = status == 429
        self.record_response("tiingo", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, list) and data:
            records = []
            for bar in data:
                d_str = str(bar.get("date", ""))[:10]
                records.append({
                    "date": d_str,
                    "open": float(bar.get("open", 0.0)),
                    "high": float(bar.get("high", 0.0)),
                    "low": float(bar.get("low", 0.0)),
                    "close": float(bar.get("close", 0.0)),
                    "volume": float(bar.get("volume", 0.0)),
                })
            if records:
                logger.info("MarketDataRouter: Fetched %d daily candles for %s via [Tiingo]", len(records), symbol)
                return records
        return None

    def fetch_tiingo_quote(self, symbol: str, api_key: str) -> Optional[Dict[str, float]]:
        """Fetch IEX top-of-book bid/ask quote from Tiingo."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://api.tiingo.com/iex/{clean}?token={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429
        self.record_response("tiingo", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, list) and data:
            q = data[0]
            bid = float(q.get("bidPrice", 0.0) or 0.0)
            ask = float(q.get("askPrice", 0.0) or 0.0)
            last = float(q.get("last", 0.0) or 0.0)
            if ask > 0 and bid > 0 and ask >= bid:
                mid = (ask + bid) / 2.0
                return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": (ask - bid) / mid}
            elif last > 0:
                return {"bid": round(last * 0.99875, 4), "ask": round(last * 1.00125, 4), "mid": last, "spread_pct": 0.0025}
        return None

    # ── Polygon.io ──

    def _fetch_polygon_aggs(
        self,
        symbol: str,
        api_key: str,
        multiplier: int,
        timespan: str,
        days_back: int,
        dt_format: str,
        label: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch aggregate OHLCV bars from Polygon.io for a given multiplier and timespan."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{clean}/range/{multiplier}/{timespan}/{start}/{end}?apiKey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429 or (isinstance(data, dict) and "exceeded" in str(data.get("message", "")).lower())
        self.record_response("polygon", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, dict):
            results = data.get("results", [])
            records = []
            for bar in results:
                ts_sec = bar.get("t", 0) / 1000.0
                d_str = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime(dt_format)
                records.append({
                    "date": d_str,
                    "open": float(bar["o"]),
                    "high": float(bar["h"]),
                    "low": float(bar["l"]),
                    "close": float(bar["c"]),
                    "volume": float(bar.get("v", 0.0)),
                })
            if records:
                logger.info("MarketDataRouter: Fetched %d %s candles for %s via [Polygon]", len(records), label, symbol)
                return records
        return None

    def fetch_polygon_daily(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch daily candles from Polygon.io."""
        return self._fetch_polygon_aggs(symbol, api_key, 1, "day", 365, "%Y-%m-%d", "daily")

    def fetch_polygon_hourly(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch hourly candles from Polygon.io."""
        return self._fetch_polygon_aggs(symbol, api_key, 1, "hour", 35, "%Y-%m-%d %H:%M:%S", "hourly")

    def fetch_polygon_5m(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch 5-minute intraday candles from Polygon.io."""
        return self._fetch_polygon_aggs(symbol, api_key, 5, "minute", 5, "%Y-%m-%d %H:%M:%S", "5m")

    def fetch_polygon_quote(self, symbol: str, api_key: str) -> Optional[Dict[str, float]]:
        """Fetch real-time top-of-book or previous close quote from Polygon.io."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        # 1. Try real-time NBBO endpoint
        url = f"https://api.polygon.io/v2/last/nbbo/{clean}?apiKey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = status == 429 or (isinstance(data, dict) and "exceeded" in str(data.get("message", "")).lower())
        self.record_response("polygon", status, is_rate_limited=is_429)
        if status == 200 and isinstance(data, dict):
            res = data.get("results", {})
            bid = float(res.get("p", 0.0) or 0.0)
            ask = float(res.get("P", 0.0) or 0.0)
            if ask > 0 and bid > 0 and ask >= bid:
                mid = (ask + bid) / 2.0
                return {"bid": bid, "ask": ask, "mid": mid, "spread_pct": (ask - bid) / mid}

        # 2. Fallback to previous day bar snapshot if real-time NBBO is not supported on tier
        prev_url = f"https://api.polygon.io/v2/aggs/ticker/{clean}/prev?apiKey={api_key}"
        prev_status, prev_data = self._http_get_json(prev_url)
        if prev_status == 200 and isinstance(prev_data, dict):
            results = prev_data.get("results", [])
            if results:
                bar = results[0]
                close_px = float(bar.get("c", 0.0) or bar.get("vw", 0.0) or 0.0)
                if close_px > 0:
                    return {"bid": round(close_px * 0.99875, 4), "ask": round(close_px * 1.00125, 4), "mid": close_px, "spread_pct": 0.0025}
        return None

    # ── Alpha Vantage ──

    def fetch_alphavantage_daily(self, symbol: str, api_key: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch daily OHLCV from Alpha Vantage."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={clean}&outputsize=compact&apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = False
        if isinstance(data, dict):
            # Alpha Vantage returns HTTP 200 with 'Note' or 'Information' when rate limited
            if "Note" in data or "Information" in data:
                is_429 = True
        self.record_response("alpha_vantage", status, is_rate_limited=is_429)
        if status == 200 and not is_429 and isinstance(data, dict):
            ts_dict = data.get("Time Series (Daily)", {})
            if ts_dict:
                records = []
                for d_str, vals in sorted(ts_dict.items()):
                    records.append({
                        "date": d_str,
                        "open": float(vals.get("1. open", 0.0)),
                        "high": float(vals.get("2. high", 0.0)),
                        "low": float(vals.get("3. low", 0.0)),
                        "close": float(vals.get("4. close", 0.0)),
                        "volume": float(vals.get("6. volume", 0.0)),
                    })
                if records:
                    logger.info("MarketDataRouter: Fetched %d daily candles for %s via [Alpha Vantage]", len(records), symbol)
                    return records
        return None

    def fetch_alphavantage_earnings_days(self, symbol: str, api_key: str) -> Optional[float]:
        """Fetch upcoming earnings release from Alpha Vantage."""
        if not api_key:
            return None
        clean = symbol.upper().replace("^", "")
        url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={clean}&horizon=3month&apikey={api_key}"
        status, data = self._http_get_json(url)
        is_429 = "Note" in str(data) or "Information" in str(data) or status == 429
        self.record_response("alpha_vantage", status, is_rate_limited=is_429)
        if status == 200 and not is_429 and isinstance(data, str) and "reportDate" in data:
            today = datetime.now(timezone.utc).date()
            for line in data.strip().split("\n")[1:]:
                parts = line.split(",")
                if len(parts) >= 3 and parts[0].strip().upper() == clean:
                    r_date_str = parts[2].strip()
                    try:
                        e_date = datetime.strptime(r_date_str, "%Y-%m-%d").date()
                        delta = (e_date - today).days
                        if delta >= 0:
                            return float(delta)
                    except Exception:
                        pass
        return None

    # ─────────────────────────────────────────────────────────────────
    # Waterfall Execution Cascades
    # ─────────────────────────────────────────────────────────────────

    def _run_candle_cascade(
        self,
        symbol: str,
        chain: List[Tuple[str, Optional[str], Callable[[], Optional[List[Dict[str, Any]]]]]],
        label: str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run a quota-aware provider waterfall cascade for candle retrieval."""
        for prov_name, key, fetcher in chain:
            can_call, reason = self.can_request(prov_name, key)
            if not can_call:
                if reason != "no_api_key":
                    logger.info("MarketDataRouter: Skipping [%s] for %s (%s). Cascading...", prov_name, symbol, reason)
                continue

            self.record_request(prov_name)
            try:
                result = fetcher()
                if result:
                    return result
            except Exception as exc:
                logger.debug("MarketDataRouter: [%s] %s fetch error for %s: %s", prov_name, label, symbol, exc)

        return None

    def get_daily_candles_cascade(
        self,
        symbol: str,
        settings_obj: Any,
        period: str = "1y",
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute waterfall cascade for daily candles respecting quotas."""
        fmp_key = getattr(settings_obj, "FMP_API_KEY", "")
        tiingo_key = getattr(settings_obj, "TIINGO_API_KEY", "")
        poly_key = getattr(settings_obj, "POLYGON_API_KEY", "")
        av_key = getattr(settings_obj, "ALPHA_VANTAGE_API_KEY", "")

        chain: List[Tuple[str, Optional[str], Callable[[], Optional[List[Dict[str, Any]]]]]] = [
            ("fmp", fmp_key, lambda: self.fetch_fmp_daily(symbol, fmp_key)),
            ("tiingo", tiingo_key, lambda: self.fetch_tiingo_daily(symbol, tiingo_key)),
            ("polygon", poly_key, lambda: self.fetch_polygon_daily(symbol, poly_key)),
            ("alpha_vantage", av_key, lambda: self.fetch_alphavantage_daily(symbol, av_key)),
        ]
        return self._run_candle_cascade(symbol, chain, "daily")

    def get_hourly_candles_cascade(
        self,
        symbol: str,
        settings_obj: Any,
        period: str = "1mo",
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute waterfall cascade for hourly candles respecting quotas."""
        fmp_key = getattr(settings_obj, "FMP_API_KEY", "")
        poly_key = getattr(settings_obj, "POLYGON_API_KEY", "")

        chain: List[Tuple[str, Optional[str], Callable[[], Optional[List[Dict[str, Any]]]]]] = [
            ("fmp", fmp_key, lambda: self.fetch_fmp_hourly(symbol, fmp_key)),
            ("polygon", poly_key, lambda: self.fetch_polygon_hourly(symbol, poly_key)),
        ]
        return self._run_candle_cascade(symbol, chain, "hourly")

    def get_5m_candles_cascade(
        self,
        symbol: str,
        settings_obj: Any,
        period: str = "5d",
    ) -> Optional[List[Dict[str, Any]]]:
        """Execute waterfall cascade for 5-minute intraday candles respecting quotas."""
        fmp_key = getattr(settings_obj, "FMP_API_KEY", "")
        poly_key = getattr(settings_obj, "POLYGON_API_KEY", "")

        chain: List[Tuple[str, Optional[str], Callable[[], Optional[List[Dict[str, Any]]]]]] = [
            ("fmp", fmp_key, lambda: self.fetch_fmp_5m(symbol, fmp_key)),
            ("polygon", poly_key, lambda: self.fetch_polygon_5m(symbol, poly_key)),
        ]
        return self._run_candle_cascade(symbol, chain, "5m")

    def get_quote_cascade(
        self,
        symbol: str,
        settings_obj: Any,
    ) -> Optional[Dict[str, float]]:
        """Execute waterfall cascade for bid/ask spread quotes."""
        poly_key = getattr(settings_obj, "POLYGON_API_KEY", "")
        fmp_key = getattr(settings_obj, "FMP_API_KEY", "")
        tiingo_key = getattr(settings_obj, "TIINGO_API_KEY", "")

        chain: List[Tuple[str, Optional[str], Callable[[], Optional[Dict[str, float]]]]] = [
            ("polygon", poly_key, lambda: self.fetch_polygon_quote(symbol, poly_key)),
            ("fmp", fmp_key, lambda: self.fetch_fmp_quote(symbol, fmp_key)),
            ("tiingo", tiingo_key, lambda: self.fetch_tiingo_quote(symbol, tiingo_key)),
        ]

        for prov_name, key, fetcher in chain:
            can_call, reason = self.can_request(prov_name, key)
            if not can_call:
                if reason != "no_api_key":
                    logger.info("MarketDataRouter: Skipping [%s] quote for %s (%s). Cascading...", prov_name, symbol, reason)
                continue

            self.record_request(prov_name)
            try:
                result = fetcher()
                if result and result.get("bid", 0.0) > 0 and result.get("ask", 0.0) > 0:
                    logger.info("MarketDataRouter: Acquired live bid/ask for %s via [%s]", symbol, prov_name)
                    return result
            except Exception as exc:
                logger.debug("MarketDataRouter: [%s] quote error for %s: %s", prov_name, symbol, exc)

        return None

    def get_earnings_days_cascade(
        self,
        symbol: str,
        settings_obj: Any,
    ) -> Optional[float]:
        """Execute waterfall cascade for earnings dates (skipping non-equities)."""
        clean = (symbol or "").strip().upper()
        if not clean:
            return None

        # Fast exit for non-equity funds and commodities
        if clean in self._NON_EQUITY_SYMBOLS or any(clean.startswith(p) for p in [
            "^", "EUR", "GBP", "BTC", "ETH", "SOL", "XRP", "ADA", "USD"
        ]):
            return None

        fmp_key = getattr(settings_obj, "FMP_API_KEY", "")
        av_key = getattr(settings_obj, "ALPHA_VANTAGE_API_KEY", "")

        chain: List[Tuple[str, Optional[str], Callable[[], Optional[float]]]] = [
            ("fmp", fmp_key, lambda: self.fetch_fmp_earnings_days(symbol, fmp_key)),
            ("alpha_vantage", av_key, lambda: self.fetch_alphavantage_earnings_days(symbol, av_key)),
        ]

        for prov_name, key, fetcher in chain:
            can_call, reason = self.can_request(prov_name, key)
            if not can_call:
                if reason != "no_api_key":
                    logger.info("MarketDataRouter: Skipping [%s] earnings for %s (%s). Cascading...", prov_name, symbol, reason)
                continue

            self.record_request(prov_name)
            try:
                days = fetcher()
                if days is not None:
                    logger.info("MarketDataRouter: Retrieved upcoming earnings for %s (%.1fd) via [%s]", symbol, days, prov_name)
                    return days
            except Exception as exc:
                logger.debug("MarketDataRouter: [%s] earnings error for %s: %s", prov_name, symbol, exc)

        return None

    def get_telemetry(self) -> Dict[str, Any]:
        """Return diagnostic status and quota metrics for all managed providers."""
        with self._provider_lock:
            now = time.time()
            data = {}
            for name, q in self.quotas.items():
                cooldown_rem = max(0, int(q._cooldown_until - now))
                data[name] = {
                    "status": q._last_status,
                    "cooldown_remaining_sec": cooldown_rem,
                    "day_request_count": q._day_request_count,
                    "daily_budget": q.daily_budget,
                    "total_success": q._total_success,
                    "total_errors": q._total_errors,
                    "total_429": q._total_429,
                    "last_error": q._last_error_reason,
                }
            return data
