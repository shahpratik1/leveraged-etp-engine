"""Quantitative trading strategies package for modest-bose."""

from trading_system.strategies.base import BaseStrategy, StrategySignal
from trading_system.strategies.directional_trend import DirectionalTrendStrategy
from trading_system.strategies.vwap import VWAPStrategy
from trading_system.strategies.ml_alpha import MLAlphaStrategy
from trading_system.strategies.registry import StrategyRegistry

__all__ = [
    "BaseStrategy",
    "StrategySignal",
    "StrategyRegistry",
    "DirectionalTrendStrategy",
    "VWAPStrategy",
    "MLAlphaStrategy",
]
