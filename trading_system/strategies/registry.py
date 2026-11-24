"""Strategy registry providing dynamic lookup, registration, and instantiation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Type

from trading_system.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Central registry and factory for quantitative trading strategies."""

    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
        """Decorator to register a BaseStrategy subclass under a canonical name.

        Args:
            name: Canonical lowercase string identifier (e.g. "directional_trend").
        """
        def decorator(strategy_cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
            canonical_name = name.lower().strip()
            cls._registry[canonical_name] = strategy_cls
            logger.debug("Registered trading strategy '%s' -> %s", canonical_name, strategy_cls.__name__)
            return strategy_cls

        return decorator

    @classmethod
    def register_class(cls, name: str, strategy_cls: Type[BaseStrategy]) -> None:
        """Direct method to register a strategy class."""
        canonical_name = name.lower().strip()
        cls._registry[canonical_name] = strategy_cls
        logger.debug("Registered trading strategy '%s' -> %s", canonical_name, strategy_cls.__name__)

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseStrategy:
        """Instantiate and return the requested strategy by name.

        Args:
            name: Strategy name identifier.
            **kwargs: Additional parameters passed to the strategy constructor.

        Returns:
            Instance of BaseStrategy.

        Raises:
            KeyError: If the strategy name is not registered.
        """
        # Ensure built-in strategies are imported and registered
        cls._ensure_builtins_registered()

        canonical_name = name.lower().strip()
        if canonical_name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Unknown trading strategy '{name}'. Available strategies: {available}"
            )

        strategy_cls = cls._registry[canonical_name]
        return strategy_cls(name=canonical_name, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of all registered strategy identifiers."""
        cls._ensure_builtins_registered()
        return sorted(list(cls._registry.keys()))

    @classmethod
    def _ensure_builtins_registered(cls) -> None:
        """Lazy-import built-in strategies to avoid circular imports during startup."""
        try:
            import trading_system.strategies.directional_trend  # noqa: F401
            import trading_system.strategies.vwap  # noqa: F401
            import trading_system.strategies.ml_alpha  # noqa: F401
        except ImportError:
            pass
