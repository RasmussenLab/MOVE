__all__ = ["style_settings"]

from typing import cast, ContextManager

import matplotlib.style


def style_settings(style: str) -> ContextManager:
    """Returns a context manager for using style settings.

    Args:
        style: Style name
    """
    return cast(ContextManager, matplotlib.style.context(style))
