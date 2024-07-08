__all__ = ["get_fully_qualname"]

from typing import Any


def get_fully_qualname(sth: Any) -> str:
    """Get the fully-qualified name of a class or object instance.

    Args:
        sth: Anything"""
    if not isinstance(sth, type):
        class_ = type(sth)
    else:
        class_ = sth
    module_name = class_.__module__
    if module_name == "builtins":
        return class_.__qualname__
    return f"{module_name}.{class_.__qualname__}"
