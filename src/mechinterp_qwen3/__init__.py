__all__ = ["probe"]


def __getattr__(name: str):
    if name == "probe":
        from . import probe as _probe
        return _probe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
