from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .persian_phonemizer import PersianPhonemizer

__all__ = ["PersianPhonemizer"]


def __getattr__(name: str) -> Any:
    if name == "PersianPhonemizer":
        from .persian_phonemizer import PersianPhonemizer

        globals()[name] = PersianPhonemizer
        return PersianPhonemizer

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
