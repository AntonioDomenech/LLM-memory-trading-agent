
import logging, sys

_logger = None
_warned = set()


def get_logger():
    """Return a module-level logger configured for stdout output."""

    global _logger
    if _logger is None:
        _logger = logging.getLogger("finmem")
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        _logger.addHandler(handler)
    return _logger


def warn_once(log, key: str, message: str):
    """Emit ``message`` at most once per ``key`` using ``log``."""

    global _warned
    if key not in _warned:
        _warned.add(key)
        log.warning(message)
