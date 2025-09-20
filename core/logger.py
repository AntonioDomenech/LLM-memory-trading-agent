
import logging, sys
_logger = None
_warned = set()
def get_logger():
    global _logger
    if _logger is None:
        _logger = logging.getLogger("finmem")
        _logger.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        _logger.addHandler(h)
    return _logger
def warn_once(log, key: str, message: str):
    global _warned
    if key not in _warned:
        _warned.add(key)
        log.warning(message)
