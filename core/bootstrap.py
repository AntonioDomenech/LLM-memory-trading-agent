# Ensures project root is on sys.path and loads .env if present.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).name.lower() == "__init__.py") else Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
except Exception:
    pass
