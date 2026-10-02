"""
Loads .env.local (repo root) into os.environ, if present, and exposes
require_env() for a clear failure message instead of a silent personal-path
fallback. See configs/paths.example.env for the full list of variables.
"""
import os

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DOTENV_PATH = os.path.join(_REPO_ROOT, '.env.local')
_loaded = False


def _load_dotenv_once():
    global _loaded
    if _loaded:
        return
    _loaded = True
    if not os.path.exists(_DOTENV_PATH):
        return
    with open(_DOTENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, val = line.partition('=')
            key, val = key.strip(), val.strip()
            os.environ.setdefault(key, val)


def get_env(name, default=None):
    _load_dotenv_once()
    return os.environ.get(name, default)


def require_env(name):
    """Return os.environ[name], or raise with a fix-it message."""
    _load_dotenv_once()
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set.\n"
            f"  Copy configs/paths.example.env to .env.local at the repo root "
            f"and fill in {name} (or export it in your shell)."
        )
    return val
