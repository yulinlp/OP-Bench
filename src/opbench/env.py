"""Optional dotenv loading without making the CLI unusable in minimal envs."""

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:  # pragma: no cover - exercised only without the base extra
    def load_dotenv(*args, **kwargs):
        return False
else:
    load_dotenv = _load_dotenv

