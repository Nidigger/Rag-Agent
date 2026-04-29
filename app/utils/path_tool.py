"""Path utilities — resolve project-relative paths."""

from app.core.config import settings


def get_project_root() -> str:
    """Return the absolute path of the project root directory."""
    return str(settings.project_root)


def get_abs_path(relative_path: str) -> str:
    """Convert a project-relative path to an absolute path."""
    return str(settings.project_root / relative_path)
