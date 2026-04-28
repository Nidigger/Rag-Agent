from app.core.config import settings


def get_project_root() -> str:
    return str(settings.PROJECT_ROOT)


def get_abs_path(relative_path: str) -> str:
    return str(settings.PROJECT_ROOT / relative_path)
