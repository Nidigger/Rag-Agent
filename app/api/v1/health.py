from fastapi import APIRouter

from app.common.response import success

router = APIRouter()


@router.get("/health")
async def health_check():
    return success(data={"status": "healthy", "version": "0.1.0"})
