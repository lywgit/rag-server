import logging
from fastapi import APIRouter, Depends, HTTPException, Response
from app.services.rag_service import RagService
from app.api.utils import get_rag_service

logger = logging.getLogger(__name__)

health_router = APIRouter(tags=["health"])

# root as health check
@health_router.get("/", description="Service Health")
async def service_health(rag_service:RagService = Depends(get_rag_service)):
    if rag_service.is_ready:
        return {"status":"ok"}
    else:# response with 503 code and retry header 
        raise HTTPException(
        status_code=503,
        detail="Service not ready. Please retry later.",
        headers={
            "Retry-After": "30",         # seconds
            "Cache-Control": "no-store"
        }
    )

# ultra-minimal liveness check: no body, fast path
@health_router.head("/", description="Service Liveness (HEAD)")
async def service_liveness_head():
    return Response(status_code=200, headers={"Cache-Control": "no-store"})