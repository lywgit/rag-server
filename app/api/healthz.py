from fastapi import APIRouter

healthz_router = APIRouter(prefix="/healthz", tags=["healthz"])

@healthz_router.get("")
def api_health():
    return {"status":"ok"}

