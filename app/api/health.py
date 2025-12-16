from fastapi import APIRouter

healthz_router = APIRouter(tags=["health"])

# root as health check
@healthz_router.get("/", description="Service Health")
async def service_health():
    return {"status":"ok"}

