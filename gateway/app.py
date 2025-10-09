import os
from typing import Optional, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import StreamingResponse, JSONResponse


GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY", "changeme")
ADMIN_BASE_URL = os.getenv("ADMIN_BASE_URL", "http://management-api:8081")


def auth(authorization: Optional[str] = Header(None)):
    if not GATEWAY_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    if token != GATEWAY_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


app = FastAPI(
    title="OpenAI Gateway for vLLM",
    version="1.0.0",
    description="Public OpenAI-compatible API routing requests to vLLM workers managed by the Admin API.",
)


@app.get("/health", tags=["Health"], summary="Health probe")
def health():
    return {"status": "ok"}


async def get_instances() -> Dict[str, Dict[str, Any]]:
    headers = {"Authorization": f"Bearer {os.getenv('ADMIN_TOKEN', '')}"} if os.getenv('ADMIN_TOKEN') else {}
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{ADMIN_BASE_URL}/instances", headers=headers)
        r.raise_for_status()
        out = {}
        for item in r.json():
            labels = item.get("labels", {})
            name = labels.get("vllm-instance") or item.get("name")
            port = labels.get("port")
            out[name] = {"name": name, "port": int(port) if port else None, "container": item.get("name")}
        return out


def instance_base_url(instance: Dict[str, Any]) -> str:
    container = instance["container"]
    port = instance["port"]
    if port is None:
        raise HTTPException(status_code=502, detail="Instance missing port")
    return f"http://{container}:{port}"


@app.get("/v1/models", tags=["OpenAI"], summary="List available models")
async def list_models(_: Any = Depends(auth)):
    instances = await get_instances()
    data = {"data": [{"id": name, "object": "model"} for name in instances.keys()]}
    return JSONResponse(data)


async def proxy_openai(request: Request, path: str, _: Any = Depends(auth)):
    body = await request.json()
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    instances = await get_instances()
    instance = instances.get(model)
    if not instance:
        raise HTTPException(status_code=404, detail="model instance not found")
    base = instance_base_url(instance)
    target = f"{base}/v1/{path}"
    # stream passthrough if stream=True
    stream = bool(body.get("stream"))
    async with httpx.AsyncClient(timeout=None if stream else 60) as client:
        if stream:
            r = await client.post(target, json=body, headers={"Content-Type": "application/json"}, timeout=None)
            r.raise_for_status()
            return StreamingResponse(r.aiter_raw(), media_type=r.headers.get("content-type", "text/event-stream"))
        else:
            r = await client.post(target, json=body, headers={"Content-Type": "application/json"})
            r.raise_for_status()
            return JSONResponse(r.json())


@app.post("/v1/chat/completions", tags=["OpenAI"], summary="Chat Completions API")
async def chat_completions(request: Request, _: Any = Depends(auth)):
    return await proxy_openai(request, "chat/completions")


@app.post("/v1/completions", tags=["OpenAI"], summary="Completions API")
async def completions(request: Request, _: Any = Depends(auth)):
    return await proxy_openai(request, "completions")


@app.post("/v1/embeddings", tags=["OpenAI"], summary="Embeddings API")
async def embeddings(request: Request, _: Any = Depends(auth)):
    return await proxy_openai(request, "embeddings")


