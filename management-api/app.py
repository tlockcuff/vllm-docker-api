import os
import re
import socket
import asyncio
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
from huggingface_hub import HfApi, hf_hub_url
import httpx
import docker as docker_sdk
import requests_unixsocket


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
DEFAULT_PORT_RANGE = os.getenv("DEFAULT_PORT_RANGE", "21100-21999")
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "vllm_net")
MODELS_VOLUME_NAME = os.getenv("MODELS_VOLUME_NAME", "vllm_models")
GATEWAY_INTERNAL_BASE_URL = os.getenv("GATEWAY_INTERNAL_BASE_URL", "http://gateway:8000")


def auth(authorization: Optional[str] = Header(None)):
    if not ADMIN_TOKEN:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def parse_port_range(spec: str) -> List[int]:
    match = re.match(r"^(\d+)-(\d+)$", spec)
    if not match:
        raise ValueError("DEFAULT_PORT_RANGE must be like start-end")
    start, end = int(match.group(1)), int(match.group(2))
    if start >= end:
        raise ValueError("Invalid port range")
    return list(range(start, end + 1))


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        return s.connect_ex(("127.0.0.1", port)) != 0


def allocate_port() -> int:
    for p in parse_port_range(DEFAULT_PORT_RANGE):
        if is_port_free(p):
            return p
    raise HTTPException(status_code=503, detail="No free ports available")


class DownloadRequest(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    patterns: Optional[List[str]] = None


class InstanceRequest(BaseModel):
    name: str
    model: str
    gpu: Optional[int] = None
    port: Optional[int] = None
    max_model_len: Optional[int] = None
    tensor_parallel: Optional[int] = None
    dtype: Optional[str] = "bfloat16"
    gpu_mem_util: Optional[float] = 0.9
    extra_args: Optional[List[str]] = None
    estimated_vram_gb: Optional[float] = None


tags_metadata = [
    {"name": "Health", "description": "Service health checks."},
    {"name": "Models", "description": "Search and manage model files."},
    {"name": "Instances", "description": "Lifecycle of vLLM worker instances."},
]

app = FastAPI(
    title="vLLM Management API",
    version="1.0.0",
    description="Admin API for orchestrating vLLM workers, models, and GPUs.",
    openapi_tags=tags_metadata,
)


@app.get("/health", tags=["Health"], summary="Health probe")
def health():
    return {"status": "ok"}


@app.get("/models/available", tags=["Models"], summary="Search models on Hugging Face")
def models_available(query: str, _: Any = Depends(auth)):
    api = HfApi()
    results = api.list_models(search=query, sort="downloads", direction=-1, limit=20)
    return [
        {"id": m.modelId, "likes": m.likes, "downloads": m.downloads}
        for m in results
    ]


@app.get("/models/local", tags=["Models"], summary="List cached models (grouped by repo)")
def models_local(_: Any = Depends(auth)):
    base = os.environ.get("HF_HOME", "/models")
    if not os.path.exists(base):
        return []
    blacklist_first = {".cache", "cache", "hub", "datasets", "spaces", "spaces-temp", "metal", "original"}
    aggregations: Dict[str, Dict[str, Any]] = {}
    for root, _, files in os.walk(base):
        for f in files:
            rel = os.path.join(root, f).replace(base + "/", "")
            parts = rel.split("/")
            if not parts or len(parts) < 2:
                continue
            first = parts[0]
            if first in blacklist_first or first.startswith('.'):
                continue
            # Determine repo key as <org>/<repo> when available
            repo_key = f"{parts[0]}/{parts[1]}"
            agg = aggregations.setdefault(repo_key, {
                "repo_id": repo_key,
                "files_total": 0,
                "size_bytes": 0,
                "has_config": False,
                "has_safetensors": False,
                "sample_files": [],
            })
            agg["files_total"] += 1
            try:
                agg["size_bytes"] += os.path.getsize(os.path.join(base, rel))
            except Exception:
                pass
            if f.endswith(".safetensors"):
                agg["has_safetensors"] = True
            if f == "config.json":
                agg["has_config"] = True
            if len(agg["sample_files"]) < 5 and (f.endswith(".safetensors") or f.endswith(".json")):
                agg["sample_files"].append(rel)
    # Only include repos that look like actual models (have config or safetensors)
    models = [m for m in aggregations.values() if m["has_config"] or m["has_safetensors"]]
    models.sort(key=lambda m: m["repo_id"])
    return models


# -------------------------------
# Download jobs with progress
# -------------------------------

class DownloadJobStart(BaseModel):
    repo_id: str
    revision: Optional[str] = None
    patterns: Optional[List[str]] = None


class DownloadJobStatus(BaseModel):
    job_id: str
    status: str
    repo_id: str
    revision: Optional[str]
    files_total: int
    files_done: int
    total_bytes: int
    downloaded_bytes: int
    rate_bps: float
    eta_seconds: Optional[float]
    progress_percent: float
    current_file: Optional[str]
    current_file_bytes: int
    current_file_size: int
    error: Optional[str] = None


DOWNLOAD_JOBS: Dict[str, Dict[str, Any]] = {}


def _job_status_snapshot(job_id: str) -> DownloadJobStatus:
    s = DOWNLOAD_JOBS.get(job_id)
    if not s:
        raise HTTPException(status_code=404, detail="Job not found")
    remaining = max(0, s["total_bytes"] - s["downloaded_bytes"]) if s["total_bytes"] else 0
    eta = (remaining / s["rate_bps"]) if s["rate_bps"] > 0 else None
    progress = (float(s["downloaded_bytes"]) / float(s["total_bytes"]) * 100.0) if s["total_bytes"] > 0 else 0.0
    return DownloadJobStatus(
        job_id=job_id,
        status=s["status"],
        repo_id=s["repo_id"],
        revision=s.get("revision"),
        files_total=s["files_total"],
        files_done=s["files_done"],
        total_bytes=s["total_bytes"],
        downloaded_bytes=s["downloaded_bytes"],
        rate_bps=s["rate_bps"],
        eta_seconds=eta,
        progress_percent=progress,
        current_file=s.get("current_file"),
        current_file_bytes=s.get("current_file_bytes", 0),
        current_file_size=s.get("current_file_size", 0),
        error=s.get("error"),
    )


async def _run_download_job(job_id: str):
    s = DOWNLOAD_JOBS[job_id]
    repo_id = s["repo_id"]
    revision = s.get("revision")
    patterns = s.get("patterns") or ["*"]
    base_dir = Path(os.environ.get("HF_HOME", "/models")).resolve()
    dest_root = base_dir / repo_id
    dest_root.mkdir(parents=True, exist_ok=True)

    # list files and sizes
    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    except Exception as e:
        s.update({"status": "error", "error": str(e)})
        return
    files = []
    total_bytes = 0
    for f in info.siblings:
        path = f.rfilename
        if any(fnmatch(path, pat) for pat in patterns):
            size = int(getattr(f, "size", 0) or 0)
            files.append({"path": path, "size": size})
            total_bytes += size
    s.update({"files_total": len(files), "total_bytes": total_bytes})

    headers = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    start_time = time.time()
    downloaded_bytes = 0
    files_done = 0
    s["status"] = "running"

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        for f in files:
            if s.get("cancel"):
                s["status"] = "cancelled"
                return
            rel_path = f["path"]
            file_size = int(f.get("size") or 0)
            url = hf_hub_url(repo_id=repo_id, filename=rel_path, revision=revision)
            dest_path = (dest_root / rel_path).resolve()
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            s.update({
                "current_file": rel_path,
                "current_file_size": file_size,
                "current_file_bytes": 0,
            })
            # Skip if already exists and matches size
            if dest_path.exists() and file_size > 0 and dest_path.stat().st_size == file_size:
                downloaded_bytes += file_size
                files_done += 1
                s.update({
                    "downloaded_bytes": downloaded_bytes,
                    "files_done": files_done,
                    "current_file_bytes": file_size,
                })
                elapsed = max(0.001, time.time() - start_time)
                s["rate_bps"] = downloaded_bytes / elapsed
                continue

            try:
                async with client.stream("GET", url, headers=headers) as resp:
                    resp.raise_for_status()
                    with dest_path.open("wb") as out:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            if s.get("cancel"):
                                s["status"] = "cancelled"
                                return
                            if chunk:
                                out.write(chunk)
                                s["current_file_bytes"] += len(chunk)
                                downloaded_bytes += len(chunk)
                                s["downloaded_bytes"] = downloaded_bytes
                                elapsed = max(0.001, time.time() - start_time)
                                s["rate_bps"] = downloaded_bytes / elapsed
                files_done += 1
                s["files_done"] = files_done
            except Exception as e:
                s.update({"status": "error", "error": str(e)})
                return

    s["status"] = "completed"


@app.post("/models/downloads/start", tags=["Models"], summary="Start async model download")
async def start_download_job(req: DownloadJobStart, _: Any = Depends(auth)):
    job_id = os.urandom(8).hex()
    DOWNLOAD_JOBS[job_id] = {
        "status": "queued",
        "repo_id": req.repo_id,
        "revision": req.revision,
        "patterns": req.patterns or ["*"],
        "files_total": 0,
        "files_done": 0,
        "total_bytes": 0,
        "downloaded_bytes": 0,
        "rate_bps": 0.0,
        "current_file": None,
        "current_file_bytes": 0,
        "current_file_size": 0,
        "error": None,
        "cancel": False,
    }
    asyncio.create_task(_run_download_job(job_id))
    return {"job_id": job_id}


@app.get("/models/downloads/{job_id}", response_model=DownloadJobStatus, tags=["Models"], summary="Get download job status")
def get_download_job(job_id: str, _: Any = Depends(auth)):
    return _job_status_snapshot(job_id)


@app.post("/models/downloads/{job_id}/cancel", tags=["Models"], summary="Cancel a running download job")
def cancel_download_job(job_id: str, _: Any = Depends(auth)):
    s = DOWNLOAD_JOBS.get(job_id)
    if not s:
        raise HTTPException(status_code=404, detail="Job not found")
    s["cancel"] = True
    return {"status": "cancelling"}


@app.get("/models/downloads", response_model=List[DownloadJobStatus], tags=["Models"], summary="List all download jobs")
def list_download_jobs(_: Any = Depends(auth)):
    return [
        _job_status_snapshot(job_id)
        for job_id in list(DOWNLOAD_JOBS.keys())
    ]


# ----------------------------------------
# Proxied Gateway OpenAPI + Swagger UI
# ----------------------------------------

@app.get("/gateway/openapi.json", tags=["Gateway"], summary="Gateway OpenAPI spec (proxied)")
def gateway_openapi(_: Any = Depends(auth)):
    try:
        r = httpx.get(f"{GATEWAY_INTERNAL_BASE_URL}/openapi.json", timeout=5)
        r.raise_for_status()
        return JSONResponse(r.json())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch gateway openapi: {e}")


@app.get("/gateway/docs", include_in_schema=False)
def gateway_docs():
    return get_swagger_ui_html(openapi_url="/gateway/openapi.json", title="Gateway API Docs")


def docker_client():
    # Use plain unix socket API client (no http+docker scheme)
    docker_sock = os.getenv("DOCKER_SOCK", "/var/run/docker.sock")
    os.environ.pop("DOCKER_HOST", None)
    os.environ.pop("DOCKER_TLS_VERIFY", None)
    os.environ.pop("DOCKER_CERT_PATH", None)
    # Helpful error if the Docker socket isn't mounted into the container
    if not os.path.exists(docker_sock):
        raise HTTPException(status_code=500, detail=f"Docker socket not found at {docker_sock}. Mount it into the management container (e.g., -v /var/run/docker.sock:/var/run/docker.sock)")
    # Normalize to a valid unix scheme URL (unix://var/run/docker.sock)
    normalized = f"unix://{docker_sock.lstrip('/')}"
    api = docker_sdk.APIClient(base_url=normalized, version="auto", timeout=60)
    return docker_sdk.DockerClient(api_client=api)


def query_gpu_stats(cli: docker_sdk.DockerClient) -> List[Dict[str, Any]]:
    image = os.getenv("CUDA_INFO_IMAGE", "nvidia/cuda:12.3.2-base-ubuntu22.04")
    cmd = ["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits"]
    try:
        device_request = docker_sdk.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        # Run synchronously to avoid race with container removal and reliably capture output
        logs_bytes = cli.containers.run(
            image,
            command=cmd,
            detach=False,
            remove=True,
            device_requests=[device_request],
            stderr=True,
            stdout=True,
        )
        logs = logs_bytes.decode("utf-8", errors="ignore") if isinstance(logs_bytes, (bytes, bytearray)) else str(logs_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query GPUs: {e}")
    stats: List[Dict[str, Any]] = []
    for line in logs.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                idx = int(parts[0])
                name = parts[1]
                total = float(parts[2])
                used = float(parts[3])
                free = float(parts[4]) if len(parts) > 4 else max(0.0, total - used)
                stats.append({"index": idx, "name": name, "total_gb": total/1024.0, "used_mb": used, "free_gb": free/1024.0})
            except Exception:
                # Skip malformed lines (e.g., when driver/toolkit is missing and output is empty/noisy)
                continue
    return sorted(stats, key=lambda s: s["free_gb"], reverse=True)


def choose_gpu_allocation(cli: docker_sdk.DockerClient, req: InstanceRequest) -> List[int]:
    # Honor explicit request if both provided
    if req.gpu is not None and req.tensor_parallel:
        return [i for i in range(req.gpu, req.gpu + req.tensor_parallel)]
    # Honor single explicit GPU when tensor_parallel not specified
    if req.gpu is not None:
        return [req.gpu]
    stats = query_gpu_stats(cli)
    if not stats:
        raise HTTPException(status_code=503, detail="No GPUs available")
    required_gb = req.estimated_vram_gb or 14.0
    for s in stats:
        if s["free_gb"] >= required_gb * 0.95:
            return [s["index"]]
    if len(stats) >= 2 and (stats[0]["free_gb"] + stats[1]["free_gb"]) >= required_gb * 0.95:
        return [stats[0]["index"], stats[1]["index"]]
    return [stats[0]["index"]]

def to_worker_command(req: InstanceRequest, port: int) -> List[str]:
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model", req.model,
        "--dtype", req.dtype,
        "--gpu-memory-utilization", str(req.gpu_mem_util),
    ]
    if req.max_model_len:
        cmd += ["--max-model-len", str(req.max_model_len)]
    if req.extra_args:
        cmd += req.extra_args
    return cmd


@app.post("/instances", tags=["Instances"], summary="Start a vLLM worker")
def create_instance(req: InstanceRequest, _: Any = Depends(auth)):
    port = req.port or allocate_port()
    cli = docker_client()

    # Sanitize name for Docker container naming rules
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", req.name).strip("-") or "instance"
    container_name = f"vllm-{safe_name}"
    labels = {
        "managed-by": "management-api",
        "app": "vllm",
        "vllm-instance": req.name,
        "model": req.model,
        "port": str(port),
    }

    # Resolve GPU allocation (auto or explicit)
    allocation = choose_gpu_allocation(cli, req)
    device_ids = [str(i) for i in allocation]
    tp_size = len(device_ids)
    labels["device_ids"] = ",".join(device_ids)
    device_request = docker_sdk.types.DeviceRequest(
        count=len(device_ids), capabilities=[["gpu"]], device_ids=device_ids
    )

    env = {
        "HF_HOME": "/models",
        "NCCL_P2P_DISABLE": os.getenv("NCCL_P2P_DISABLE", "1"),
    }

    cmd = to_worker_command(req, port)
    cmd += ["--tensor-parallel-size", str(tp_size)]

    # Create container without publishing port on host; internal network only
    try:
        container = cli.containers.run(
            VLLM_IMAGE,
            name=container_name,
            command=cmd,
            detach=True,
            network=DOCKER_NETWORK,
            device_requests=[device_request],
            environment=env,
            volumes={
                # mount the named volume from compose
                MODELS_VOLUME_NAME: {"bind": "/models", "mode": "rw"},
            },
            labels=labels,
        )
    except docker_sdk.errors.APIError as e:
        # Surface Docker daemon error back to client
        detail = getattr(e, 'explanation', str(e))
        raise HTTPException(status_code=500, detail=f"Docker failed to start vLLM: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start vLLM container: {e}")

    return {"id": container.id, "name": req.name, "port": port, "gpus": device_ids, "tp": tp_size}


@app.get("/instances", tags=["Instances"], summary="List vLLM workers with health")
def list_instances(_: Any = Depends(auth)):
    cli = docker_client()
    containers = cli.containers.list(all=True, filters={"label": "managed-by=management-api"})
    out: List[Dict[str, Any]] = []
    for c in containers:
        # try to check health endpoint of the worker
        health = None
        try:
            import requests
            port = c.labels.get("port")
            if port:
                r = requests.get(f"http://{c.name}:{port}/health", timeout=1.5)
                health = r.json() if r.ok else {"status": "unhealthy"}
        except Exception:
            health = {"status": "unknown"}
        out.append({
            "id": c.id,
            "name": c.name,
            "status": c.status,
            "labels": c.labels,
            "health": health,
        })
    return out


@app.get("/instances/{container_id}", tags=["Instances"], summary="Get worker details")
def get_instance(container_id: str, _: Any = Depends(auth)):
    cli = docker_client()
    try:
        c = cli.containers.get(container_id)
    except docker_sdk.errors.NotFound:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": c.id,
        "name": c.name,
        "status": c.status,
        "labels": c.labels,
    }


@app.delete("/instances/{container_id}", tags=["Instances"], summary="Stop and remove worker")
def delete_instance(container_id: str, _: Any = Depends(auth)):
    cli = docker_client()
    try:
        c = cli.containers.get(container_id)
        c.remove(force=True)
    except docker_sdk.errors.NotFound:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "deleted"}


