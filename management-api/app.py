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


class DockerEngineClient:
    def __init__(self, docker_sock: str = "/var/run/docker.sock"):
        if not os.path.exists(docker_sock):
            raise HTTPException(status_code=500, detail=f"Docker socket not found at {docker_sock}. Mount it into the management container (e.g., -v /var/run/docker.sock:/var/run/docker.sock)")
        # Install unixsocket adapter
        try:
            requests_unixsocket.monkeypatch()
        except Exception:
            pass
        self.session = requests_unixsocket.Session()
        # URL-encode the socket path
        encoded = docker_sock.replace("/", "%2F")
        self.base = f"http+unix://{encoded}"

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base + path

    def version(self) -> Dict[str, Any]:
        r = self.session.get(self._url("/version"), timeout=5)
        r.raise_for_status()
        return r.json()

    def containers_create(self, config: Dict[str, Any]) -> str:
        r = self.session.post(self._url("/containers/create"), json=config, timeout=30)
        if r.status_code == 409 and "message" in r.json():
            raise HTTPException(status_code=409, detail=r.json()["message"])
        r.raise_for_status()
        return r.json()["Id"]

    def containers_start(self, container_id: str) -> None:
        r = self.session.post(self._url(f"/containers/{container_id}/start"), timeout=30)
        r.raise_for_status()

    def containers_wait(self, container_id: str, condition: str = "not-running") -> Dict[str, Any]:
        r = self.session.post(self._url(f"/containers/{container_id}/wait?condition={condition}"), timeout=None)
        r.raise_for_status()
        return r.json()

    def containers_logs(self, container_id: str, stdout: bool = True, stderr: bool = False, tail: Optional[int] = None) -> str:
        params = [f"stdout={str(stdout).lower()}", f"stderr={str(stderr).lower()}"]
        if tail is not None:
            params.append(f"tail={tail}")
        qs = "&".join(params)
        r = self.session.get(self._url(f"/containers/{container_id}/logs?{qs}"), timeout=30)
        r.raise_for_status()
        return r.text

    def containers_remove(self, container_id: str, force: bool = False) -> None:
        r = self.session.delete(self._url(f"/containers/{container_id}?force={'true' if force else 'false'}"), timeout=30)
        # 404 is fine for remove
        if r.status_code not in (204, 404):
            r.raise_for_status()

    def containers_list(self, all_containers: bool = True, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        import json as _json
        params = [f"all={'1' if all_containers else '0'}"]
        if filters:
            params.append("filters=" + _json.dumps(filters))
        qs = "&".join(params)
        r = self.session.get(self._url(f"/containers/json?{qs}"), timeout=10)
        r.raise_for_status()
        return r.json()

    def containers_inspect(self, container_id: str) -> Dict[str, Any]:
        r = self.session.get(self._url(f"/containers/{container_id}/json"), timeout=10)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="Not found")
        r.raise_for_status()
        return r.json()

    def run(self, image: str, command: List[str], name: Optional[str] = None, detach: bool = True, remove: bool = False,
            environment: Optional[Dict[str, str]] = None, volumes: Optional[Dict[str, Dict[str, str]]] = None,
            labels: Optional[Dict[str, str]] = None, network: Optional[str] = None, device_ids: Optional[List[str]] = None) -> str:
        # Build HostConfig
        binds = []
        if volumes:
            for vol_name, mount in volumes.items():
                bind = f"{vol_name}:{mount.get('bind')}:{mount.get('mode','rw')}"
                binds.append(bind)
        host_config: Dict[str, Any] = {"Binds": binds}
        if network:
            host_config["NetworkMode"] = network
        if device_ids:
            host_config["DeviceRequests"] = [{
                "Count": len(device_ids),
                "Capabilities": [["gpu"]],
                "DeviceIDs": device_ids,
            }]
        config: Dict[str, Any] = {
            "Image": image,
            "Cmd": command,
            "Env": [f"{k}={v}" for k, v in (environment or {}).items()],
            "Labels": labels or {},
            "HostConfig": host_config,
        }
        if name:
            config["name"] = name
        cid = self.containers_create(config)
        try:
            self.containers_start(cid)
        except Exception:
            # On failure to start, remove created container
            try:
                self.containers_remove(cid, force=True)
            except Exception:
                pass
            raise
        if not detach:
            self.containers_wait(cid)
            if remove:
                self.containers_remove(cid, force=True)
        return cid


def docker_client() -> DockerEngineClient:
    docker_sock = os.getenv("DOCKER_SOCK", "/var/run/docker.sock")
    try:
        client = DockerEngineClient(docker_sock=docker_sock)
        # Validate connectivity early
        _ = client.version()
        return client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize Docker client: {e}")


def query_gpu_stats(cli: Any) -> List[Dict[str, Any]]:
    image = os.getenv("CUDA_INFO_IMAGE", "nvidia/cuda:12.3.2-base-ubuntu22.04")
    cmd = ["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits"]
    try:
        # Run a short-lived container with GPU access
        cid = cli.run(
            image=image,
            command=cmd,
            detach=True,
            remove=False,
            device_ids=None,  # all GPUs by default via Count below
            environment=None,
            volumes=None,
            labels={"purpose": "gpu-query"},
        )
        # Wait for completion and collect logs
        cli.containers_wait(cid)
        logs = cli.containers_logs(cid, stdout=True, stderr=False)
        cli.containers_remove(cid, force=True)
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

    env = {
        "HF_HOME": "/models",
        "NCCL_P2P_DISABLE": os.getenv("NCCL_P2P_DISABLE", "1"),
    }

    cmd = to_worker_command(req, port)
    cmd += ["--tensor-parallel-size", str(tp_size)]

    # Create container without publishing port on host; internal network only
    try:
        container_id = cli.run(
            image=VLLM_IMAGE,
            name=container_name,
            command=cmd,
            detach=True,
            remove=False,
            network=DOCKER_NETWORK,
            device_ids=device_ids,
            environment=env,
            volumes={
                MODELS_VOLUME_NAME: {"bind": "/models", "mode": "rw"},
            },
            labels=labels,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start vLLM container: {e}")

    return {"id": container_id, "name": req.name, "port": port, "gpus": device_ids, "tp": tp_size}


@app.get("/instances", tags=["Instances"], summary="List vLLM workers with health")
def list_instances(_: Any = Depends(auth)):
    cli = docker_client()
    containers = cli.containers_list(all_containers=True, filters={"label": ["managed-by=management-api"]})
    out: List[Dict[str, Any]] = []
    for c in containers:
        # try to check health endpoint of the worker
        health = None
        try:
            import requests
            labels = c.get("Labels") or {}
            port = labels.get("port")
            name = (c.get("Names") or [None])[0]
            if name and name.startswith("/"):
                name = name[1:]
            if port and name:
                r = requests.get(f"http://{name}:{port}/health", timeout=1.5)
                health = r.json() if r.ok else {"status": "unhealthy"}
        except Exception:
            health = {"status": "unknown"}
        out.append({
            "id": c.get("Id"),
            "name": name or c.get("Id"),
            "status": c.get("State"),
            "labels": labels,
            "health": health,
        })
    return out


@app.get("/instances/{container_id}", tags=["Instances"], summary="Get worker details")
def get_instance(container_id: str, _: Any = Depends(auth)):
    cli = docker_client()
    c = cli.containers_inspect(container_id)
    return {
        "id": c.get("Id"),
        "name": c.get("Name", "").lstrip("/"),
        "status": ((c.get("State") or {}).get("Status")),
        "labels": c.get("Config", {}).get("Labels", {}),
    }


@app.delete("/instances/{container_id}", tags=["Instances"], summary="Stop and remove worker")
def delete_instance(container_id: str, _: Any = Depends(auth)):
    cli = docker_client()
    cli.containers_remove(container_id, force=True)
    return {"status": "deleted"}


