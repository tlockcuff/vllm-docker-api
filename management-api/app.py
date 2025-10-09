import os
import re
import socket
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from huggingface_hub import HfApi
import docker as docker_sdk


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
DEFAULT_PORT_RANGE = os.getenv("DEFAULT_PORT_RANGE", "21100-21999")
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:latest")
DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "vllm_net")
MODELS_VOLUME_NAME = os.getenv("MODELS_VOLUME_NAME", "vllm_models")


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


@app.post("/models/download", tags=["Models"], summary="Download/cache model files")
def models_download(req: DownloadRequest, _: Any = Depends(auth)):
    api = HfApi()
    dest = os.environ.get("HF_HOME", "/models")
    patterns = req.patterns or ["*"]
    for pattern in patterns:
        api.snapshot_download(repo_id=req.repo_id, revision=req.revision, local_dir=dest, allow_patterns=pattern, local_dir_use_symlinks=False)
    return {"status": "ok", "cached_in": dest}


@app.get("/models/local", tags=["Models"], summary="List cached model files")
def models_local(_: Any = Depends(auth)):
    base = os.environ.get("HF_HOME", "/models")
    out: List[str] = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".json") or f.endswith(".safetensors"):
                out.append(os.path.join(root, f).replace(base + "/", ""))
    return sorted(out)


def docker_client():
    return docker_sdk.from_env()


def query_gpu_stats(cli: docker_sdk.DockerClient) -> List[Dict[str, Any]]:
    image = os.getenv("CUDA_INFO_IMAGE", "nvidia/cuda:12.3.2-base-ubuntu22.04")
    cmd = ["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits"]
    try:
        device_request = docker_sdk.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        c = cli.containers.run(image, command=cmd, detach=True, remove=True, device_requests=[device_request])
        logs = c.logs(stream=False, stdout=True, stderr=False).decode("utf-8", errors="ignore")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query GPUs: {e}")
    stats: List[Dict[str, Any]] = []
    for line in logs.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            idx = int(parts[0])
            name = parts[1]
            total = float(parts[2])
            used = float(parts[3])
            free = float(parts[4]) if len(parts) > 4 else max(0.0, total - used)
            stats.append({"index": idx, "name": name, "total_gb": total/1024.0, "used_mb": used, "free_gb": free/1024.0})
    return sorted(stats, key=lambda s: s["free_gb"], reverse=True)


def choose_gpu_allocation(cli: docker_sdk.DockerClient, req: InstanceRequest) -> List[int]:
    # Honor explicit request if both provided
    if req.gpu is not None and req.tensor_parallel:
        return [i for i in range(req.gpu, req.gpu + req.tensor_parallel)]
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
        "--tensor-parallel-size", str(req.tensor_parallel or 1),
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

    container_name = f"vllm-{req.name}"
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


