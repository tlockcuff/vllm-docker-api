import os
import time
import threading
import shutil
from typing import List, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from huggingface_hub import HfApi, snapshot_download, repo_info
import docker

app = FastAPI()

model_dir = "/models"
os.makedirs(model_dir, exist_ok=True)

progress: Dict[str, float] = {}  # model: progress (0-100)
total_sizes: Dict[str, int] = {}  # model: total size
current_threads: Dict[str, threading.Thread] = {}

docker_client = docker.from_env()

api = HfApi(token=os.getenv("HF_TOKEN"))

def calculate_dir_size(path: str) -> int:
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total

def monitor_progress(model: str, local_path: str):
    while True:
        current_size = calculate_dir_size(local_path)
        prog = (current_size / total_sizes[model]) * 100 if total_sizes[model] > 0 else 0
        progress[model] = min(prog, 100)
        if prog >= 100:
            break
        time.sleep(5)  # Poll every 5 seconds

def download_model(model: str):
    local_path = os.path.join(model_dir, model.replace("/", "_"))
    os.makedirs(local_path, exist_ok=True)

    # Get total size
    try:
        r_info = repo_info(repo_id=model, repo_type="model", files_metadata=True)
        total_size = sum(sibling.size or 0 for sibling in r_info.siblings if sibling.size is not None)
        total_sizes[model] = total_size
    except Exception:
        total_sizes[model] = 0  # Fallback if size can't be fetched

    progress[model] = 0

    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_progress, args=(model, local_path))
    monitor_thread.start()

    # Download
    snapshot_download(repo_id=model, local_dir=local_path, resume_download=True, local_dir_use_symlinks=False)

    # Wait for monitor to finish
    monitor_thread.join()
    progress[model] = 100

@app.get("/pulled", response_model=List[str])
def view_pulled_models():
    return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

@app.get("/available")
def view_available_models(search: str = ""):
    models = api.list_models(search=search, sort="downloads", direction=-1, limit=20)
    model_details = []
    for m in models:
        model_info = {
            "id": m.id,
            "modelId": m.modelId,
            "author": getattr(m, 'author', None),
            "downloads": m.downloads,
            "likes": m.likes,
            "created_at": m.created_at.isoformat() if hasattr(m, 'created_at') and m.created_at else None,
            "last_modified": m.last_modified.isoformat() if hasattr(m, 'last_modified') and m.last_modified else None,
            "pipeline_tag": m.pipeline_tag,
            "library_name": m.library_name,
            "tags": m.tags,
            "private": m.private,
            "trending_score": getattr(m, 'trending_score', None),
            "gated": getattr(m, 'gated', None),
            "inference": getattr(m, 'inference', None),
        }
        model_details.append(model_info)
    return model_details

@app.get("/available/{model_id}")
def get_model_details(model_id: str):
    """Get detailed information for a specific model including description, config, and other metadata."""
    try:
        # Get basic model info from search results
        models = api.list_models(search=model_id, limit=1)
        basic_model = None
        for m in models:
            if m.id == model_id:
                basic_model = m
                break

        if not basic_model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get detailed repo info
        repo_details = repo_info(repo_id=model_id, repo_type="model", files_metadata=False)

        # Extract additional metadata from card_data and config
        card_data = repo_details.card_data
        config = repo_details.config if hasattr(repo_details, 'config') and repo_details.config else {}

        # Build comprehensive model details
        model_info = {
            "id": repo_details.id,
            "modelId": repo_details.modelId,
            "author": repo_details.author,
            "downloads": repo_details.downloads,
            "likes": repo_details.likes,
            "created_at": repo_details.created_at.isoformat() if repo_details.created_at else None,
            "last_modified": repo_details.last_modified.isoformat() if repo_details.last_modified else None,
            "pipeline_tag": repo_details.pipeline_tag,
            "library_name": repo_details.library_name,
            "tags": repo_details.tags,
            "private": repo_details.private,
            "gated": repo_details.gated,
            "inference": repo_details.inference,
            "trending_score": getattr(repo_details, 'trending_score', None),
            "usedStorage": getattr(repo_details, 'usedStorage', None),
        }

        # Add card data if available
        if card_data:
            card_fields = [
                "base_model", "license", "language", "description",
                "summary", "model_name", "model_type", "task_specific_params"
            ]
            for field in card_fields:
                if hasattr(card_data, field) and getattr(card_data, field) is not None:
                    model_info[field] = getattr(card_data, field)

        # Add config data if available
        if config:
            # Extract context window and other model parameters from config
            if 'max_position_embeddings' in config:
                model_info['context_window'] = config['max_position_embeddings']
            if 'vocab_size' in config:
                model_info['vocab_size'] = config['vocab_size']
            if 'hidden_size' in config:
                model_info['hidden_size'] = config['hidden_size']
            if 'num_attention_heads' in config:
                model_info['num_attention_heads'] = config['num_attention_heads']
            if 'num_hidden_layers' in config:
                model_info['num_hidden_layers'] = config['num_hidden_layers']
            if 'torch_dtype' in config:
                model_info['torch_dtype'] = config['torch_dtype']

        return model_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model details: {str(e)}")

@app.post("/pull/{model}")
def pull_model(model: str, background_tasks: BackgroundTasks):
    if model in current_threads and current_threads[model].is_alive():
        raise HTTPException(status_code=400, detail="Download already in progress")
    thread = threading.Thread(target=download_model, args=(model,))
    current_threads[model] = thread
    thread.start()
    background_tasks.add_task(lambda: thread.join() and current_threads.pop(model, None))
    return {"status": "started"}

@app.get("/progress/{model}")
def get_download_progress(model: str):
    return {"progress": progress.get(model, 0)}

@app.get("/running", response_model=List[Dict])
def show_running_models():
    running = []
    containers = docker_client.containers.list(filters={"name": "vllm-"})
    for cont in containers:
        name = cont.name.replace("vllm-", "")
        model = next((cmd for cmd in cont.attrs["Config"]["Cmd"] if cmd.startswith("--model")), None)
        if model:
            model = model.split(" ", 1)[1]
        status = cont.status
        running.append({"name": name, "model": model, "status": status})
    return running

@app.delete("/delete/{model}")
def delete_model(model: str):
    local_path = os.path.join(model_dir, model.replace("/", "_"))
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Model not found")

# Bonus: Start a model (assuming you have GPUs available; adjust as needed)
@app.post("/start/{model}")
def start_model(model: str, tensor_parallel_size: int = 1):
    local_path = os.path.join(model_dir, model.replace("/", "_"))
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Model not pulled")
    
    container_name = f"vllm-{model.replace('/', '_')}"
    if any(c.name == container_name for c in docker_client.containers.list(all=True)):
        raise HTTPException(status_code=400, detail="Container already exists")

    # Run vLLM container
    docker_client.containers.run(
        "vllm/vllm-openai:latest",
        name=container_name,
        command=[
            "--model", local_path,
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--dtype", "auto",
            "--api-key", "token-abc123"  # Change as needed
        ],
        ports={"8000/tcp": None},  # Host port auto-assigned; query container for it
        volumes={model_dir: {"bind": model_dir, "mode": "rw"}},
        runtime="nvidia",  # For GPU
        detach=True
    )
    return {"status": "started", "container": container_name}

# Bonus: Stop a model
@app.post("/stop/{model}")
def stop_model(model: str):
    container_name = f"vllm-{model.replace('/', '_')}"
    try:
        cont = docker_client.containers.get(container_name)
        cont.stop()
        cont.remove()
        return {"status": "stopped"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")