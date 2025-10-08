import os
import time
import threading
import shutil
from typing import List, Dict, Optional, Union
from datetime import datetime
from urllib.parse import unquote

from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum
from huggingface_hub import HfApi, snapshot_download, repo_info
import docker

app = FastAPI(
    title="vLLM Docker API",
    description="API for managing vLLM models and containers",
    version="1.0.0"
)

model_dir = "/models"

# Pydantic Response Models
class ModelInfo(BaseModel):
    id: str
    modelId: Optional[str] = None
    author: Optional[str] = None
    downloads: int
    likes: int
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    tags: List[str]
    private: bool
    trending_score: Optional[float] = None
    gated: Optional[bool] = None
    inference: Optional[str] = None

class DetailedModelInfo(ModelInfo):
    usedStorage: Optional[Union[str, int]] = None
    base_model: Optional[str] = None
    license: Optional[str] = None
    language: Optional[str] = None
    description: Optional[str] = None
    summary: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    task_specific_params: Optional[Dict] = None
    context_window: Optional[int] = None
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    torch_dtype: Optional[str] = None

class QuantizationLevel(str, Enum):
    """Quantization levels for model size reduction"""
    FP32 = "fp32"  # Full precision (largest size)
    FP16 = "fp16"  # Half precision (~50% size reduction)
    INT8 = "int8"  # 8-bit quantization (~75% size reduction)
    INT4 = "int4"  # 4-bit quantization (~87% size reduction)
    GPTQ = "gptq"  # GPTQ quantization (very small, optimized)

class ModelFormat(str, Enum):
    """Model file formats"""
    SAFETENSORS = "safetensors"  # Recommended for vLLM
    PYTORCH = "pytorch"  # Traditional PyTorch format

class PullOptions(BaseModel):
    """Options for model downloading and optimization"""
    quantization: QuantizationLevel = QuantizationLevel.FP16  # Default to FP16 for good balance
    format: ModelFormat = ModelFormat.SAFETENSORS
    use_optimized_version: bool = True  # Try to find pre-optimized versions

class PullStatus(BaseModel):
    status: str
    model_path: Optional[str] = None
    original_size: Optional[int] = None
    final_size: Optional[int] = None
    compression_ratio: Optional[float] = None

class ProgressInfo(BaseModel):
    progress: float  # Progress percentage (0-100)
    downloaded_bytes: int  # Current downloaded bytes
    total_bytes: int  # Total bytes to download
    download_speed: float  # Bytes per second (average)
    eta_seconds: Optional[int] = None  # Estimated time remaining in seconds
    status: str  # "downloading", "completed", "failed", "not_started"

class ContainerInfo(BaseModel):
    name: str
    model: Optional[str] = None
    status: str

class DeleteStatus(BaseModel):
    status: str

class StartStatus(BaseModel):
    status: str
    container: str

class StopStatus(BaseModel):
    status: str
os.makedirs(model_dir, exist_ok=True)

progress: Dict[str, float] = {}  # model: progress (0-100)
total_sizes: Dict[str, int] = {}  # model: total size
downloaded_sizes: Dict[str, int] = {}  # model: current downloaded size
download_start_times: Dict[str, float] = {}  # model: download start timestamp
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

def format_bytes(bytes_value: int) -> str:
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def format_time(seconds: int) -> str:
    """Format seconds in human readable format"""
    if seconds is None:
        return "Unknown"

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def find_optimized_model_version(model_id: str, options: PullOptions) -> tuple[str, str]:
    """
    Find the best optimized version of a model based on user preferences.
    Returns (optimized_model_id, reason)
    """
    # Common quantization suffixes for HuggingFace models
    quantization_suffixes = {
        QuantizationLevel.FP16: ["-fp16", "-f16"],
        QuantizationLevel.INT8: ["-int8", "-i8", "-gptq-8bit"],
        QuantizationLevel.INT4: ["-int4", "-i4", "-gptq-4bit", "-awq"],
        QuantizationLevel.GPTQ: ["-gptq", "-gptq-4bit"]
    }

    base_model = model_id

    # Try to find pre-quantized versions
    if options.use_optimized_version and options.quantization != QuantizationLevel.FP32:
        suffixes = quantization_suffixes.get(options.quantization, [])

        for suffix in suffixes:
            test_model = f"{base_model}{suffix}"
            try:
                # Try to get model info to see if it exists
                repo_info(repo_id=test_model, repo_type="model")
                return test_model, f"Found pre-quantized version: {options.quantization.value}"
            except:
                continue

    # If no optimized version found or FP32 requested, return original
    return base_model, "Using original model" if options.quantization == QuantizationLevel.FP32 else f"No optimized version found for {options.quantization.value}, using original"

def estimate_model_size(model_id: str, quantization: QuantizationLevel) -> int:
    """
    Estimate model size based on quantization level
    These are rough estimates - actual sizes vary by model architecture
    """
    size_multipliers = {
        QuantizationLevel.FP32: 1.0,
        QuantizationLevel.FP16: 0.5,  # ~50% reduction
        QuantizationLevel.INT8: 0.25,  # ~75% reduction
        QuantizationLevel.INT4: 0.125,  # ~87.5% reduction
        QuantizationLevel.GPTQ: 0.1   # ~90% reduction (very aggressive)
    }

    # Get base model size estimate (this is approximate)
    try:
        r_info = repo_info(repo_id=model_id, repo_type="model", files_metadata=True)
        base_size = sum(sibling.size or 0 for sibling in r_info.siblings if sibling.size is not None)
        return int(base_size * size_multipliers[quantization])
    except:
        # Fallback estimates for common model sizes
        if "20b" in model_id.lower() or "22b" in model_id.lower():
            return int(41e9 * size_multipliers[quantization])  # ~41GB base for 20B models
        elif "70b" in model_id.lower():
            return int(140e9 * size_multipliers[quantization])  # ~140GB base for 70B models
        elif "30b" in model_id.lower():
            return int(60e9 * size_multipliers[quantization])  # ~60GB base for 30B models
        else:
            return int(10e9 * size_multipliers[quantization])  # Generic fallback

def monitor_progress(model: str, local_path: str):
    start_time = download_start_times.get(model, time.time())
    last_size = 0
    last_time = start_time

    while True:
        current_size = calculate_dir_size(local_path)
        current_time = time.time()

        # Update downloaded size
        downloaded_sizes[model] = current_size

        # Calculate progress percentage
        total_size = total_sizes.get(model, 0)
        prog = (current_size / total_size) * 100 if total_size > 0 else 0
        progress[model] = min(prog, 100)

        # Calculate download speed (bytes per second)
        time_diff = current_time - last_time
        size_diff = current_size - last_size
        if time_diff > 0 and size_diff > 0:
            download_speed = size_diff / time_diff
        else:
            download_speed = 0

        # Calculate ETA
        if download_speed > 0 and prog < 100:
            remaining_bytes = total_size - current_size
            eta_seconds = int(remaining_bytes / download_speed)
        else:
            eta_seconds = None

        # Update progress info (we'll store this in a more detailed structure)
        # For now, we'll use the existing progress dict to store the percentage
        # and add other info to a separate structure

        last_size = current_size
        last_time = current_time

        if prog >= 100:
            break
        time.sleep(2)  # Poll every 2 seconds for more responsive updates

def download_model(model: str, options: PullOptions):
    local_path = os.path.join(model_dir, model.replace("/", "_"))
    os.makedirs(local_path, exist_ok=True)

    # Find optimized version if requested
    optimized_model, reason = find_optimized_model_version(model, options)
    print(f"Using model: {optimized_model} ({reason})")

    # Get total size for the selected model
    try:
        r_info = repo_info(repo_id=optimized_model, repo_type="model", files_metadata=True)
        total_size = sum(sibling.size or 0 for sibling in r_info.siblings if sibling.size is not None)
        total_sizes[model] = total_size  # Use original model name as key for tracking
    except Exception:
        total_sizes[model] = 0  # Fallback if size can't be fetched

    progress[model] = 0
    download_start_times[model] = time.time()

    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_progress, args=(model, local_path))
    monitor_thread.start()

    try:
        # Download with size optimization options
        snapshot_download(
            repo_id=optimized_model,
            local_dir=local_path,
            resume_download=True,
            local_dir_use_symlinks=False
        )

        # Calculate final metrics
        final_size = calculate_dir_size(local_path)

        # Wait for monitor to finish
        monitor_thread.join()
        progress[model] = 100

        print(f"Download completed: {optimized_model}")
        print(f"Original estimated size: {format_bytes(total_size)}")
        print(f"Final size: {format_bytes(final_size)}")
        if total_size > 0:
            compression_ratio = final_size / total_size
            print(f"Compression ratio: {compression_ratio:.2%}")

    except Exception as e:
        print(f"Download failed: {e}")
        progress[model] = -1  # Error state
        raise
    finally:
        # Clean up tracking data after completion
        downloaded_sizes.pop(model, None)
        download_start_times.pop(model, None)

@app.get("/pulled",
         response_model=List[str],
         summary="Get list of pulled models",
         description="Returns a list of model names that have been downloaded locally",
         tags=["models"])
def view_pulled_models():
    """Get list of models that have been pulled to local storage"""
    return [d.replace('_', '/') for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]

@app.get("/available",
         response_model=List[ModelInfo],
         summary="Search available models",
         description="Search for available models on Hugging Face Hub. Returns top 20 models sorted by downloads",
         tags=["models"])
def view_available_models(search: str = Query("", description="Search query to filter models")):
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

@app.get("/available/model",
         response_model=DetailedModelInfo,
         summary="Get detailed model information",
         description="Get comprehensive information about a specific model including configuration, metadata, and parameters",
         tags=["models"])
def get_model_details(model_id: str = Query(..., description="Model ID to get details for (e.g., 'openai/gpt-oss-20b')")):
    """Get detailed information for a specific model including description, config, and other metadata."""
    # URL decode the model_id to handle special characters like '/'
    decoded_model_id = unquote(model_id)

    try:
        # Get basic model info from search results
        models = api.list_models(search=decoded_model_id, limit=1)
        basic_model = None
        for m in models:
            if m.id == decoded_model_id:
                basic_model = m
                break

        if not basic_model:
            raise HTTPException(status_code=404, detail="Model not found")

        # Get detailed repo info
        repo_details = repo_info(repo_id=decoded_model_id, repo_type="model", files_metadata=False)

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

@app.get("/estimate-size",
         summary="Estimate model size for different quantization levels",
         description="Get size estimates for a model at different quantization levels to help choose optimal settings",
         tags=["models"])
def estimate_model_sizes(
    model: str = Query(..., description="Model name to estimate sizes for (e.g., 'openai/gpt-oss-20b')")
):
    """Estimate model sizes for different quantization levels"""
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)
    estimates = {}

    for quant_level in QuantizationLevel:
        try:
            size_bytes = estimate_model_size(decoded_model, quant_level)
            size_formatted = format_bytes(size_bytes)
            estimates[quant_level.value] = {
                "size_bytes": size_bytes,
                "size_formatted": size_formatted,
                "reduction_percent": round((1 - size_bytes / estimate_model_size(decoded_model, QuantizationLevel.FP32)) * 100, 1) if quant_level != QuantizationLevel.FP32 else 0
            }
        except Exception as e:
            estimates[quant_level.value] = {
                "error": f"Could not estimate size: {str(e)}"
            }

    return {
        "model": decoded_model,
        "estimates": estimates,
        "note": "These are estimates based on typical model architectures. Actual sizes may vary."
    }

@app.post("/pull",
         response_model=PullStatus,
         summary="Download a model with size optimization",
         description="Download a model from Hugging Face Hub with options for quantization and size reduction. This is an asynchronous operation that runs in the background",
         tags=["models"])
def pull_model(
    background_tasks: BackgroundTasks,
    model: str = Query(..., description="Base model name to download (e.g., 'openai/gpt-oss-20b')"),
    quantization: QuantizationLevel = Query(QuantizationLevel.FP16, description="Quantization level for size reduction"),
    format: ModelFormat = Query(ModelFormat.SAFETENSORS, description="Model file format"),
    use_optimized_version: bool = Query(True, description="Try to find pre-optimized/quantized versions")
):
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)

    if decoded_model in current_threads and current_threads[decoded_model].is_alive():
        raise HTTPException(status_code=400, detail="Download already in progress")

    # Create options object
    options = PullOptions(
        quantization=quantization,
        format=format,
        use_optimized_version=use_optimized_version
    )

    thread = threading.Thread(target=download_model, args=(decoded_model, options))
    current_threads[decoded_model] = thread
    thread.start()
    background_tasks.add_task(lambda: thread.join() and current_threads.pop(decoded_model, None))

    # Estimate final size
    estimated_size = estimate_model_size(decoded_model, quantization)

    return {
        "status": "started",
        "model_path": f"/models/{decoded_model.replace('/', '_')}",
        "original_size": None,  # Will be calculated during download
        "final_size": estimated_size,
        "compression_ratio": None  # Will be calculated after download
    }

@app.get("/progress",
         response_model=ProgressInfo,
         summary="Get detailed download progress",
         description="Get comprehensive download progress including bytes downloaded, total size, speed, and ETA",
         tags=["models"])
def get_download_progress(model: str = Query(..., description="Model name to check progress for")):
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)

    current_progress = progress.get(decoded_model, 0)
    downloaded = downloaded_sizes.get(decoded_model, 0)
    total = total_sizes.get(decoded_model, 0)
    start_time = download_start_times.get(decoded_model, None)

    # Determine status
    if current_progress >= 100:
        status = "completed"
    elif current_progress > 0:
        status = "downloading"
    elif decoded_model in current_threads and current_threads[decoded_model].is_alive():
        status = "starting"
    else:
        status = "not_started"

    # Calculate average speed and ETA if we have timing data
    download_speed = 0.0
    eta_seconds = None

    if start_time and downloaded > 0:
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            download_speed = downloaded / elapsed_time

        if download_speed > 0 and current_progress < 100:
            remaining_bytes = total - downloaded
            eta_seconds = int(remaining_bytes / download_speed)

    return {
        "progress": current_progress,
        "downloaded_bytes": downloaded,
        "total_bytes": total,
        "download_speed": download_speed,
        "eta_seconds": eta_seconds,
        "status": status
    }

@app.get("/running",
         response_model=List[ContainerInfo],
         summary="Get running model containers",
         description="Get information about currently running vLLM containers",
         tags=["containers"])
def show_running_models():
    running = []
    containers = docker_client.containers.list(filters={"name": "vllm-"})
    for cont in containers:
        name = cont.name.replace("vllm-", "")
        model = None

        # Find the --model argument and extract the path
        cmd_args = cont.attrs["Config"]["Cmd"]
        for i, cmd in enumerate(cmd_args):
            if cmd == "--model" and i + 1 < len(cmd_args):
                model = cmd_args[i + 1]
                break

        status = cont.status
        running.append({"name": name, "model": model, "status": status})
    return running

@app.delete("/delete",
           response_model=DeleteStatus,
           summary="Delete a local model",
           description="Delete a locally stored model and its files",
           tags=["models"])
def delete_model(model: str = Query(..., description="Model name to delete")):
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)
    local_path = os.path.join(model_dir, decoded_model.replace("/", "_"))

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Model not found")

# Bonus: Start a model (assuming you have GPUs available; adjust as needed)
@app.post("/start",
         response_model=StartStatus,
         summary="Start a model container",
         description="Start a vLLM container for the specified model",
         tags=["containers"])
def start_model(model: str = Query(..., description="Model name to start"), tensor_parallel_size: int = Query(1, description="Number of GPUs to use for tensor parallelism")):
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)
    local_path = os.path.join(model_dir, decoded_model.replace("/", "_"))
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Model not pulled")

    container_name = f"vllm-{decoded_model.replace('/', '_')}"
    if any(c.name == container_name for c in docker_client.containers.list(all=True)):
        raise HTTPException(status_code=400, detail="Container already exists")

    # Run vLLM container with multi-GPU support
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
        device_requests=[
            {
                "Driver": "nvidia",
                "Count": tensor_parallel_size,
                "Capabilities": [["gpu"]],
                "Options": {}
            }
        ] if tensor_parallel_size > 0 else None,
        detach=True,
        environment={
            "NVIDIA_VISIBLE_DEVICES": "0,1" if tensor_parallel_size >= 2 else "0"
        } if tensor_parallel_size > 0 else {}
    )
    return {"status": "started", "container": container_name}

# Bonus: Stop a model
@app.post("/stop",
         response_model=StopStatus,
         summary="Stop a model container",
         description="Stop and remove a running vLLM container for the specified model",
         tags=["containers"])
def stop_model(model: str = Query(..., description="Model name to stop")):
    # URL decode the model name to handle special characters like '/'
    decoded_model = unquote(model)
    container_name = f"vllm-{decoded_model.replace('/', '_')}"
    try:
        cont = docker_client.containers.get(container_name)
        cont.stop()
        cont.remove()
        return {"status": "stopped"}
    except docker.errors.NotFound:
        raise HTTPException(status_code=404, detail="Container not found")