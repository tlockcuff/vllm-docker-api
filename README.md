# vLLM Multi-Model Docker Compose Stack

Run multiple vLLM model servers on a Linux host with NVIDIA GPUs, managed via a FastAPI Admin API and exposed behind an OpenAI-compatible Gateway.

## Components
- management-api: lifecycle, HF downloads, GPU-aware orchestration of vLLM workers
- gateway: OpenAI-compatible proxy that routes by `model` to the correct worker

## Prereqs
- NVIDIA driver + nvidia-container-toolkit installed
- Docker Compose v2

## Quick start
1. Export tokens (optional but recommended):
   - `export ADMIN_TOKEN=changeme`
   - `export GATEWAY_API_KEY=changeme`
2. Start services:
   - `docker compose up -d --build`
3. Download models (optional pre-cache):
   - `curl -H "Authorization: Bearer $ADMIN_TOKEN" -X POST http://localhost:8081/models/download -H 'Content-Type: application/json' -d '{"repo_id":"meta-llama/Llama-3.1-8B-Instruct"}'`
4. Start workers (one per GPU):
   - GPU0: `curl -H "Authorization: Bearer $ADMIN_TOKEN" -X POST http://localhost:8081/instances -H 'Content-Type: application/json' -d '{"name":"llama3-gpu0","model":"meta-llama/Llama-3.1-8B-Instruct","gpu":0}'`
   - GPU1: `curl -H "Authorization: Bearer $ADMIN_TOKEN" -X POST http://localhost:8081/instances -H 'Content-Type: application/json' -d '{"name":"mistral-gpu1","model":"mistralai/Mistral-7B-Instruct-v0.2","gpu":1}'`
5. List models via Gateway:
   - `curl -H "Authorization: Bearer $GATEWAY_API_KEY" http://localhost:8000/v1/models`
6. Chat completion:
   - `curl -H "Authorization: Bearer $GATEWAY_API_KEY" -H 'Content-Type: application/json' http://localhost:8000/v1/chat/completions -d '{"model":"llama3-gpu0","messages":[{"role":"user","content":"Hello!"}]}'`

## Notes
- Workers run on an internal network only; Gateway is the single public interface.
- Set `tensor_parallel` to 2 in the instance request to span two GPUs for a large model.
- Model cache stored under the `vllm_models` volume.
