import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import { DEFAULT_MODEL, VLLM_CONTAINER, VLLM_IMAGE, VLLM_PORT, VLLM_USE_GPU } from "../config.js";
import { logger } from "../logger.js";
import type { ChildProcessWithoutNullStreams } from "node:child_process";

const execFileAsync = promisify(execFile);

export async function runDocker(args: string[]) {
  const { stdout, stderr } = await execFileAsync("docker", args, { env: process.env });
  if (stderr && stderr.trim().length > 0) {
    // Docker prints progress to stderr sometimes; not necessarily an error.
  }
  return stdout.trim();
}

export async function containerExists(name: string) {
  const output = await runDocker(["ps", "-a", "--format", "{{.Names}}"]);
  return output.split("\n").includes(name);
}

export async function containerRunning(name: string) {
  const output = await runDocker(["ps", "--format", "{{.Names}}"]);
  return output.split("\n").includes(name);
}

export async function ensureVllm(model: string = DEFAULT_MODEL) {
  const exists = await containerExists(VLLM_CONTAINER);
  if (!exists) {
    const envArgs: string[] = [];
    if (process.env.VLLM_LOGGING_LEVEL) {
      envArgs.push("-e", `VLLM_LOGGING_LEVEL=${process.env.VLLM_LOGGING_LEVEL}`);
    }
    if (process.env.HUGGING_FACE_HUB_TOKEN) {
      envArgs.push("-e", `HUGGING_FACE_HUB_TOKEN=${process.env.HUGGING_FACE_HUB_TOKEN}`);
    }
    if (process.env.VLLM_DEVICE) {
      envArgs.push("-e", `VLLM_DEVICE=${process.env.VLLM_DEVICE}`);
    } else if (!VLLM_USE_GPU) {
      // Default to CPU when GPU is not explicitly requested/available
      envArgs.push("-e", "VLLM_DEVICE=cpu");
    }

    const deviceCliArgs: string[] = [];
    if (process.env.VLLM_DEVICE) {
      deviceCliArgs.push("--device", process.env.VLLM_DEVICE);
    } else if (!VLLM_USE_GPU) {
      deviceCliArgs.push("--device", "cpu");
    }

    const dtypeCliArgs: string[] = [];
    if (process.env.VLLM_DTYPE) {
      dtypeCliArgs.push("--dtype", process.env.VLLM_DTYPE);
    } else if (VLLM_USE_GPU) {
      dtypeCliArgs.push("--dtype", "float16");
    }

    // Tensor parallel size via env fallback (for base container)
    let envTpSize: number | undefined;
    if (process.env.VLLM_TP_SIZE) {
      const n = Number(process.env.VLLM_TP_SIZE);
      if (Number.isInteger(n) && n > 1) envTpSize = n;
    }

    const tensorCliArgs: string[] = [];
    if (VLLM_USE_GPU && envTpSize) {
      tensorCliArgs.push("--tensor-parallel-size", String(envTpSize));
    }

    const args = [
      "run",
      "-d",
      "--restart",
      "unless-stopped",
      "--name",
      VLLM_CONTAINER,
      "-p",
      `${VLLM_PORT}:8000`,
      ...envArgs,
      ...(VLLM_USE_GPU ? ["--gpus", "all"] : []),
      VLLM_IMAGE,
      ...deviceCliArgs,
      ...dtypeCliArgs,
      ...tensorCliArgs,
      "--model",
      model,
    ];
    await runDocker(args);
    ensureLogStreaming(VLLM_CONTAINER);
    return;
  }
  const running = await containerRunning(VLLM_CONTAINER);
  if (!running) {
    await runDocker(["start", VLLM_CONTAINER]);
    ensureLogStreaming(VLLM_CONTAINER);
  } else {
    ensureLogStreaming(VLLM_CONTAINER);
  }
}

// Multi-model support helpers

function slugifyModel(model: string): string {
  const lower = model.toLowerCase();
  const replaced = lower.replace(/[^a-z0-9]+/g, "-");
  const trimmed = replaced.replace(/^-+|-+$/g, "");
  return trimmed || "model";
}

export function getContainerNameForModel(model: string): string {
  const slug = slugifyModel(model);
  // Use base container as prefix for discoverability
  return `${VLLM_CONTAINER}-${slug}`;
}

export async function getHostPort(containerName: string): Promise<number> {
  // Prefer docker inspect to extract HostPort for 8000/tcp
  try {
    const output = await runDocker(["inspect", "-f", '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}', containerName]);
    const port = Number(String(output).trim());
    if (!Number.isNaN(port) && port > 0) return port;
  } catch {}
  // Fallback to `docker port` parsing
  try {
    const portOut = await runDocker(["port", containerName, "8000/tcp"]);
    // Example: "0.0.0.0:32772" or ":::32772"; take the last colon segment
    const firstLine = portOut.split("\n").find(Boolean) || "";
    const parts = firstLine.trim().split(":");
    const last = parts[parts.length - 1];
    const port = Number(last);
    if (!Number.isNaN(port) && port > 0) return port;
  } catch {}
  return VLLM_PORT;
}

export async function ensureVllmForModel(model: string): Promise<{ name: string; port: number }> {
  const name = getContainerNameForModel(model);
  const exists = await containerExists(name);
  if (!exists) {
    const envArgs: string[] = [];
    if (process.env.VLLM_LOGGING_LEVEL) {
      envArgs.push("-e", `VLLM_LOGGING_LEVEL=${process.env.VLLM_LOGGING_LEVEL}`);
    }
    if (process.env.HUGGING_FACE_HUB_TOKEN) {
      envArgs.push("-e", `HUGGING_FACE_HUB_TOKEN=${process.env.HUGGING_FACE_HUB_TOKEN}`);
    }
    if (process.env.VLLM_DEVICE) {
      envArgs.push("-e", `VLLM_DEVICE=${process.env.VLLM_DEVICE}`);
    } else if (!VLLM_USE_GPU) {
      // Default to CPU when GPU is not explicitly requested/available
      envArgs.push("-e", "VLLM_DEVICE=cpu");
    }

    const vllmArgs: string[] = [];
    // https://docs.vllm.ai/en/v0.4.3/models/engine_args.html
    vllmArgs.push("--device", "gpu");
    vllmArgs.push("--dtype", "float16");
    vllmArgs.push("--kv-cache-dtype", "auto");
    vllmArgs.push("--gpu-memory-utilization", "0.95");
    vllmArgs.push("--tensor-parallel-size", "2");
    vllmArgs.push("--quantization", "fp8");
    vllmArgs.push("--max-num-seqs", "128");

    const args = [
      "run",
      "-d",
      "--restart",
      "unless-stopped",
      "--name",
      name,
      // Let Docker assign a random available host port; we'll discover it via inspect
      "-p",
      "0:8000",
      ...envArgs,
      ...(VLLM_USE_GPU ? ["--gpus", "all"] : []),
      VLLM_IMAGE,
      ...vllmArgs,
      "--model",
      model,
    ];
    await runDocker(args);
    ensureLogStreaming(name);
  } else {
    const running = await containerRunning(name);
    if (!running) {
      await runDocker(["start", name]);
      ensureLogStreaming(name);
    } else {
      ensureLogStreaming(name);
    }
  }
  const port = await getHostPort(name);
  return { name, port };
}

// --- Log streaming management ---
const logStreams = new Map<string, ChildProcessWithoutNullStreams>();

export function ensureLogStreaming(containerName: string) {
  if (logStreams.has(containerName)) return;
  try {
    const child = spawn("docker", ["logs", "-f", containerName], { env: process.env });
    logStreams.set(containerName, child);

    const handleChunk = (chunk: Buffer | string, stream: "stdout" | "stderr") => {
      const text = chunk.toString("utf8");
      const lines = text.split(/\r?\n/);
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed.length === 0) continue;
        if (stream === "stdout") {
          logger.info("vllm_container_log", { l: trimmed });
        } else {
          logger.warn("vllm_container_log", { l: trimmed });
        }
      }
    };

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk) => handleChunk(chunk, "stdout"));
    child.stderr.on("data", (chunk) => handleChunk(chunk, "stderr"));
    child.on("error", (err) => {
      logger.error("docker_logs_stream_error", { errorMessage: (err as Error).message });
    });
    child.on("close", (code, signal) => {
      logger.info("docker_logs_stream_closed", { code, signal });
      logStreams.delete(containerName);
    });
  } catch (err) {
    logger.error("docker_logs_spawn_failed", { errorMessage: err instanceof Error ? err.message : String(err) });
  }
}

export function stopLogStreaming(containerName: string) {
  const child = logStreams.get(containerName);
  if (child) {
    try {
      child.kill("SIGTERM");
    } catch {}
    logStreams.delete(containerName);
  }
}
