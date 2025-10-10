import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { DEFAULT_MODEL, VLLM_CONTAINER, VLLM_IMAGE, VLLM_PORT, VLLM_USE_GPU } from '../config.js';

const execFileAsync = promisify(execFile);

export async function runDocker(args: string[]) {
  const { stdout, stderr } = await execFileAsync('docker', args, { env: process.env });
  if (stderr && stderr.trim().length > 0) {
    // Docker prints progress to stderr sometimes; not necessarily an error.
  }
  return stdout.trim();
}

export async function containerExists(name: string) {
  const output = await runDocker(['ps', '-a', '--format', '{{.Names}}']);
  return output.split('\n').includes(name);
}

export async function containerRunning(name: string) {
  const output = await runDocker(['ps', '--format', '{{.Names}}']);
  return output.split('\n').includes(name);
}

export async function ensureVllm(model: string = DEFAULT_MODEL) {
  const exists = await containerExists(VLLM_CONTAINER);
  if (!exists) {
    const args = [
      'run', '-d', '--restart', 'unless-stopped', '--name', VLLM_CONTAINER,
      '-p', `${VLLM_PORT}:8000`,
      ...(VLLM_USE_GPU ? ['--gpus', 'all'] : []),
      VLLM_IMAGE,
      '--model', model,
    ];
    await runDocker(args);
    return;
  }
  const running = await containerRunning(VLLM_CONTAINER);
  if (!running) {
    await runDocker(['start', VLLM_CONTAINER]);
  }
}

// Multi-model support helpers

function slugifyModel(model: string): string {
  const lower = model.toLowerCase();
  const replaced = lower.replace(/[^a-z0-9]+/g, '-');
  const trimmed = replaced.replace(/^-+|-+$/g, '');
  return trimmed || 'model';
}

export function getContainerNameForModel(model: string): string {
  const slug = slugifyModel(model);
  // Use base container as prefix for discoverability
  return `${VLLM_CONTAINER}-${slug}`;
}

export async function getHostPort(containerName: string): Promise<number> {
  // Prefer docker inspect to extract HostPort for 8000/tcp
  try {
    const output = await runDocker([
      'inspect',
      '-f',
      '{{(index (index .NetworkSettings.Ports "8000/tcp") 0).HostPort}}',
      containerName,
    ]);
    const port = Number(String(output).trim());
    if (!Number.isNaN(port) && port > 0) return port;
  } catch {}
  // Fallback to `docker port` parsing
  try {
    const portOut = await runDocker(['port', containerName, '8000/tcp']);
    // Example: "0.0.0.0:32772" or ":::32772"; take the last colon segment
    const firstLine = portOut.split('\n').find(Boolean) || '';
    const parts = firstLine.trim().split(':');
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
    const args = [
      'run', '-d', '--restart', 'unless-stopped', '--name', name,
      // Let Docker assign a random available host port; we'll discover it via inspect
      '-p', '0:8000',
      ...(VLLM_USE_GPU ? ['--gpus', 'all'] : []),
      VLLM_IMAGE,
      '--model', model,
    ];
    await runDocker(args);
  } else {
    const running = await containerRunning(name);
    if (!running) {
      await runDocker(['start', name]);
    }
  }
  const port = await getHostPort(name);
  return { name, port };
}

