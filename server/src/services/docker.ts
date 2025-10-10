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

