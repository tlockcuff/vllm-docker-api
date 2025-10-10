import type { Request, Response } from 'express';
import { VLLM_PORT } from '../config.js';

export async function proxyJson(path: string, payload: any) {
  const axios = (await import('axios')).default;
  const url = `http://localhost:${VLLM_PORT}${path}`;
  const { data } = await axios.post(url, payload);
  return data;
}

export async function proxyJsonToPort(port: number, path: string, payload: any) {
  const axios = (await import('axios')).default;
  const url = `http://localhost:${port}${path}`;
  const { data } = await axios.post(url, payload);
  return data;
}

export async function streamSSE(req: Request, res: Response, path: string, payload: any) {
  const axios = (await import('axios')).default;
  const url = `http://localhost:${VLLM_PORT}${path}`;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const upstream = await axios.post(url, payload, { responseType: 'stream' });

  req.on('close', () => upstream.data.destroy());
  upstream.data.on('data', (chunk: Buffer) => res.write(chunk));
  upstream.data.on('end', () => res.end());
  upstream.data.on('error', () => res.end());
}

export async function streamSSEToPort(req: Request, res: Response, port: number, path: string, payload: any) {
  const axios = (await import('axios')).default;
  const url = `http://localhost:${port}${path}`;

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const upstream = await axios.post(url, payload, { responseType: 'stream' });

  req.on('close', () => upstream.data.destroy());
  upstream.data.on('data', (chunk: Buffer) => res.write(chunk));
  upstream.data.on('end', () => res.end());
  upstream.data.on('error', () => res.end());
}

