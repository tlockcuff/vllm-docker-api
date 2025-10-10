import { VLLM_PORT } from '../config.js';
export async function proxyJson(path, payload) {
    const axios = (await import('axios')).default;
    const url = `http://localhost:${VLLM_PORT}${path}`;
    const { data } = await axios.post(url, payload);
    return data;
}
export async function streamSSE(req, res, path, payload) {
    const axios = (await import('axios')).default;
    const url = `http://localhost:${VLLM_PORT}${path}`;
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    const upstream = await axios.post(url, payload, { responseType: 'stream' });
    req.on('close', () => upstream.data.destroy());
    upstream.data.on('data', (chunk) => res.write(chunk));
    upstream.data.on('end', () => res.end());
    upstream.data.on('error', () => res.end());
}
