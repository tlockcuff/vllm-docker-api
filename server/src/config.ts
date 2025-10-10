export const VLLM_IMAGE = process.env.VLLM_IMAGE || 'vllm/vllm-openai:latest';
export const VLLM_PORT = Number(process.env.VLLM_PORT || 8000);
export const VLLM_CONTAINER = process.env.VLLM_CONTAINER || 'vllm-openai';
export const DEFAULT_MODEL = process.env.VLLM_MODEL || 'mistralai/Mistral-7B-Instruct-v0.3';
export const VLLM_USE_GPU = process.env.VLLM_USE_GPU === '1';
export const PORT = Number(process.env.PORT || 3000);

export const IS_PRODUCTION = (process.env.NODE_ENV || '').toLowerCase() === 'production';

