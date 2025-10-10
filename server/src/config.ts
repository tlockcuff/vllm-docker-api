export const VLLM_IMAGE = process.env.VLLM_IMAGE || 'vllm/vllm-openai:latest';
export const VLLM_PORT = Number(process.env.VLLM_PORT || 8000);
export const VLLM_CONTAINER = process.env.VLLM_CONTAINER || 'vllm-openai';
export const PORT = Number(process.env.PORT || 3000);

export const IS_PRODUCTION = (process.env.NODE_ENV || '').toLowerCase() === 'production';

