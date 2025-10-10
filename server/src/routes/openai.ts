import type { Express, Request, Response } from 'express';
import { registry } from '../openapi.js';
import { DEFAULT_MODEL, VLLM_CONTAINER, VLLM_PORT } from '../config.js';
import { ensureVllm, containerRunning } from '../services/docker.js';
import { proxyJson, streamSSE } from '../services/proxy.js';
import { ChatResponseSchema, CompletionsResponseSchema, EmbeddingsResponseSchema, ModelsResponseSchema, OpenAIChatRequestSchema, OpenAICompletionsRequestSchema, OpenAIEmbeddingsRequestSchema } from '../schemas.js';
import { getLocalHuggingFaceModels } from '../services/downloads.js';

export function mountOpenAIRoutes(app: Express) {
  // Register OpenAPI paths
  registry.registerPath({
    method: 'post',
    path: '/v1/chat/completions',
    request: { body: { content: { 'application/json': { schema: OpenAIChatRequestSchema } } } },
    responses: {
      200: { description: 'Chat completion', content: { 'application/json': { schema: ChatResponseSchema } } },
      207: { description: 'SSE stream', content: { 'text/event-stream': { schema: { type: 'string' } } } },
    },
  });

  registry.registerPath({
    method: 'post',
    path: '/v1/completions',
    request: { body: { content: { 'application/json': { schema: OpenAICompletionsRequestSchema } } } },
    responses: {
      200: { description: 'Text completion', content: { 'application/json': { schema: CompletionsResponseSchema } } },
      207: { description: 'SSE stream', content: { 'text/event-stream': { schema: { type: 'string' } } } },
    },
  });

  registry.registerPath({
    method: 'post',
    path: '/v1/embeddings',
    request: { body: { content: { 'application/json': { schema: OpenAIEmbeddingsRequestSchema } } } },
    responses: { 200: { description: 'Embeddings response', content: { 'application/json': { schema: EmbeddingsResponseSchema } } } },
  });

  registry.registerPath({ method: 'get', path: '/v1/models', responses: { 200: { description: 'List models', content: { 'application/json': { schema: ModelsResponseSchema } } } } });

  app.post('/v1/chat/completions', async (req: Request, res: Response) => {
    try {
      const bodyData = OpenAIChatRequestSchema.parse(req.body);
      const queryModel = req.query.model as string;
      const model = queryModel || bodyData.model || DEFAULT_MODEL;
      const requestData = { ...bodyData, model };
      await ensureVllm(model);
      if (requestData.stream) {
        return await streamSSE(req, res, '/v1/chat/completions', requestData);
      }
      const data = await proxyJson('/v1/chat/completions', requestData);
      const response = ChatResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });

  app.post('/v1/completions', async (req: Request, res: Response) => {
    try {
      const bodyData = OpenAICompletionsRequestSchema.parse(req.body);
      const queryModel = req.query.model as string;
      const model = queryModel || bodyData.model || DEFAULT_MODEL;
      const requestData = { ...bodyData, model };
      await ensureVllm(model);
      if (requestData.stream) {
        return await streamSSE(req, res, '/v1/completions', requestData);
      }
      const data = await proxyJson('/v1/completions', requestData);
      const response = CompletionsResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });

  app.post('/v1/embeddings', async (req: Request, res: Response) => {
    try {
      const bodyData = OpenAIEmbeddingsRequestSchema.parse(req.body);
      const queryModel = req.query.model as string;
      const model = queryModel || bodyData.model;
      const requestData = { ...bodyData, model };
      await ensureVllm(model);
      const data = await proxyJson('/v1/embeddings', requestData);
      const response = EmbeddingsResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });

  app.get('/v1/models', async (_req: Request, res: Response) => {
    try {
      const localModels = getLocalHuggingFaceModels();
      let vllmModels: Array<{ id: string; object: string; created: number; owned_by: string }> = [];
      try {
        const isRunning = await containerRunning(VLLM_CONTAINER);
        if (isRunning) {
          const axios = (await import('axios')).default;
          const url = `http://localhost:${VLLM_PORT}/v1/models`;
          const { data } = await axios.get(url);
          if (data.data && Array.isArray(data.data)) {
            vllmModels = data.data.map((model: any) => ({ id: model.id, object: 'model', created: model.created || Math.floor(Date.now() / 1000), owned_by: model.owned_by || 'vllm' }));
          }
        }
      } catch {}
      const allModels = [...localModels];
      for (const vllmModel of vllmModels) {
        if (!allModels.find(m => m.id === vllmModel.id)) {
          allModels.push(vllmModel);
        }
      }
      const response = ModelsResponseSchema.parse({ object: 'list', data: allModels });
      res.json(response);
    } catch (e) {
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });
}


