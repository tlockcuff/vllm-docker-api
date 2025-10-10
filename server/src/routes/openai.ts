import type { Express, Request, Response } from 'express';
import { logger } from '../logger.js';
import { registry } from '../openapi.js';
import { ChatResponseSchema, CompletionsResponseSchema, EmbeddingsResponseSchema, ModelsResponseSchema, OpenAIChatRequestSchema, OpenAICompletionsRequestSchema, OpenAIEmbeddingsRequestSchema } from '../schemas.js';
import { ensureVllmForModel } from '../services/docker.js';
import { getLocalHuggingFaceModels } from '../services/downloads.js';
import { proxyJsonToPort, streamSSEToPort } from '../services/proxy.js';

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
      const model = queryModel || bodyData.model;
      const requestData = { ...bodyData, model };
      const { port } = await ensureVllmForModel(model);
      if (requestData.stream) {
        return await streamSSEToPort(req, res, port, '/v1/chat/completions', requestData);
      }
      const data = await proxyJsonToPort(port, '/v1/chat/completions', requestData);
      const response = ChatResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      logger.error('route_error', { route: '/v1/chat/completions', method: req.method, path: req.originalUrl, errorMessage: String(e), issues: (e as any)?.issues });
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
      const model = queryModel || bodyData.model;
      const requestData = { ...bodyData, model };
      const { port } = await ensureVllmForModel(model);
      if (requestData.stream) {
        return await streamSSEToPort(req, res, port, '/v1/completions', requestData);
      }
      const data = await proxyJsonToPort(port, '/v1/completions', requestData);
      const response = CompletionsResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      logger.error('route_error', { route: '/v1/completions', method: req.method, path: req.originalUrl, errorMessage: String(e), issues: (e as any)?.issues });
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
      const { port } = await ensureVllmForModel(model);
      const data = await proxyJsonToPort(port, '/v1/embeddings', requestData);
      const response = EmbeddingsResponseSchema.parse(data);
      res.json(response);
    } catch (e) {
      logger.error('route_error', { route: '/v1/embeddings', method: req.method, path: req.originalUrl, errorMessage: String(e), issues: (e as any)?.issues });
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });

  app.get('/v1/models', async (req: Request, res: Response) => {
    try {
      const localModels = getLocalHuggingFaceModels();
      let vllmModels: Array<{ id: string; object: string; created: number; owned_by: string }> = [];

      const allModels = [...localModels];
      for (const vllmModel of vllmModels) {
        if (!allModels.find(m => m.id === vllmModel.id)) {
          allModels.push(vllmModel);
        }
      }
      const response = ModelsResponseSchema.parse({ object: 'list', data: allModels });
      res.json(response);
    } catch (e) {
      logger.error('route_error', { route: '/v1/models', method: req.method, path: req.originalUrl, errorMessage: String(e), issues: (e as any)?.issues });
      if ((e as any).issues) {
        res.status(400).json({ error: { message: 'Invalid request data', type: 'invalid_request_error', param: null, code: null }, details: (e as any).issues });
      } else {
        res.status(500).json({ error: { message: String(e), type: 'internal_error', param: null, code: null } });
      }
    }
  });
}


