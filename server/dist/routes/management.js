import { registry } from '../openapi.js';
import { DEFAULT_MODEL, VLLM_CONTAINER, VLLM_IMAGE, VLLM_PORT } from '../config.js';
import { ensureVllm, containerExists, containerRunning, runDocker } from '../services/docker.js';
import { HealthResponseSchema, RemoveResponseSchema, StartRequestSchema, StartResponseSchema, StatusResponseSchema, StopResponseSchema } from '../schemas.js';
export function mountManagementRoutes(app) {
    // OpenAPI registrations
    registry.registerPath({ method: 'get', path: '/api/status', responses: { 200: { description: 'Container status', content: { 'application/json': { schema: StatusResponseSchema } } } } });
    registry.registerPath({ method: 'post', path: '/api/start', request: { body: { content: { 'application/json': { schema: StartRequestSchema } } } }, responses: { 200: { description: 'Start vLLM container', content: { 'application/json': { schema: StartResponseSchema } } } } });
    registry.registerPath({ method: 'post', path: '/api/stop', responses: { 200: { description: 'Stop vLLM container', content: { 'application/json': { schema: StopResponseSchema } } } } });
    registry.registerPath({ method: 'delete', path: '/api/remove', responses: { 200: { description: 'Remove vLLM container', content: { 'application/json': { schema: RemoveResponseSchema } } } } });
    registry.registerPath({ method: 'get', path: '/api/health', responses: { 200: { description: 'Health check', content: { 'application/json': { schema: HealthResponseSchema } } } } });
    app.get('/api/status', async (_req, res) => {
        try {
            const running = await containerRunning(VLLM_CONTAINER);
            const response = StatusResponseSchema.parse({ container: VLLM_CONTAINER, running, port: VLLM_PORT, image: VLLM_IMAGE });
            res.json(response);
        }
        catch (e) {
            res.status(500).json({ error: String(e) });
        }
    });
    app.post('/api/start', async (req, res) => {
        try {
            const requestData = StartRequestSchema.parse(req.body);
            const model = requestData.model && requestData.model.length > 0 ? requestData.model : DEFAULT_MODEL;
            await ensureVllm(model);
            const response = StartResponseSchema.parse({ ok: true, model });
            res.json(response);
        }
        catch (e) {
            if (e.issues) {
                res.status(400).json({ error: 'Invalid request data', details: e.issues });
            }
            else {
                res.status(500).json({ error: String(e) });
            }
        }
    });
    app.post('/api/stop', async (_req, res) => {
        try {
            const exists = await containerExists(VLLM_CONTAINER);
            if (!exists) {
                const response = StopResponseSchema.parse({ ok: true, stopped: false, message: 'not found' });
                return res.json(response);
            }
            const running = await containerRunning(VLLM_CONTAINER);
            if (!running) {
                const response = StopResponseSchema.parse({ ok: true, stopped: false, message: 'already stopped' });
                return res.json(response);
            }
            await runDocker(['stop', VLLM_CONTAINER]);
            const response = StopResponseSchema.parse({ ok: true, stopped: true });
            res.json(response);
        }
        catch (e) {
            res.status(500).json({ error: String(e) });
        }
    });
    app.delete('/api/remove', async (_req, res) => {
        try {
            const exists = await containerExists(VLLM_CONTAINER);
            if (!exists) {
                const response = RemoveResponseSchema.parse({ ok: true, removed: false, message: 'not found' });
                return res.json(response);
            }
            const running = await containerRunning(VLLM_CONTAINER);
            if (running)
                await runDocker(['stop', VLLM_CONTAINER]);
            await runDocker(['rm', VLLM_CONTAINER]);
            const response = RemoveResponseSchema.parse({ ok: true, removed: true });
            res.json(response);
        }
        catch (e) {
            res.status(500).json({ error: String(e) });
        }
    });
    app.get('/api/health', (_req, res) => {
        const response = HealthResponseSchema.parse({ ok: true });
        res.json(response);
    });
}
