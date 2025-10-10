import type { Express, Request, Response } from "express";
import { registry } from "../openapi.js";
import { DEFAULT_MODEL, PORT, VLLM_CONTAINER, VLLM_IMAGE, VLLM_PORT } from "../config.js";
import {
  ensureVllm,
  containerExists,
  containerRunning,
  runDocker,
  ensureVllmForModel,
  getContainerNameForModel,
  getHostPort,
  stopLogStreaming,
} from "../services/docker.js";
import {
  HealthResponseSchema,
  RemoveResponseSchema,
  StartRequestSchema,
  StartResponseSchema,
  StatusResponseSchema,
  StopResponseSchema,
} from "../schemas.js";
import { logger } from "../logger.js";

export function mountManagementRoutes(app: Express) {
  // OpenAPI registrations
  registry.registerPath({
    method: "post",
    path: "/api/start",
    request: { body: { content: { "application/json": { schema: StartRequestSchema } } } },
    responses: { 200: { description: "Start vLLM container", content: { "application/json": { schema: StartResponseSchema } } } },
  });
  registry.registerPath({
    method: "get",
    path: "/api/health",
    responses: { 200: { description: "Health check", content: { "application/json": { schema: HealthResponseSchema } } } },
  });
  registry.registerPath({
    method: "post",
    path: "/api/stop/{model}",
    parameters: [{ name: "model", in: "path", required: true }],
    responses: { 200: { description: "Stop model-scoped vLLM container", content: { "application/json": { schema: StopResponseSchema } } } },
  });
  registry.registerPath({
    method: "delete",
    path: "/api/remove/{model}",
    parameters: [{ name: "model", in: "path", required: true }],
    responses: { 200: { description: "Remove model-scoped vLLM container", content: { "application/json": { schema: RemoveResponseSchema } } } },
  });

  app.post("/api/start", async (req: Request, res: Response) => {
    try {
      const requestData = StartRequestSchema.parse(req.body);
      const model = requestData.model && requestData.model.length > 0 ? requestData.model : DEFAULT_MODEL;
      const { tensorParallelSize, dtype, enableSleepMode, cpuOffloadGb } = requestData as any;
      await ensureVllmForModel(model, {
        tensorParallelSize,
        dtype,
        enableSleepMode,
        cpuOffloadGb,
      });
      const response = StartResponseSchema.parse({ ok: true, model });
      res.json(response);
    } catch (e) {
      logger.error("route_error", {
        route: "/api/start",
        method: req.method,
        path: req.originalUrl,
        errorMessage: String(e),
        issues: (e as any)?.issues,
      });
      if ((e as any).issues) {
        res.status(400).json({ error: "Invalid request data", details: (e as any).issues });
      } else {
        res.status(500).json({ error: String(e) });
      }
    }
  });

  // Model-scoped management helpers
  app.get("/api/status/:model", async (req: Request, res: Response) => {
    try {
      const model = req.params.model;
      const name = getContainerNameForModel(model);
      const exists = await containerExists(name);
      if (!exists) return res.json(StatusResponseSchema.parse({ container: name, running: false, port: VLLM_PORT, image: VLLM_IMAGE }));
      const running = await containerRunning(name);
      const port = await getHostPort(name);
      const response = StatusResponseSchema.parse({ container: name, running, port, image: VLLM_IMAGE });
      res.json(response);
    } catch (e) {
      logger.error("route_error", {
        route: "/api/status/:model",
        method: req.method,
        path: req.originalUrl,
        model: req.params.model,
        errorMessage: String(e),
      });
      res.status(500).json({ error: String(e) });
    }
  });

  app.delete("/api/remove/:model", async (req: Request, res: Response) => {
    try {
      const model = req.params.model;
      const name = getContainerNameForModel(model);
      const exists = await containerExists(name);
      if (!exists) {
        const response = RemoveResponseSchema.parse({ ok: true, removed: false, message: "not found" });
        return res.json(response);
      }
      const running = await containerRunning(name);
      if (running) await runDocker(["stop", name]);
      stopLogStreaming(name);
      await runDocker(["rm", name]);
      const response = RemoveResponseSchema.parse({ ok: true, removed: true });
      res.json(response);
    } catch (e) {
      logger.error("route_error", {
        route: "/api/remove/:model",
        method: req.method,
        path: req.originalUrl,
        model: req.params.model,
        errorMessage: String(e),
      });
      res.status(500).json({ error: String(e) });
    }
  });

  app.post("/api/stop/:model", async (req: Request, res: Response) => {
    try {
      const model = req.params.model;
      const name = getContainerNameForModel(model);
      const exists = await containerExists(name);
      if (!exists) {
        const response = StopResponseSchema.parse({ ok: true, stopped: false, message: "not found" });
        return res.json(response);
      }
      const running = await containerRunning(name);
      if (!running) {
        const response = StopResponseSchema.parse({ ok: true, stopped: false, message: "already stopped" });
        return res.json(response);
      }
      await runDocker(["stop", name]);
      const response = StopResponseSchema.parse({ ok: true, stopped: true });
      res.json(response);
    } catch (e) {
      logger.error("route_error", {
        route: "/api/stop/:model",
        method: req.method,
        path: req.originalUrl,
        model: req.params.model,
        errorMessage: String(e),
      });
      res.status(500).json({ error: String(e) });
    }
  });

  app.get("/api/health", (_req: Request, res: Response) => {
    const response = HealthResponseSchema.parse({ ok: true });
    res.json(response);
  });
}
