import express, { json } from 'express';
import type { Express, Request, Response, NextFunction } from 'express';
import swaggerUi from 'swagger-ui-express';
import { mountManagementRoutes } from './routes/management.js';
import { mountDownloadRoutes } from './routes/downloads.js';
import { mountOpenAIRoutes } from './routes/openai.js';
import { generateOpenApiDocument } from './openapi.js';
import { logger } from './logger.js';

export function createApp(): Express {
  const app = express();
  app.use(json({ limit: '2mb' }));

  // Request/Response logging
  app.use((req: Request, res: Response, next: NextFunction) => {
    const start = Date.now();
    const { method, originalUrl } = req;
    res.on('finish', () => {
      const durationMs = Date.now() - start;
      logger.info('http_request', {
        method,
        path: originalUrl,
        status: res.statusCode,
        durationMs,
        ip: req.ip,
      });
    });
    next();
  });

  // Routes
  mountManagementRoutes(app);
  mountDownloadRoutes(app);
  mountOpenAIRoutes(app);

  // OpenAPI generation and Swagger UI
  const openapi = generateOpenApiDocument();
  app.get('/docs/openapi.json', (_req: Request, res: Response) => res.json(openapi));
  app.use('/docs', swaggerUi.serve, swaggerUi.setup(openapi));

  // Error handler
  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    logger.error('http_error', {
      errorMessage: err?.message || String(err),
      issues: err?.issues,
      stack: err?.stack,
    });
    if (err && err.issues) {
      return res.status(400).json({ error: 'Invalid request data', details: err.issues });
    }
    res.status(500).json({ error: 'Internal server error' });
  });

  return app;
}


