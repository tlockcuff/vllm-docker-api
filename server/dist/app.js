import express, { json } from 'express';
import swaggerUi from 'swagger-ui-express';
import { mountManagementRoutes } from './routes/management.js';
import { mountDownloadRoutes } from './routes/downloads.js';
import { mountOpenAIRoutes } from './routes/openai.js';
import { generateOpenApiDocument } from './openapi.js';
export function createApp() {
    const app = express();
    app.use(json({ limit: '2mb' }));
    // Routes
    mountManagementRoutes(app);
    mountDownloadRoutes(app);
    mountOpenAIRoutes(app);
    // OpenAPI generation and Swagger UI
    const openapi = generateOpenApiDocument();
    app.get('/docs/openapi.json', (_req, res) => res.json(openapi));
    app.use('/docs', swaggerUi.serve, swaggerUi.setup(openapi));
    // Error handler
    app.use((err, _req, res, _next) => {
        if (err && err.issues) {
            return res.status(400).json({ error: 'Invalid request data', details: err.issues });
        }
        res.status(500).json({ error: 'Internal server error' });
    });
    return app;
}
