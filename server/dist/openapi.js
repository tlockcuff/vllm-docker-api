import { z } from 'zod';
import { OpenAPIRegistry, OpenApiGeneratorV3, extendZodWithOpenApi } from '@asteasolutions/zod-to-openapi';
extendZodWithOpenApi(z);
export const registry = new OpenAPIRegistry();
export function generateOpenApiDocument() {
    const generator = new OpenApiGeneratorV3(registry.definitions);
    return generator.generateDocument({
        openapi: '3.0.3',
        info: { title: 'vLLM Docker API', version: '0.1.0' },
        servers: [{ url: 'http://localhost:3000' }],
    });
}
