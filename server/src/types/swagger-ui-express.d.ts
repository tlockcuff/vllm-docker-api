declare module 'swagger-ui-express' {
  import { RequestHandler } from 'express';
  export const serve: RequestHandler[];
  export function setup(document?: any, customOptions?: any, options?: any, customCss?: string, customfavIcon?: string, swaggerUrl?: string): RequestHandler;
}
