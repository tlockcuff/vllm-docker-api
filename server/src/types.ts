import { z } from 'zod';
import {
  StatusResponseSchema,
  StartRequestSchema,
  StartResponseSchema,
  StopResponseSchema,
  RemoveResponseSchema,
  HealthResponseSchema,
  ChatMessageSchema,
  ChatRequestSchema,
  OpenAIChatRequestSchema,
  OpenAICompletionsRequestSchema,
  OpenAIEmbeddingsRequestSchema,
  ModelsResponseSchema,
  CompletionsResponseSchema,
  EmbeddingsResponseSchema,
  DownloadModelRequestSchema,
} from './schemas.js';

export type StatusResponse = z.infer<typeof StatusResponseSchema>;
export type StartRequest = z.infer<typeof StartRequestSchema>;
export type StartResponse = z.infer<typeof StartResponseSchema>;
export type StopResponse = z.infer<typeof StopResponseSchema>;
export type RemoveResponse = z.infer<typeof RemoveResponseSchema>;
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type ChatRequest = z.infer<typeof ChatRequestSchema>;
export type OpenAIChatRequest = z.infer<typeof OpenAIChatRequestSchema>;
export type OpenAICompletionsRequest = z.infer<typeof OpenAICompletionsRequestSchema>;
export type OpenAIEmbeddingsRequest = z.infer<typeof OpenAIEmbeddingsRequestSchema>;

export type ModelsResponse = z.infer<typeof ModelsResponseSchema>;
export type CompletionsResponse = z.infer<typeof CompletionsResponseSchema>;
export type EmbeddingsResponse = z.infer<typeof EmbeddingsResponseSchema>;
export type DownloadModelRequest = z.infer<typeof DownloadModelRequestSchema>;
