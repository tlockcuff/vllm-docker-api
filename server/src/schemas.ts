import { z } from "zod";
import {
  AsyncEngineArgsSchema,
  CacheConfigSchema,
  EngineArgsSchema,
  LoadConfigSchema,
  LoRAConfigSchema,
  ModelConfigSchema,
  MultiModalConfigSchema,
  ObservabilityConfigSchema,
  ParallelConfigSchema,
  SchedulerConfigSchema,
  StructuredOutputsConfigSchema,
  VllmConfigSchema,
} from "./vllm-schema";

// API Schemas
export const StatusResponseSchema = z.object({
  container: z.string().describe("Container name"),
  running: z.boolean().describe("Whether container is running"),
  port: z.number().describe("Container port"),
  image: z.string().describe("Docker image name"),
});

export const StartRequestSchema = z.object({
  model: z.string().optional().describe("Model to load (optional, uses default if not provided)"),
  ...ModelConfigSchema.shape,
  ...LoadConfigSchema.shape,
  ...StructuredOutputsConfigSchema.shape,
  ...ParallelConfigSchema.shape,
  ...CacheConfigSchema.shape,
  ...MultiModalConfigSchema.shape,
  ...LoRAConfigSchema.shape,
  ...ObservabilityConfigSchema.shape,
  ...SchedulerConfigSchema.shape,
  ...VllmConfigSchema.shape,
  ...EngineArgsSchema.shape,
  ...AsyncEngineArgsSchema.shape,
});

export const StartResponseSchema = z.object({
  ok: z.boolean(),
  model: z.string().describe("Model that was loaded"),
});

export const StopResponseSchema = z.object({
  ok: z.boolean(),
  stopped: z.boolean().describe("Whether container was stopped"),
  message: z.string().optional().describe("Additional message (if not found or already stopped)"),
});

export const RemoveResponseSchema = z.object({
  ok: z.boolean(),
  removed: z.boolean().describe("Whether container was removed"),
  message: z.string().optional().describe("Additional message (if not found)"),
});

export const HealthResponseSchema = z.object({ ok: z.boolean() });

export const ChatMessageSchema = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string(),
});

export const ChatRequestSchema = z.object({
  messages: z.array(ChatMessageSchema),
  model: z.string().optional().describe("Model name (optional, uses loaded model)"),
  temperature: z.number().min(0).max(2).default(0.7),
  stream: z.boolean().default(false).describe("Whether to stream the response"),
});

export const OpenAIChatRequestSchema = z.object({
  messages: z.array(ChatMessageSchema),
  model: z.string().optional().describe("Model name (optional, uses loaded model)"),
  temperature: z.number().min(0).max(2).optional().default(0.7),
  stream: z.boolean().optional().default(false).describe("Whether to stream the response"),
  max_tokens: z.number().optional(),
  top_p: z.number().min(0).max(1).optional(),
  frequency_penalty: z.number().optional(),
  presence_penalty: z.number().optional(),
});

export const OpenAICompletionsRequestSchema = z.object({
  model: z.string().optional().describe("Model name (optional, uses loaded model)"),
  prompt: z.union([z.string(), z.array(z.string())]).optional(),
  temperature: z.number().min(0).max(2).optional().default(0.7),
  max_tokens: z.number().optional(),
  stream: z.boolean().optional().default(false),
  top_p: z.number().min(0).max(1).optional(),
  frequency_penalty: z.number().optional(),
  presence_penalty: z.number().optional(),
});

export const OpenAIEmbeddingsRequestSchema = z.object({
  model: z.string().describe("Model name"),
  input: z.union([z.string(), z.array(z.string())]).describe("Input text(s) to embed"),
  user: z.string().optional(),
});

export const ModelsResponseSchema = z.object({
  object: z.string(),
  data: z.array(
    z.object({
      id: z.string(),
      object: z.string(),
      created: z.number(),
      owned_by: z.string(),
    })
  ),
});

export const CompletionsResponseSchema = z.object({
  id: z.string(),
  object: z.string(),
  created: z.number(),
  model: z.string(),
  choices: z.array(
    z.object({
      index: z.number(),
      text: z.string(),
      finish_reason: z.string().nullable(),
    })
  ),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number(),
      total_tokens: z.number(),
    })
    .optional(),
});

export const ChatChoiceSchema = z.object({
  index: z.number(),
  message: z.object({
    role: z.string(),
    content: z.string(),
  }),
  finish_reason: z.string().nullable(),
});

export const ChatResponseSchema = z.object({
  id: z.string(),
  object: z.string(),
  created: z.number(),
  model: z.string(),
  choices: z.array(ChatChoiceSchema),
  usage: z
    .object({
      prompt_tokens: z.number(),
      completion_tokens: z.number(),
      total_tokens: z.number(),
    })
    .optional(),
});

export const EmbeddingsResponseSchema = z.object({
  object: z.string(),
  data: z.array(
    z.object({
      object: z.string(),
      embedding: z.array(z.number()),
      index: z.number(),
    })
  ),
  model: z.string(),
  usage: z.object({
    prompt_tokens: z.number(),
    total_tokens: z.number(),
  }),
});

export const DownloadModelRequestSchema = z.object({
  model: z.string().describe('Hugging Face model ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3")'),
});
