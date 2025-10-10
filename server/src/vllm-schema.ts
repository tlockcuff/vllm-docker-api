import { z } from 'zod';

// Common helpers
const humanReadableInt = z.string().or(z.number()); // accepts "1k", "25.6k", or numeric
const jsonLike = z.string().or(z.record(z.any()));  // accepts JSON string or object
const boolFlag = z.boolean();                       // for --flag / --no-flag style
const port = z.number().int().min(1).max(65535).or(z.string()); // CLI often uses string

// ModelConfig
export const ModelConfigSchema = z.object({
  runner: z.enum(["auto", "draft", "generate", "pooling"]).default("auto"),
  convert: z.enum(["auto", "classify", "embed", "none", "reward"]).default("auto"),
  task: z.enum(["auto", "classify", "draft", "embed", "embedding", "generate", "reward", "score", "transcription", "None"]).optional(),
  tokenizer: z.string().nullable().optional(),
  tokenizerMode: z.enum(["auto", "custom", "mistral", "slow"]).default("auto"),
  trustRemoteCode: boolFlag.default(false),
  dtype: z.enum(["auto", "bfloat16", "float", "float16", "float32", "half"]).default("auto"),
  seed: z.number().int().nullable().optional(),
  hfConfigPath: z.string().nullable().optional(),
  allowedLocalMediaPath: z.string().default(""),
  allowedMediaDomains: z.string().nullable().optional(),
  revision: z.string().nullable().optional(),
  codeRevision: z.string().nullable().optional(),
  ropeScaling: jsonLike.default({}),
  ropeTheta: z.number().nullable().optional(),
  tokenizerRevision: z.string().nullable().optional(),
  maxModelLen: humanReadableInt.nullable().optional(),
  quantization: z.string().nullable().optional(),
  enforceEager: boolFlag.default(false),
  maxLogprobs: z.number().int().default(20),
  logprobsMode: z.enum(["processed_logits", "processed_logprobs", "raw_logits", "raw_logprobs"]).default("raw_logprobs"),
  disableSlidingWindow: boolFlag.default(false),
  disableCascadeAttn: boolFlag.default(false),
  skipTokenizerInit: boolFlag.default(false),
  enablePromptEmbeds: boolFlag.default(false),
  servedModelName: z.string().nullable().optional(),
  configFormat: z.enum(["auto", "hf", "mistral"]).default("auto"),
  hfToken: z.union([z.string(), z.boolean()]).nullable().optional(), // True => use CLI login token
  hfOverrides: jsonLike.default({}),
  poolerConfig: jsonLike.nullable().optional(),
  overridePoolerConfig: jsonLike.nullable().optional(), // deprecated
  logitsProcessorPattern: z.string().nullable().optional(),
  generationConfig: z.union([z.literal("auto"), z.literal("vllm"), z.string()]).default("auto"),
  overrideGenerationConfig: jsonLike.default({}),
  enableSleepMode: boolFlag.default(false),
  modelImpl: z.enum(["auto", "terratorch", "transformers", "vllm"]).default("auto"),
  overrideAttentionDtype: z.string().nullable().optional(),
  logitsProcessors: z.union([z.array(z.string()), z.string()]).nullable().optional(),
  ioProcessorPlugin: z.string().nullable().optional(),
});

// LoadConfig
export const LoadConfigSchema = z.object({
  loadFormat: z.enum([
    "auto",
    "pt",
    "safetensors",
    "npcache",
    "dummy",
    "tensorizer",
    "runai_streamer",
    "bitsandbytes",
    "sharded_state",
    "gguf",
    "mistral",
  ]).default("auto"),
  downloadDir: z.string().nullable().optional(),
  safetensorsLoadStrategy: z.enum(["lazy", "eager", "torchao"]).default("lazy"),
  modelLoaderExtraConfig: jsonLike.default({}),
  ignorePatterns: z.array(z.string()).default(["original/**/*"]),
  useTqdmOnLoad: boolFlag.default(true),
  ptLoadMapLocation: z.union([z.literal("cpu"), z.literal("cuda"), jsonLike]).default("cpu"),
});

// StructuredOutputsConfig
export const StructuredOutputsConfigSchema = z.object({
  reasoningParser: z.enum([
    "deepseek_r1",
    "glm45",
    "openai_gptoss",
    "granite",
    "hunyuan_a13b",
    "mistral",
    "olmo3",
    "qwen3",
    "seed_oss",
    "step3",
    "",
  ]).default(""),
  guidedDecodingBackend: z.string().nullable().optional(), // deprecated
  guidedDecodingDisableFallback: z.boolean().nullable().optional(), // deprecated
  guidedDecodingDisableAnyWhitespace: z.boolean().nullable().optional(), // deprecated
  guidedDecodingDisableAdditionalProperties: z.boolean().nullable().optional(), // deprecated
});

// ParallelConfig
export const ParallelConfigSchema = z.object({
  distributedExecutorBackend: z.enum(["external_launcher", "mp", "ray", "uni"]).nullable().optional(),
  pipelineParallelSize: z.number().int().min(1).default(1),
  tensorParallelSize: z.number().int().min(1).default(1),
  decodeContextParallelSize: z.number().int().min(1).default(1),
  dataParallelSize: z.number().int().min(1).default(1),
  dataParallelRank: z.number().int().nullable().optional(),
  dataParallelStartRank: z.number().int().nullable().optional(),
  dataParallelSizeLocal: z.number().int().nullable().optional(),
  dataParallelAddress: z.string().nullable().optional(),
  dataParallelRpcPort: port.nullable().optional(),
  dataParallelBackend: z.enum(["mp", "ray"]).default("mp"),
  dataParallelHybridLb: boolFlag.default(false),
  enableExpertParallel: boolFlag.default(false),
  enableDbo: boolFlag.default(false),
  dboDecodeTokenThreshold: z.number().int().default(32),
  dboPrefillTokenThreshold: z.number().int().default(512),
  disableNcclForDpSynchronization: boolFlag.default(false),
  enableEplb: boolFlag.default(false),
  eplbConfig: jsonLike.default({
    window_size: 1000,
    step_interval: 3000,
    num_redundant_experts: 0,
    log_balancedness: false,
  }),
  expertPlacementStrategy: z.enum(["linear", "round_robin"]).default("linear"),
  numRedundantExperts: z.number().int().nullable().optional(), // deprecated
  eplbWindowSize: z.number().int().nullable().optional(),      // deprecated
  eplbStepInterval: z.number().int().nullable().optional(),    // deprecated
  eplbLogBalancedness: z.boolean().nullable().optional(),      // deprecated
  maxParallelLoadingWorkers: z.number().int().nullable().optional(),
  rayWorkersUseNsight: boolFlag.default(false),
  disableCustomAllReduce: boolFlag.default(false),
  workerCls: z.union([z.literal("auto"), z.string()]).default("auto"),
  workerExtensionCls: z.string().default(""),
  enableMultimodalEncoderDataParallel: boolFlag.default(false),
});

// CacheConfig
export const CacheConfigSchema = z.object({
  blockSize: z.union([z.literal(1), z.literal(8), z.literal(16), z.literal(32), z.literal(64), z.literal(128)]).nullable().optional(),
  gpuMemoryUtilization: z.number().min(0).max(1).default(0.9),
  kvCacheMemoryBytes: humanReadableInt.nullable().optional(),
  swapSpace: z.number().int().min(0).default(4),
  kvCacheDtype: z.enum(["auto", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"]).default("auto"),
  numGpuBlocksOverride: z.number().int().nullable().optional(),
  enablePrefixCaching: boolFlag.nullable().optional(), // default depends on V1
  prefixCachingHashAlgo: z.enum(["sha256", "sha256_cbor"]).default("sha256"),
  cpuOffloadGb: z.number().min(0).default(0),
  calculateKvScales: boolFlag.default(false),
  kvSharingFastPrefill: boolFlag.default(false),
  mambaCacheDtype: z.enum(["auto", "float32"]).default("auto"),
  mambaSsmCacheDtype: z.enum(["auto", "float32"]).default("auto"),
});

// MultiModalConfig
export const MultiModalConfigSchema = z.object({
  limitMmPerPrompt: jsonLike.default({}), // supports legacy and mixed formats
  mediaIoKwargs: jsonLike.default({}),
  mmProcessorKwargs: jsonLike.nullable().optional(),
  mmProcessorCacheGb: z.number().min(0).default(4),
  disableMmPreprocessorCache: boolFlag.default(false),
  mmProcessorCacheType: z.enum(["lru", "shm"]).default("lru"),
  mmShmCacheMaxObjectSizeMb: z.number().int().min(1).default(128),
  mmEncoderTpMode: z.enum(["data", "weights"]).default("weights"),
  interleaveMmStrings: boolFlag.default(false),
  skipMmProfiling: boolFlag.default(false),
  videoPruningRate: z.number().min(0).max(1).nullable().optional(),
});

// LoRAConfig
export const LoRAConfigSchema = z.object({
  enableLora: boolFlag.nullable().optional(),
  maxLoras: z.number().int().min(1).default(1),
  maxLoraRank: z.union([
    z.literal(1),
    z.literal(8),
    z.literal(16),
    z.literal(32),
    z.literal(64),
    z.literal(128),
    z.literal(256),
    z.literal(320),
    z.literal(512),
  ]).default(16),
  loraExtraVocabSize: z.union([z.literal(256), z.literal(512)]).default(256), // deprecated
  loraDtype: z.enum(["auto", "bfloat16", "float16"]).default("auto"),
  maxCpuLoras: z.number().int().nullable().optional(),
  fullyShardedLoras: boolFlag.default(false),
  defaultMmLoras: jsonLike.nullable().optional(),
});

// ObservabilityConfig
export const ObservabilityConfigSchema = z.object({
  showHiddenMetricsForVersion: z.string().nullable().optional(),
  otlpTracesEndpoint: z.string().url().nullable().optional(),
  collectDetailedTraces: z.enum([
    "all",
    "model",
    "worker",
    "None",
    "model,worker",
    "model,all",
    "worker,model",
    "worker,all",
    "all,model",
    "all,worker",
  ]).nullable().optional(),
});

// SchedulerConfig
export const SchedulerConfigSchema = z.object({
  maxNumBatchedTokens: humanReadableInt.nullable().optional(),
  maxNumSeqs: z.number().int().nullable().optional(),
  maxNumPartialPrefills: z.number().int().min(1).default(1),
  maxLongPartialPrefills: z.number().int().min(1).default(1),
  cudaGraphSizes: z.array(z.number().int().min(1)).default([]),
  longPrefillTokenThreshold: z.number().int().min(0).default(0),
  numLookaheadSlots: z.number().int().min(0).default(0),
  schedulingPolicy: z.enum(["fcfs", "priority"]).default("fcfs"),
  enableChunkedPrefill: boolFlag.nullable().optional(),
  disableChunkedMmInput: boolFlag.default(false),
  schedulerCls: z.string().default("vllm.core.scheduler.Scheduler"),
  disableHybridKvCacheManager: boolFlag.default(false),
  asyncScheduling: boolFlag.default(false),
});

// VllmConfig
export const VllmConfigSchema = z.object({
  speculativeConfig: jsonLike.nullable().optional(),
  kvTransferConfig: jsonLike.nullable().optional(),
  kvEventsConfig: jsonLike.nullable().optional(),
  compilationConfig: jsonLike.default({
    level: null,
    debug_dump_path: null,
    cache_dir: "",
    backend: "",
    custom_ops: [],
    splitting_ops: null,
    use_inductor: true,
    compile_sizes: null,
    inductor_compile_config: { enable_auto_functionalized_v2: false },
    inductor_passes: {},
    cudagraph_mode: null,
    use_cudagraph: true,
    cudagraph_num_of_warmups: 0,
    cudagraph_capture_sizes: null,
    cudagraph_copy_inputs: false,
    full_cuda_graph: false,
    use_inductor_graph_partition: false,
    pass_config: {},
    max_capture_size: null,
    local_cache_dir: null,
  }),
  additionalConfig: jsonLike.default({}),
  structuredOutputsConfig: jsonLike.default({
    backend: "auto",
    disable_fallback: false,
    disable_any_whitespace: false,
    disable_additional_properties: false,
    reasoning_parser: "",
  }),
});

// EngineArgs (top-level)
export const EngineArgsSchema = z.object({
  disableLogStats: boolFlag.default(false),
  modelConfig: ModelConfigSchema,
  loadConfig: LoadConfigSchema,
  structuredOutputsConfig: StructuredOutputsConfigSchema,
  parallelConfig: ParallelConfigSchema,
  cacheConfig: CacheConfigSchema,
  multiModalConfig: MultiModalConfigSchema,
  loraConfig: LoRAConfigSchema,
  observabilityConfig: ObservabilityConfigSchema,
  schedulerConfig: SchedulerConfigSchema,
  vllmConfig: VllmConfigSchema,
});

// AsyncEngineArgs
export const AsyncEngineArgsSchema = z.object({
  enableLogRequests: boolFlag.default(false),
  disableLogRequests: boolFlag.default(true), // deprecated
});