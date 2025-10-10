## vLLM Docker API

A clean TypeScript Node.js API to manage vLLM containers via Docker CLI with type-safe Zod validation. No duplication, just solid validation and types.

### Setup & Usage

#### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd vllm-docker-api

# Configure environment variables (optional)
# Create a .env file with any variables you want to override
# See "Environment Variables" section below for available options

# Start the API and vLLM services
docker-compose up -d

# Or for GPU support (requires NVIDIA Docker)
docker-compose --profile gpu up -d
```

#### Manual Setup (Development)

```bash
# Install dependencies
npm install

# For development with auto-reload
npm run dev

# For production
npm run build
npm start
```

### API Features

**Type-Safe & Validated:**
- ✅ **Zod schema validation** for all requests/responses
- ✅ **Runtime type checking** with detailed error messages
- ✅ **TypeScript-first** design with full type safety

### API Documentation

🌐 **[Interactive API Documentation](./docs/index.html)** - Browsable webpage with detailed schemas, examples, and endpoint specifications.

📖 **[API Reference](./docs/api.md)** - Markdown version of the API documentation.

📋 **[OpenAPI Specification](./docs/openapi.json)** - Import into Postman, Insomnia, or other API testing tools.

#### Using with Postman
1. Open Postman
2. Click "Import" → "File"
3. Select `docs/openapi.json`
4. Choose your workspace and click "Import"
5. Set the base URL to `http://localhost:3000`

### API Endpoints

**OpenAI-Compatible Endpoints:**
- `GET /v1/models` - List models from project directory and vLLM-loaded models
- `POST /v1/chat/completions` - Chat completions with query parameter model support
- `POST /v1/completions` - Text completions with query parameter model support
- `POST /v1/embeddings` - Generate embeddings with query parameter model support

**Management Endpoints:**
- `POST /api/start` - Start vLLM container with optional `{ model }` body
- `POST /api/stop` - Stop vLLM container
- `DELETE /api/remove` - Remove vLLM container
- `GET /api/status` - Get container status
- `POST /api/chat` - Proxy to OpenAI-compatible chat completions (legacy)
- `POST /api/models/download` - Download a model from Hugging Face to local storage
- `GET /api/models/download/progress` - Get download progress for all or specific models
- `DELETE /api/models/download/:model` - Cancel a model download
- `GET /api/health` - Health check

### Docker Architecture

The application consists of two services:

1. **API Service** - Node.js application with Hugging Face CLI for model downloads
2. **vLLM Service** - The actual vLLM inference server (optional, can run separately)

**Volumes:**
- `models/` - Persists downloaded models between container restarts
- `.cache/` - Caches Hugging Face downloads for faster subsequent access

**Networks:**
- Both services communicate via an isolated Docker network