# Science Arena Challenge - Example Submission

This is a demonstration project that implements four endpoints for scientific research tasks using LLM models.

## Features

- **Literature Review** (`/literature_review`): Conducts comprehensive literature reviews using standard LLM
- **Paper Q&A** (`/paper_qa`): Answers questions about papers using reasoning model with PDF upload
- **Ideation** (`/ideation`): Generates research ideas with optional semantic similarity analysis using embedding model
- **Paper Review** (`/paper_review`): Provides structured peer reviews for papers with PDF upload

All endpoints support streaming responses in Server-Sent Events (SSE) format.

## Models Used

- **deepseek-chat**: Standard LLM for literature review, ideation, and paper review
- **deepseek-reasoner**: Reasoning model for deep paper analysis
- **Qwen/Qwen3-Embedding-4B**: Embedding model for semantic similarity in ideation

## Configuration

The application is configured via `config.yml` in the root directory. This file allows you to adjust various parameters such as retry logic, concurrency limits, and model temperatures for different agents.

### Example `config.yml`

```yaml
# Global configuration for s1-review
llm:
  max_retries: 5
  initial_retry_delay: 1.0
  max_retry_delay: 60.0
  backoff_multiplier: 2.0
  default_temperature: 0.2
  default_max_tokens: 65536

retrieval:
  retry_delay_base: 2
  semantic_scholar_max_retries: 5
  arxiv_max_retries: 3
  max_results: 15

concurrency:
  cheating_detector: 6
  motivation_evaluator: 5

agents:
  motivation_evaluator:
    temperature: 0.3
  paper_memory_summarizer:
    temperature: 0.4
  paper_structurer_old:
    temperature: 0.2
  cheating_detector:
    temperature: 1.0
  baseline:
    temperature: 0.2
  paper_structurer:
    temperature: 0.0
  neurodong:
    temperature: 0.2
  summarizer:
    temperature: 1.0
```

## Setup

1. Copy the example environment file and configure your settings:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your API credentials:
   ```
   SCI_MODEL_BASE_URL=https://your-api-endpoint.com/v1
   SCI_MODEL_API_KEY=your-api-key-here
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up -d --build
   ```

4. The service will be available at `http://localhost:3000`

### Custom Port

To use a different host port, set the `HOST_PORT` environment variable:
```bash
HOST_PORT=8080 docker-compose up
```

## API Endpoints

### 1. Literature Review

**Endpoint:** `POST /literature_review`

**Request:**
```json
{
  "query": "What are the latest advances in transformer models?"
}
```

**Response:** Streaming SSE format
```
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"## Recent"}}]}
data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":" Advances"}}]}
...
data: [DONE]
```

### 2. Paper Q&A

**Endpoint:** `POST /paper_qa`

**Request:**
```json
{
  "query": "Please carefully analyze and explain the reinforcement learning training methods used in this article.",
  "pdf_content": "<base64_encoded_pdf_content>"
}
```

**Response:** Streaming SSE format with reasoning model output

### 3. Ideation

**Endpoint:** `POST /ideation`

**Request:**
```json
{
  "query": "Generate research ideas about climate change"
}
```

**Note:** This endpoint uses hardcoded reference ideas to demonstrate the embedding model's semantic similarity capabilities. The reference ideas include various AI/ML research topics, and the endpoint will:
1. Compute embeddings for the query and reference ideas using the embedding model
2. Rank reference ideas by cosine similarity to the query
3. Generate new ideas based on the similarity analysis

**Response:** Streaming SSE format with ideas informed by semantic similarity analysis

### 4. Paper Review

**Endpoint:** `POST /paper_review`

**Request:**
```json
{
  "query": "Please review this paper",
  "pdf_content": "<base64_encoded_pdf_content>"
}
```

**Response:** Streaming SSE format with structured review

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "ok"
}
```

## Error Handling

All endpoints return standard HTTP status codes with JSON response bodies for error scenarios:

- **400 Bad Request**: Missing required parameters
  ```json
  {
    "error": "Bad Request",
    "message": "Query is required"
  }
  ```

- **500 Internal Server Error**: Server-side errors
  ```json
  {
    "error": "Internal Server Error",
    "message": "Error description"
  }
  ```

**Note:** For streaming endpoints, errors that occur during streaming will interrupt the stream. All validation errors are caught before streaming begins and returned as HTTP status code + JSON format.

## Development

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

### Testing

You can test the endpoints using curl or any HTTP client:

```bash
# Literature Review
curl -X POST http://localhost:3000/literature_review \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformers in NLP?"}'

# Ideation (demonstrates embedding model for similarity)
curl -X POST http://localhost:3000/ideation \
  -H "Content-Type: application/json" \
  -d '{"query": "Generate AI research ideas for drug discovery"}'
```

## Technology Stack

- **FastAPI**: Modern async web framework
- **AsyncOpenAI**: Asynchronous OpenAI SDK for non-blocking API calls
- **PyPDF2**: PDF text extraction
- **NumPy**: Vector operations for semantic similarity
- **Uvicorn**: ASGI server
- **Docker**: Containerization

## Notes

- All endpoints use async/await for non-blocking operations
- PDF content is extracted from base64-encoded PDFs and used in full for analysis
- `/ideation` endpoint demonstrates the embedding model by using hardcoded reference ideas to compute semantic similarity with the user's query
- `/paper_qa` endpoint uses the reasoning model (deepseek-reasoner) for deep analysis
- `/paper_review` and `/literature_review` use the standard LLM model (deepseek-chat)
