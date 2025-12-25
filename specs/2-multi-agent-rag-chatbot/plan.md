# Implementation Plan: Multi-Agent RAG Chatbot for Textbook

**Branch**: `2-multi-agent-rag-chatbot` | **Date**: 2025-12-19 | **Last Amended**: 2025-12-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/2-multi-agent-rag-chatbot/spec.md`

---

## Changelog (Analysis Remediation)

| Date | Issue | Change | Rationale |
|------|-------|--------|-----------|
| 2025-12-22 | C1 | Constitution updated to use OpenAI Agents SDK | Plan already correct; constitution was stale |
| 2025-12-22 | H1/M5 | Safety Guardian moved to Phase 3 in tasks.md | FR-003 compliance; plan references updated |

**Source**: `/sp.analyze` run on 2025-12-22. Plan was already aligned with spec; changes applied to constitution.md and tasks.md.

## Summary

Build a multi-agent RAG chatbot system that answers questions about the Physical AI & Humanoid Robotics textbook using OpenAI Agents SDK and Gemini Free API. The system employs an orchestrator pattern with four specialized sub-agents (RAG Agent for retrieval, Answer/Tutor Agent for generation, Tone Agent for formatting, Safety Guardian for validation) operating in a pipeline architecture. The chatbot integrates seamlessly into Docusaurus website with selected-text context binding, multi-tone responses (English, Roman Urdu, Bro-Guide), multi-turn conversation history, and automatic API key rotation under free-tier constraints.

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**:
- OpenAI Agents SDK (multi-agent orchestration)
- FastAPI (async API server)
- google-generativeai (Gemini Free API)
- qdrant-client (vector DB client)
- sqlalchemy + alembic (database ORM + migrations)
- pydantic (type validation)
- black, pylint, pytest (code quality + testing)

**Storage**:
- Qdrant Cloud Free Tier (20MB vector storage)
- Neon Serverless Postgres (0.5GB storage, 30-day conversation retention)

**Testing**: pytest (unit, integration, E2E, agent behavior, hallucination detection)

**Target Platform**: Linux/macOS dev environment, Docker containerized backend, GitHub Pages + React for frontend

**Project Type**: Web application (backend: FastAPI + agents, frontend: React modal/sidebar)

**Performance Goals**:
- Chat response latency < 2 seconds (p99) including orchestrator pipeline
- Qdrant vector search < 200ms
- Agent handoff < 50ms
- Tone formatting < 100ms
- Safety validation < 500ms

**Constraints**:
- Gemini Free API: 15 requests/minute (auto-rotation across multiple keys)
- Qdrant: 20MB vector storage (semantic chunks only, no full texts)
- Neon: 0.5GB storage (30-day conversation history retention)
- React bundle: < 50KB gzipped (async loading)
- Type safety: 100% type hints on FastAPI endpoints
- Test coverage: Minimum 80%
- Code quality: Black formatted, pylint > 8.0

**Scale/Scope**:
- 4 specialized agents in orchestrator pipeline
- ~40 textbook chapters to index
- 512-token chunks with 50-token overlap
- Support for 100 concurrent user sessions
- Multi-turn conversations with 30-day history

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principles Alignment

| Principle | Requirement | Status | Notes |
|-----------|-------------|--------|-------|
| Content-First RAG Architecture | Retrieve exclusively from book chapters, zero hallucinations | ✅ PASS | Orchestrator pipeline with RAG Agent + Safety Guardian ensures all responses traceable to indexed documents |
| Gemini Free API Efficiency | Use Gemini Free API only, handle 15 req/min limit | ✅ PASS | Multiple API key rotation, caching in Postgres, fallback strategy documented |
| Modular Integration | Book and Chatbot operate independently | ✅ PASS | Phase 1 book (003) complete; Phase 2 chatbot deployable as separate service |
| User Context Binding | Selected text accompanies queries | ✅ PASS | JavaScript text selection detected, passed as selectedText parameter, > 90% context precision target |
| Test-First Development | TDD mandatory, 80% coverage, all public functions tested | ✅ PASS | Agent behavior tests, hallucination detection tests, integration tests planned |
| Multi-Tone Responses | English, Roman Urdu, Bro-Guide supported | ✅ PASS | Tone Agent specialized for formatting; all three modes functional and tested |

### Gate Evaluation

| Gate | Requirement | Status | Notes |
|------|-------------|--------|-------|
| Feature Specification | Requirements clearly defined, user stories prioritized | ✅ PASS | 3 user stories (P1, P2, P3), 30 functional requirements, 28 non-functional requirements |
| Architecture Design | Technical stack determined, no ambiguities | ✅ PASS | Orchestrator pattern with 4 specialized agents, all tech choices clarified in `/sp.clarify` |
| Constitution Alignment | No principle violations | ✅ PASS | All 6 core principles aligned; no complexity justification needed |
| Dependencies Available | All external services accessible | ✅ PASS | Qdrant Cloud Free, Neon Postgres Free, Gemini Free API all available |
| Base Code Available | Existing codebase to refactor from | ✅ PASS | Reference: https://github.com/Saba-Gul/LLMOps-Deploying-RaG-Application-With-Qdrant-Langchain-OpenAI |

**Constitution Check Result**: ✅ PASS - No violations. Proceed to Phase 0 research.

## Project Structure

### Documentation (this feature)

```text
specs/2-multi-agent-rag-chatbot/
├── spec.md              # Feature specification (DONE)
├── plan.md              # This file (implementation plan)
├── research.md          # Phase 0: Research findings
├── data-model.md        # Phase 1: Entity definitions, relationships
├── quickstart.md        # Phase 1: Development setup guide
├── contracts/           # Phase 1: API contracts
│   ├── chat-endpoint.yaml
│   ├── index-endpoint.yaml
│   ├── health-endpoint.yaml
│   └── search-endpoint.yaml
└── bonus-features.md    # Bonus features + judge metrics
```

### Source Code Structure

```text
backend/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── coordinator.py           # Main Coordinator Agent orchestration
│   │   ├── rag_agent.py             # RAG Agent (retrieval from Qdrant)
│   │   ├── answer_tutor_agent.py    # Answer/Tutor Agent (generation via Gemini)
│   │   ├── tone_agent.py            # Tone Agent (formatting English/Urdu/Bro-Guide)
│   │   └── safety_guardian.py       # Safety Guardian (validation, hallucination detection)
│   ├── models/
│   │   ├── conversation.py          # Conversation entity (user_id, query, response, tone, history)
│   │   ├── embeddings_metadata.py   # Embeddings metadata (chapter, section, difficulty)
│   │   ├── api_key_quota.py         # API key quota tracking (requests, status, rotation)
│   │   └── database.py              # Database initialization and session management
│   ├── services/
│   │   ├── qdrant_service.py        # Vector store operations (index, search, chunk)
│   │   ├── gemini_service.py        # Gemini API client with key rotation + caching
│   │   ├── orchestration_service.py # Orchestrator pipeline (RAG → Answer → Tone → Safety)
│   │   └── conversation_service.py  # Conversation history persistence
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                # FastAPI endpoints: /chat, /index, /health, /search
│   │   └── middleware.py            # CORS, rate limiting, input sanitization
│   ├── config.py                    # Environment variables, configuration
│   ├── main.py                      # FastAPI app initialization
│   └── utils.py                     # Helper functions, logging, error handling
├── tests/
│   ├── unit/
│   │   ├── test_rag_agent.py
│   │   ├── test_answer_tutor_agent.py
│   │   ├── test_tone_agent.py
│   │   ├── test_safety_guardian.py
│   │   ├── test_gemini_service.py
│   │   └── test_orchestration_service.py
│   ├── integration/
│   │   ├── test_orchestrator_pipeline.py
│   │   ├── test_qdrant_retrieval.py
│   │   ├── test_conversation_history.py
│   │   └── test_api_key_rotation.py
│   ├── e2e/
│   │   ├── test_full_chat_flow.py
│   │   ├── test_hallucination_detection.py
│   │   └── test_multi_turn_context.py
│   └── conftest.py                  # pytest fixtures
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Multi-stage Docker build
├── docker-compose.yml               # Local dev environment
└── README.md                        # Backend setup and API docs

frontend/
├── src/
│   ├── components/
│   │   ├── ChatBot.tsx              # Main chatbot modal/sidebar component
│   │   ├── TextSelector.tsx         # Text selection detection hook
│   │   ├── ConversationHistory.tsx  # Chat message display
│   │   ├── ToneSelector.tsx         # Tone selection dropdown
│   │   └── LoadingSpinner.tsx       # Loading state UI
│   ├── services/
│   │   ├── chatService.ts           # /chat endpoint client
│   │   ├── apiClient.ts             # HTTP client with error handling
│   │   └── textSelectionService.ts  # JavaScript highlight detection
│   ├── types/
│   │   ├── chat.ts                  # TypeScript interfaces for chat entities
│   │   └── agent.ts                 # Agent response types
│   ├── hooks/
│   │   ├── useTextSelection.ts      # Text selection hook
│   │   ├── useChat.ts               # Chat state management
│   │   └── useTone.ts               # Tone preference hook
│   └── App.tsx                      # Entry point
├── package.json                     # NPM dependencies
├── tsconfig.json                    # TypeScript config
├── README.md                        # Frontend setup guide
└── dist/                            # Build output (< 50KB gzipped)
```

**Structure Decision**: Web application with separated backend (FastAPI + Agents + Postgres) and frontend (React modal component embedded in Docusaurus). Backend deployed as Docker container; frontend bundled into Docusaurus static build.

## Data Flow Architecture

```
1. USER INTERACTION (Docusaurus Site)
   ├─ Student opens chapter
   ├─ Selects text (JavaScript highlight detection)
   └─ Types question into chatbot modal

2. FRONTEND → BACKEND (HTTP POST /chat)
   └─ Request: { query, selectedText, context, tone }

3. MAIN COORDINATOR AGENT ORCHESTRATION
   ├─ Receive user query + selectedText
   ├─ Initialize pipeline context
   └─ Route through specialized agents

4. PIPELINE STAGE 1: RAG AGENT (RETRIEVAL)
   ├─ Input: query + selectedText
   ├─ Combine selectedText + query for context
   ├─ Search Qdrant with semantic similarity
   ├─ Retrieve top-k chunks with metadata
   └─ Output: [{ chunk, chapter, section, difficulty }, ...]

5. PIPELINE STAGE 2: ANSWER/TUTOR AGENT (GENERATION)
   ├─ Input: query + retrieved_chunks + conversation_history
   ├─ Build Gemini prompt with context
   ├─ Send to Gemini API (with key rotation if quota exceeded)
   ├─ Generate accurate response citing sources
   └─ Output: { response_text, sources: [chapter, section, ...] }

6. PIPELINE STAGE 3: TONE AGENT (FORMATTING)
   ├─ Input: response_text + user_tone_preference
   ├─ Apply tone transformation (English/Roman Urdu/Bro-Guide)
   ├─ Implement conciseness logic (brief + "Ask for details?")
   ├─ Preserve technical accuracy across all tones
   └─ Output: { formatted_response, tone_applied }

7. PIPELINE STAGE 4: SAFETY GUARDIAN (VALIDATION)
   ├─ Input: formatted_response + query + retrieved_chunks
   ├─ Check for hallucinations (unsourced claims)
   ├─ If uncertain: Ask clarifying question → user reponds → re-answer
   ├─ If hallucination detected: Rewrite or flag with transparency note
   └─ Output: { validated_response, validation_status, safety_notes }

8. RESPONSE STORAGE (POSTGRES)
   ├─ Save to conversations table
   ├─ Record: user_id, query, selectedText, response, agent_used, tone, timestamp
   └─ Maintain history for multi-turn context

9. RETURN TO FRONTEND
   └─ Response: { response, sources, agent_used, tone, validation_status, latency }

10. FRONTEND DISPLAY
    └─ Render response with sources and "Ask for longer explanation?" prompt
```

## API Design

### POST /chat
```
Request:
{
  "query": "What is ROS 2?",
  "selectedText": "ROS 2 is a middleware platform",
  "context": "I was reading about robotics fundamentals",
  "tone": "english" | "roman_urdu" | "bro_guide",
  "conversationId": "uuid" (optional)
}

Response:
{
  "response": "ROS 2 is a middleware platform...",
  "sources": [
    { "chapter": "Module 1", "section": "ros2-fundamentals", "confidence": 0.95 },
    { "chapter": "Module 1", "section": "ros2-architecture", "confidence": 0.87 }
  ],
  "agentUsed": "Answer/Tutor Agent",
  "tone": "english",
  "concise": true,
  "validation": {
    "status": "approved",
    "hallucinations_checked": true
  },
  "latency": 1240,
  "latencyBreakdown": {
    "retrieval": 200,
    "generation": 950,
    "formatting": 50,
    "validation": 40
  }
}
```

### POST /index
```
Request: {}

Response:
{
  "status": "indexing_complete",
  "indexed_chunks": 4823,
  "storage_used_mb": 18.7,
  "chapters_processed": 40,
  "timestamp": "2025-12-19T14:30:00Z"
}
```

### GET /health
```
Response:
{
  "status": "healthy",
  "qdrant": "operational",
  "gemini_keys": ["active", "active", "exhausted"],
  "postgres": "connected",
  "api_rotations_today": 42,
  "last_rotation": "2025-12-19T14:30:00Z"
}
```

### GET /search?q=SLAM&top_k=5
```
Response:
{
  "query": "SLAM",
  "results": [
    {
      "chunk_id": "uuid",
      "text": "Simultaneous Localization and Mapping...",
      "chapter": "Module 1",
      "section": "ros2-navigation",
      "similarity_score": 0.92
    },
    ...
  ]
}
```

## Complexity Tracking

| Aspect | Justification | Alternatives Rejected |
|--------|---------------|----------------------|
| 4 specialized agents in orchestrator pattern | Orchestrator ensures single pipeline, not parallel agents competing; each agent has specific tool scope (retrieval, generation, formatting, validation) → accuracy guaranteed | Parallel agents doing same work would cause redundancy and accuracy issues; single monolithic agent can't handle specialized concerns |
| Multiple Gemini API keys with auto-rotation | Free tier limited to 15 req/min; rotation extends effective limit and ensures resilience | Could use premium API (violates constitution); could fail after quota (violates availability) |
| 30-day conversation retention with auto-delete | Respects free tier storage limit (0.5GB) while providing sufficient context window for student learning | Longer retention risks exceeding quota; shorter would hurt learning continuity |
| Background indexing on startup with query queueing | Allows system to accept queries immediately while indexing progresses; queued queries process once ready → best UX | Blocking indexing would delay availability; failing queries would frustrate users |
| Safety Guardian with clarification flow | Asks clarifying question first to resolve uncertainty before rewriting → improves accuracy; transparency note maintains trust | Silently rewriting is deceptive; outright rejection frustrates users |

## Non-Functional Requirements by Agent

| Requirement | RAG Agent | Answer/Tutor Agent | Tone Agent | Safety Guardian |
|-------------|-----------|-------------------|------------|-----------------|
| Latency (p99) | < 200ms | < 1000ms | < 100ms | < 500ms |
| Error Handling | Return empty chunks if Qdrant fails | Rotate API keys; fallback to previous response | Skip tone if unsure | Flag for manual review |
| Type Safety | 100% type hints | 100% type hints | 100% type hints | 100% type hints |
| Test Coverage | 80%+ unit + integration | 80%+ unit + E2E | 80%+ unit | 80%+ behavior + hallucination |
| Logging | Log retrieval queries, top-k scores | Log prompt, API key used, generation time | Log tone applied, conciseness logic | Log all hallucinations, clarifications |

## Phase 0: Research Deliverables

**research.md will address:**
- OpenAI Agents SDK best practices for orchestrator patterns
- Gemini Free API quota management patterns (multi-key rotation)
- LangChain → OpenAI Agents SDK migration considerations
- Qdrant semantic search optimization for 20MB storage
- Neon Postgres Free Tier performance characteristics
- React async component loading for chatbot modal
- Docusaurus integration patterns for external React components
- Text selection detection in JavaScript
- Multi-tone response generation (English → Roman Urdu → Bro-Guide)
- Hallucination detection in LLM responses

## Phase 1: Design Deliverables

**data-model.md will define:**
- Conversation entity (id, user_id, query, selected_text, response, agent_used, tone, created_at)
- API Key Quota entity (api_key_id, requests_today, last_reset, status)
- Embeddings Metadata entity (chunk_id, chapter, section, difficulty_level, token_count, source_url)
- User entity (id, email, background_level, preferred_tone) - for future Better-Auth bonus feature
- Relationships (Conversations ← User, Embeddings → Chapters, API_Keys → Rotation_Log)
- State transitions (active → exhausted → reset for API keys)
- Validation rules (selected_text max 5000 chars, response max 10000 chars, tone ∈ {english, roman_urdu, bro_guide})

**contracts/chat-endpoint.yaml** will specify:
- Request/response schemas (JSON Schema)
- HTTP status codes (200, 400, 429, 500, 503)
- Rate limiting headers (X-RateLimit-Limit, X-RateLimit-Remaining)
- Timeout specification (30 seconds)
- Error response format

**quickstart.md** will cover:
- Local development setup (Python venv, Docker Compose)
- Environment variables (.env.example with GEMINI_API_KEY_1, GEMINI_API_KEY_2, QDRANT_URL, DATABASE_URL)
- Running tests (pytest unit/integration/E2E)
- Building Docker image
- Deploying to staging
- Connecting to Docusaurus site
- Common troubleshooting

## Next Steps

1. ✅ **Constitution Check**: PASSED - Proceed to Phase 0
2. **Phase 0: Research** → Generate research.md with findings on OpenAI Agents SDK, Gemini API patterns, storage optimization
3. **Phase 1: Design** → Generate data-model.md, contracts/, quickstart.md
4. **Phase 1: Agent Context Update** → Run `.specify/scripts/bash/update-agent-context.sh` to update CLAUDE.md
5. **Phase 2: Task Generation** → Run `/sp.tasks` to create actionable tasks.md from this plan

---

## References

- **Feature Spec**: `/specs/2-multi-agent-rag-chatbot/spec.md`
- **Constitution**: `/.specify/memory/constitution.md` (v1.0.0)
- **Base Code**: https://github.com/Saba-Gul/LLMOps-Deploying-RaG-Application-With-Qdrant-Langchain-OpenAI
- **Book Content** (Phase 1): `/specs/003-complete-book-content/` (deployed to Docusaurus)
