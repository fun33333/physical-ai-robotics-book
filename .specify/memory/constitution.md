<!--
================================================================================
SYNC IMPACT REPORT
================================================================================
Version Change: 1.0.0 → 1.1.0 (MINOR - LangChain removal, OpenAI Agents SDK alignment)
Ratified: 2025-12-19 (initial)
Amended: 2025-12-22 (analysis remediation)

Modified Sections:
  - Backend Standards: LangChain → OpenAI Agents SDK (per spec.md:L205 mandate)
  - Data Flow Standards: Updated pipeline to Orchestrator pattern (RAG → Answer → Tone → Safety)
  - Code Quality: "LangChain chains" → "agent classes" for type safety
  - Document Processing: Embedding model clarified as Gemini embeddings
  - Principle V: Coverage target updated to "agent orchestration"

Rationale for Change:
  - /sp.analyze identified CRITICAL conflict C1: Constitution referenced LangChain
    but spec.md:L205 explicitly states "NO LangChain: Must use OpenAI Agents SDK exclusively"
  - This amendment aligns constitution with feature specification decisions

Templates Requiring Updates:
  ✅ plan-template.md - Already compatible (uses OpenAI Agents SDK)
  ✅ spec-template.md - Already compatible (prohibits LangChain)
  ✅ tasks-template.md - Already compatible (TDD structure intact)

Follow-up TODOs:
  - Verify all plan.md references use "OpenAI Agents SDK" terminology
  - Ensure no downstream templates reference LangChain

================================================================================
-->

# Physical AI & Humanoid Robotics Textbook with Integrated RAG Chatbot Constitution

## Core Principles

### I. Content-First RAG Architecture

The chatbot MUST retrieve knowledge exclusively from book chapters and user-selected text.
No external data sources or general LLM knowledge may be used for answers.

**Rules:**
- All responses MUST be traceable to indexed documents (zero hallucinations policy)
- Context precision for user-selected text binding MUST exceed 90%
- Qdrant vector search MUST be the sole retrieval mechanism
- Pre-retrieve top-k chunks from Qdrant before sending to Gemini

**Rationale:** Ensures educational accuracy and prevents misinformation in technical robotics content.

### II. Gemini Free API Efficiency

All LLM operations MUST use Google Gemini Free API exclusively. No premium APIs permitted.

**Rules:**
- Prompt optimization required for 15 requests/minute rate limit
- Implement caching in Postgres to avoid re-querying Gemini for repeated queries
- Fallback to Qdrant results directly if Gemini quota exceeded
- Log all API calls for quota tracking and monitoring
- Batch requests where possible to maximize efficiency

**Rationale:** Maintains zero-cost infrastructure while ensuring reliable service availability.

### III. Modular Integration

Book (Phase 1) and Chatbot (Phase 2) MUST operate independently but integrate seamlessly.

**Rules:**
- Phase 1 (Book) MUST be fully functional without chatbot components
- Phase 2 (Chatbot) MUST be deployable as a separate service
- React chatbot component MUST load asynchronously (no page load degradation)
- Chatbot JS bundle MUST be < 50KB gzipped
- Clear API boundaries between book frontend and chatbot backend

**Rationale:** Enables independent deployment, testing, and maintenance of each system component.

### IV. User Context Binding

Selected text from the book MUST always accompany user queries to the chatbot.

**Rules:**
- JavaScript highlight detection MUST capture user text selections
- Selected text MUST be passed as `selectedText` parameter in all chat requests
- API endpoint MUST accept `{ query, selectedText, context }` structure
- Context binding precision MUST exceed 90% accuracy
- Empty selections permitted but logged for analytics

**Rationale:** Provides relevant context for accurate, targeted responses to user questions.

### V. Test-First Development (NON-NEGOTIABLE)

TDD is mandatory. Tests MUST be written before implementation.

**Rules:**
- Red-Green-Refactor cycle strictly enforced
- Minimum 80% code coverage for FastAPI routes, agent orchestration, embedding logic
- Unit tests for all public functions
- Integration tests for: document indexing pipeline, Qdrant retrieval accuracy, Gemini API
- End-to-end tests for: book page → text selection → chatbot query → response
- Performance tests: query latency < 2 seconds (p99), Qdrant search < 200ms
- Tests MUST fail before implementation begins

**Rationale:** Ensures code quality, prevents regressions, and validates behavior before deployment.

### VI. Multi-Tone Responses

Deliver content in English, Roman Urdu, and mixed Bro-Guide style with Karachi flavor.

**Rules:**
- Default tone: Professional, conversational, Karachi-local accent
- Bro Mode: Friendly slang, Roman Urdu phrases, mixed language responses
  - Example: "Yaar, iska matlab ye hai—[technical explanation]—samajh gaya?"
- User levels: Beginner (ELI5), Intermediate (standard), Advanced (deep technical)
- Tone and level MUST be switchable at runtime (user preference)
- All three modes MUST be functional and tested before release

**Rationale:** Enhances accessibility and engagement for diverse Pakistani audience.

## Technical Standards

### Backend Standards

- **Framework**: FastAPI (Python 3.10+) for async API endpoints
- **Vector Database**: Qdrant Cloud Free Tier (no on-premise setup)
- **LLM Provider**: Google Gemini Free API only (no OpenAI)
- **Document Processing**: OpenAI Agents SDK for multi-agent orchestration pipeline (NO LangChain per spec.md:L205)
- **Data Persistence**: Neon Serverless Postgres for conversation history and user profiles
- **Authentication**: Better-Auth integration (Phase 2.5+)

### Frontend Standards

- **Book Framework**: Docusaurus v3.x with React component system
- **Chatbot UI**: React standalone modal/sidebar component embedded in book site
- **Text Selection**: JavaScript highlight detection → pass selectedText to chatbot context
- **Responsive Design**: Mobile-first, accessible (WCAG 2.1 AA)

### Data Flow Standards

- **Document Indexing**: Extract Docusaurus chapters → OpenAI Agents SDK processing → Qdrant embeddings
- **Query Processing**: User query + selectedText → Orchestrator pipeline (RAG → Answer → Tone → Safety) with Qdrant context retrieval
- **Response Generation**: Gemini API → Format response in selected tone → Stream to frontend
- **Conversation History**: Store in Postgres for context awareness in multi-turn conversations

### Testing Standards

- **Unit Tests**: Minimum 80% code coverage
- **Integration Tests**: Document indexing, Qdrant retrieval, Gemini API integration
- **End-to-End Tests**: Full user journey validation
- **Performance Tests**: Query < 2s (p99), Qdrant search < 200ms

## Constraints & Limits

### API & Quota Constraints

| Resource | Limit | Mitigation |
|----------|-------|------------|
| Gemini Free API | 15 req/min | Batching, caching, fallback responses |
| Qdrant Cloud Free | 20MB vectors | Efficient embedding compression |
| Neon Free Tier | 0.5GB storage | Conversation history retention policy |
| Rate Limiting | 15 req/min/IP | FastAPI middleware enforcement |

### Code Quality Standards

- **Linting**: Black formatter, isort imports, pylint (score > 8.0)
- **Type Safety**: 100% type hints on FastAPI endpoints and agent classes
- **Documentation**: Google-style docstrings on all public functions
- **Git Workflow**: Feature branches, PR reviews, conventional commits

### Document Processing Standards

- **Accepted Formats**: Markdown (Docusaurus source), HTML (generated), text blocks
- **Embedding Model**: Gemini embeddings via google-generativeai SDK
- **Chunk Strategy**: 512-token chunks with 50-token overlap
- **Metadata**: Store chapter, section, difficulty level with each embedding

### Personalization Standards

- **Tones**: Professional, Roman Urdu, Bro-Guide (Karachi flavor)
- **Levels**: Beginner (ELI5), Intermediate, Advanced
- **Switchability**: Runtime preference selection per user

## Deployment Standards

### Phase 1 (Book) Deployment

- **Hosting**: GitHub Pages or Vercel
- **Build**: `npm run build` → static site generation
- **Repository**: Public GitHub repo with CI/CD pipeline
- **Trigger**: Push to main → automatic build and deploy
- **Environment**: Node.js 18+, npm/yarn

### Phase 2 (RAG Chatbot) Deployment

**Backend:**
- **Hosting**: Docker container (AWS ECS, Render, Railway, or local)
- **Build**: `docker build -t rag-chatbot .` with multi-stage Dockerfile
- **Port**: FastAPI on port 8000 (configurable via env)
- **Dependencies**: requirements.txt with pinned versions

**Frontend:**
- **Build**: `npm run build` within Docusaurus project
- **Integration**: React component compiled into Docusaurus bundle

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini Free API key |
| `QDRANT_URL` | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | Qdrant authentication token |
| `DATABASE_URL` | Neon Postgres connection string |
| `BETTER_AUTH_SECRET` | Auth secret (Phase 2.5+) |
| `CORS_ORIGIN` | Docusaurus site URL for CORS |

### CI/CD Pipeline

- **Trigger**: PR creation and push to main
- **Linting**: Black, isort, pylint checks
- **Tests**: pytest with coverage report (minimum 80%)
- **Build**: Docker image creation and push to registry
- **Deploy**: Automated to staging; production requires manual approval

### Database Schema

| Table | Purpose |
|-------|---------|
| `conversations` | id, user_id, query, selected_text, response, created_at, tone, user_level |
| `embeddings_metadata` | id, chapter, section, difficulty_level, source_url, created_at |
| `users` | id, email, background_level, preferred_tone, language (Phase 2.5+) |

### API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/chat` | Accept query + selectedText → Return response + sources |
| POST | `/index` | Trigger document re-indexing |
| GET | `/health` | Deployment health check |
| GET | `/search` | Direct vector search (debugging) |

## Governance

### Amendment Process

1. Document proposed change with rationale
2. Team review and approval required
3. Update constitution version per semantic versioning
4. Create migration plan if breaking changes
5. Update all dependent templates and documentation

### Versioning Policy

- **MAJOR**: Breaking changes to principles or governance
- **MINOR**: New sections, expanded guidance, new constraints
- **PATCH**: Clarifications, typo fixes, non-semantic refinements

### Compliance Requirements

- All PRs MUST verify constitution compliance
- Complexity MUST be justified in Complexity Tracking section of plans
- Test coverage requirements are non-negotiable
- Security standards (input validation, SQL injection prevention) MUST be followed
- API key handling via environment variables only (never committed)

### Success Criteria

- Chatbot accurately answers questions based on book content
- User-selected text correctly bound to queries (context precision > 90%)
- Response latency under 2 seconds (p99)
- All three tone modes functional and tested
- Zero hallucinations: All answers traceable to indexed documents
- Seamless Docusaurus integration (no page load degradation)

### Security Standards

- API keys via environment variables only (never in repo)
- CORS restricted to Docusaurus domain
- Input validation and sanitization before Gemini calls
- SQL injection prevention via parameterized queries
- Rate limiting at FastAPI level (15 req/min/IP)

**Version**: 1.1.0 | **Ratified**: 2025-12-19 | **Last Amended**: 2025-12-22
