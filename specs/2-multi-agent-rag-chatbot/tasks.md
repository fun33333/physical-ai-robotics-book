# Tasks: Multi-Agent RAG Chatbot for Textbook

**Input**: Design documents from `/specs/2-multi-agent-rag-chatbot/`
**Prerequisites**: plan.md (COMPLETE), spec.md (COMPLETE), research.md (recommended before Phase 2)

**Tests**: Included - TDD mandatory per constitution (Test-First Development Principle V). All tests written BEFORE implementation, must FAIL initially.

**Organization**: Tasks grouped by user story to enable independent implementation, testing, and deployment of each story as an MVP increment.

---

## Changelog (Analysis Remediation - 2025-12-22)

| Issue ID | Severity | Change Made | Rationale |
|----------|----------|-------------|-----------|
| C2 | CRITICAL | Added `⛔ TEST GATE` checkpoints after each phase | T044 was incomplete but implementation marked done - violated TDD |
| H1/M5 | HIGH | Moved Safety Guardian (T080) to Phase 3, added blocking prereqs | FR-003: Safety Guardian MUST NOT be bypassed; was in Phase 7 |
| H2 | HIGH | Added T094a for query queueing unit test | FR-024a: Query queueing during indexing had no dedicated test |
| M3 | MEDIUM | Added coverage gate (80%) validation after each phase | NFR-012: 80% coverage requirement was only checked in Phase 12 |
| L1 | LOW | Removed all timeline estimates from Implementation Strategy | Planning guidelines: focus on what, not when |

**Source**: `/sp.analyze` run on 2025-12-22 identified these issues before implementation.

---

## Format: `[ID] [P?] [Story?] Description`

- **[ID]**: Task identifier (T001, T002, etc.)
- **[P]**: Parallelizable (can run in parallel, different files, no dependencies)
- **[Story]**: User story mapping (US1, US2, US3) - REQUIRED for user story phases
- **Include exact file paths** in all descriptions

## Path Conventions

- **Backend**: `backend/src/` for source, `backend/tests/` for tests
- **Frontend**: `frontend/src/` for React components
- **Specs**: `specs/2-multi-agent-rag-chatbot/` for documentation

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize project structure, dependencies, and shared configuration

- [X] T001 Create backend project structure: `mkdir -p backend/src/agents backend/src/models backend/src/services backend/src/api backend/tests/{unit,integration,e2e}`
- [X] T002 Create frontend project structure: `mkdir -p frontend/src/{components,services,types,hooks}`
- [X] T003 [P] Initialize backend requirements.txt with: FastAPI, uvicorn, openai, google-generativeai, qdrant-client, sqlalchemy, psycopg2, pydantic, python-dotenv, black, pylint, pytest, pytest-cov at `backend/requirements.txt`
- [X] T004 [P] Create `.env.example` at backend root with: GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3, QDRANT_URL, QDRANT_API_KEY, DATABASE_URL, CORS_ORIGIN
- [X] T005 [P] Initialize backend/pyproject.toml with Black, Pylint, and pytest configuration
- [X] T006 [P] Create Dockerfile at backend root (multi-stage: Python 3.10+ base, requirements install, app run on port 8000)
- [X] T007 [P] Create docker-compose.yml at backend root with: FastAPI service (port 8000), Postgres service (port 5432 for local dev testing - optional, uses Neon in production)
- [X] T008 Create `backend/src/config.py` with environment variable loading, validation, and configuration classes
- [X] T009 [P] Create `backend/src/main.py` with FastAPI app initialization, CORS middleware, rate limiting middleware setup (15 req/min)
- [X] T010 [P] Create `backend/src/utils.py` with structured logging (JSON format), error handling utilities, helpers for latency tracking

**Checkpoint**: Project structure initialized, environment configured, ready for foundational infrastructure

⛔ **TEST GATE (Phase 1)**: `pytest backend/tests/ -v --cov=backend/src` - Must pass before Phase 2

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure MUST be complete before any user story implementation begins

**CRITICAL**: No user story work can begin until this phase completes.

### Database & ORM Setup

- [X] T011 Create database models in `backend/src/models/database.py`: SQLAlchemy session management, Base declarative class, database initialization function
- [X] T012 [P] Create Conversation model in `backend/src/models/conversation.py`: id (UUID), user_id (str), query (text), selected_text (text, nullable), response (text), agent_used (str), tone (str), sources (JSON), created_at, updated_at; include __repr__ and relationship definitions
- [X] T013 [P] Create EmbeddingsMetadata model in `backend/src/models/embeddings_metadata.py`: id (UUID), chunk_id (str), chapter (str), section (str), difficulty_level (str), token_count (int), source_url (str), created_at; include indexes on chapter + section
- [X] T014 [P] Create APIKeyQuota model in `backend/src/models/api_key_quota.py`: id (UUID), api_key_id (str), requests_today (int), last_reset (timestamp), status (str: active/exhausted/error), last_rotated_at; include quota tracking logic

### Service Layer - Qdrant Integration

- [X] T015 Create `backend/src/services/qdrant_service.py` with methods: initialize_collection(collection_name), index_document(chunk_id, text, metadata), search_similar(query, selectedText, top_k), get_chunk_by_id(chunk_id), delete_collection(); include error handling for connection failures
- [X] T016 [P] Create semantic chunking function in qdrant_service.py: split_document(text) → 512-token chunks with 50-token overlap, add chapter/section/difficulty metadata

### Service Layer - Gemini API & Key Rotation

- [X] T017 Create `backend/src/services/gemini_service.py` with: list of API keys from env, request method that rotates keys on quota exceeded, caching layer using Postgres, fallback response when all keys exhausted ("Your today's limit of AI Guide is exceeded. Please try again tomorrow.")
- [X] T018 [P] Create quota tracking in gemini_service.py: track requests per key, reset at 00:00 UTC, log all rotations with timestamp, return key status in health check
- [X] T019 [P] Add prompt caching in gemini_service.py: check if query+context hash exists in Postgres before calling Gemini, store successful responses for reuse

### Service Layer - Conversation History

- [X] T020 Create `backend/src/services/conversation_service.py` with methods: save_conversation(user_id, query, response, tone, sources), get_conversation_history(user_id, limit=50), delete_old_conversations(older_than_days=30), get_recent_context(user_id, session_id)
- [X] T021 [P] Add session isolation in conversation_service.py: maintain separate history per session, prevent cross-session context leakage with session_id checks

### Service Layer - Orchestration

- [X] T022 Create `backend/src/services/orchestration_service.py` with: coordinator() method that orchestrates pipeline, pipeline_context class to pass data between agents, error handling for agent failures, timeout handling (< 2s total)
- [X] T023 [P] Add latency tracking in orchestration_service.py: measure each stage (RAG, Answer, Tone, Safety), log breakdown per response, target < 2s p99

### Agents - Base Classes & Initialization

- [X] T024 Create `backend/src/agents/__init__.py` with imports of all agent classes
- [X] T025 [P] Create base agent class in `backend/src/agents/coordinator.py`: MainCoordinatorAgent with initialize_pipeline(), route_through_pipeline(query, selectedText, tone) → orchestrates handoffs

### FastAPI Routes Skeleton

- [X] T026 Create `backend/src/api/routes.py` with endpoint stubs: POST /chat, POST /index, GET /health, GET /search; all return placeholder responses
- [X] T027 [P] Create `backend/src/api/middleware.py` with: CORS middleware (Docosaurus domain only), rate limiting middleware (15 req/min/IP), input sanitization middleware (truncate large inputs, validate JSON)

### Database Migrations

- [X] T028 Create Alembic migration for initial schema: `alembic init backend/alembic`, create initial revision with Conversation, EmbeddingsMetadata, APIKeyQuota tables
- [X] T029 [P] Create migration helper in backend root: script to reset DB, run migrations, seed test data for local dev

**Checkpoint**: All infrastructure ready, agents framework initialized, services layer operational, database schema deployed. No user story code yet - foundational layer complete and testable independently.

⛔ **TEST GATE (Phase 2)**: `pytest backend/tests/ -v --cov=backend/src` - Must achieve 80%+ coverage on services layer before Phase 3

---

## Phase 3: User Story 1 - Textbook Q&A with Multi-Agent System (Priority: P1) 🎯 MVP

**BLOCKING PREREQUISITE**: Safety Guardian (T080-T084) MUST be implemented in this phase per FR-003: "System MUST NOT bypass Safety Guardian"

**Goal**: Enable students to ask questions about textbook content and receive accurate, sourced answers via orchestrator pipeline

**Independent Test**: Student can ask question about ROS 2, receive response sourced from textbook chapter 2, with > 90% accuracy and < 2 second latency

### Tests for User Story 1 (WRITTEN FIRST - TDD)

- [X] T030 [P] [US1] Write unit test for RAG Agent in `backend/tests/unit/test_rag_agent.py`: Mock Qdrant, verify retrieval returns top-k chunks with metadata, verify selectedText prioritized
- [X] T031 [P] [US1] Write unit test for Answer/Tutor Agent in `backend/tests/unit/test_answer_tutor_agent.py`: Mock Gemini response, verify response cites sources, verify no hallucinations (response contains only phrases from retrieved chunks)
- [X] T032 [P] [US1] Write unit test for Orchestration Service in `backend/tests/unit/test_orchestration_service.py`: Mock all agents, verify pipeline executes in order (RAG → Answer → Tone → Safety), verify context passes between stages
- [X] T033 [P] [US1] Write unit test for API key rotation in `backend/tests/unit/test_gemini_service.py`: Mock quota exceeded, verify rotation to next key, verify fallback message when all exhausted
- [X] T034 [P] [US1] Write integration test for full chat flow in `backend/tests/integration/test_orchestrator_pipeline.py`: Real Qdrant mock, real Gemini mock, verify end-to-end pipeline, verify latency < 2s, verify sources included
- [X] T035 [P] [US1] Write integration test for hallucination detection in `backend/tests/integration/test_safety_guardian.py`: Inject false claim into Answer Agent response, verify Safety Guardian catches it, verify rewrite/flag with transparency note
- [X] T036 [US1] Write E2E test for /chat endpoint in `backend/tests/e2e/test_full_chat_flow.py`: Real request with selectedText, verify response includes sources, verify tone parameter honored, verify < 2s latency

### Implementation for User Story 1

- [X] T037 [US1] Implement RAG Agent in `backend/src/agents/rag_agent.py`: class RAGAgent with retrieve(query, selectedText) method, Qdrant client integration, top-k retrieval (k=5), metadata handling
- [X] T038 [P] [US1] Implement Answer/Tutor Agent in `backend/src/agents/answer_tutor_agent.py`: class AnswerTutorAgent with generate(query, retrieved_chunks, conversation_history) method, Gemini integration, source citation, no hallucination guarantees
- [X] T039 [P] [US1] Implement Orchestration Service pipeline in `backend/src/services/orchestration_service.py`: process_chat_us1(query, selectedText) method, sequential agent handoff, error recovery, latency tracking
- [X] T040 [US1] Implement POST /chat endpoint in `backend/src/api/routes.py`: Accept { query, selectedText, tone }, call orchestration_service, return { response, sources, agent_used, latency_breakdown }
- [X] T041 [US1] Implement POST /index endpoint in `backend/src/api/routes.py`: Accept empty body or file path, trigger Qdrant indexing, return { status, indexed_chunks, storage_used_mb }
- [X] T042 [US1] Implement GET /health endpoint in `backend/src/api/routes.py`: Check Qdrant connectivity, Gemini key status, Postgres connection, API rotations today, return { status, all_checks }
- [X] T043 [US1] Add conversation history logging in conversation_service.py: save each chat to Postgres after Safety Guardian approval, include all metadata
- [X] T044 [US1] Run pytest on all US1 tests: Verify all pass, confirm 80%+ coverage for RAG + Answer agents, confirm all test assertions validate requirements

### Safety Guardian (Moved from Phase 7 - FR-003 Compliance)

**Rationale**: FR-003 states "System MUST NOT bypass Safety Guardian" - this MUST be in Phase 3 before US1 release

- [X] T044a [P] [US1] Write unit test for Safety Guardian in `backend/tests/unit/test_safety_guardian.py`: Mock response with unsourced claim, verify detect_hallucination() catches it, verify rewrite or clarification question triggered
- [X] T044b [P] [US1] Write hallucination scenario tests in `backend/tests/unit/test_hallucination_scenarios.py`: Test 5+ hallucination scenarios (false claim, extrapolation, external knowledge), verify all caught
- [X] T044c [US1] Implement Safety Guardian Agent in `backend/src/agents/safety_guardian.py`: class SafetyGuardian with validate(response, query, retrieved_chunks) method, hallucination detection, clarification flow
- [X] T044d [P] [US1] Add fact-checking logic in safety_guardian.py: For each claim in response, verify sourced in retrieved_chunks, flag unsourced claims
- [X] T044e [US1] Integrate Safety Guardian into orchestration pipeline: Add as final stage after Tone Agent, all responses must pass validation

**Checkpoint**: User Story 1 complete - students can ask textbook questions and receive accurate, sourced answers via orchestrator. System ready for MVP validation.

⛔ **TEST GATE (Phase 3/US1)**: `pytest backend/tests/ -v --cov=backend/src --cov-fail-under=80` - MUST pass with 80%+ coverage before proceeding

---

## Phase 4: User Story 2 - Multi-Tone Communication (Priority: P2)

**Goal**: Allow students to receive responses in preferred communication style (English, Roman Urdu, Bro-Guide) while maintaining technical accuracy

**Independent Test**: Student selects "roman_urdu" tone, receives response in friendly Karachi-local style with Urdu phrases, same technical content as English response

### Tests for User Story 2 (WRITTEN FIRST - TDD)

- [X] T045 [P] [US2] Write unit test for Tone Agent in `backend/tests/unit/test_tone_agent.py`: Mock response, test transformation to each of 3 tones (english, roman_urdu, bro_guide), verify technical accuracy preserved across tones
- [X] T046 [P] [US2] Write unit test for conciseness logic in `backend/tests/unit/test_tone_agent_concise.py`: Mock long response, verify truncation to 1-2 sentences with "Ask for longer?" prompt, verify full response returned on explicit request
- [X] T047 [P] [US2] Write integration test for tone switching mid-conversation in `backend/tests/integration/test_tone_switching.py`: Ask Q1 in english, switch to roman_urdu for Q2, verify tone changes but context maintained
- [X] T048 [US2] Write E2E test for tone parameter in POST /chat in `backend/tests/e2e/test_tone_parameter.py`: Send tone=roman_urdu, verify response uses Urdu phrases and colloquial language, verify same accuracy as english tone

### Implementation for User Story 2

- [X] T049 [US2] Implement Tone Agent in `backend/src/agents/tone_agent.py`: class ToneAgent with apply_tone(response, tone_preference) method, 3 tone transformers (english, roman_urdu, bro_guide), conciseness logic for brief responses
- [X] T050 [P] [US2] Create tone templates in `backend/src/agents/tone_prompts.py`: System prompts for each tone (English: formal/educational, Roman Urdu: friendly + Urdu phrases, Bro-Guide: Karachi slang/colloquial), apply to Gemini prompt
- [X] T051 [P] [US2] Implement conciseness logic in tone_agent.py: Truncate response > 250 chars to brief version + "Ask for longer explanation?" prompt, store full response for retrieval on follow-up
- [X] T052 [P] [US2] Add tone parameter to /chat endpoint: Accept tone ∈ {english, roman_urdu, bro_guide}, validate, pass to Tone Agent, include tone in response JSON
- [X] T053 [US2] Test all 3 tones with same question: Verify all 3 return different phrasings but same technical content, verify accuracy maintained, verify Urdu phrases realistic
- [X] T054 [US2] Run pytest on all US2 tests: Verify all pass, confirm 80%+ coverage for Tone Agent, confirm tone switching works correctly

**Checkpoint**: User Story 2 complete - students can select preferred tone. System now supports multi-tone communication for accessibility.

⛔ **TEST GATE (Phase 4/US2)**: `pytest backend/tests/unit/test_tone*.py -v --cov=backend/src/agents/tone_agent.py --cov-fail-under=80`

---

## Phase 5: User Story 3 - Multi-Turn Conversations with History (Priority: P3)

**Goal**: Enable multi-turn conversations where agents reference prior exchanges for deeper learning

**Independent Test**: Student asks Q1 about ROS 2, asks follow-up Q2 ("explain more about publishers"), agents reference prior exchange in Q2 response

### Tests for User Story 3 (WRITTEN FIRST - TDD)

- [X] T055 [P] [US3] Write unit test for conversation history retrieval in `backend/tests/unit/test_conversation_history.py`: Mock Postgres, verify get_recent_context(user_id, session_id) returns last 10 exchanges, verify 30-day retention cutoff
- [X] T056 [P] [US3] Write unit test for Answer Agent context awareness in `backend/tests/unit/test_answer_tutor_agent_context.py`: Mock conversation history, verify generate() method includes prior Q&A in Gemini prompt, verify follow-up references prior exchange
- [X] T057 [P] [US3] Write integration test for multi-turn flow in `backend/tests/integration/test_multi_turn_context.py`: Send Q1, save response, send Q2 with session_id, verify Q2 response references Q1 context, verify latency < 2s for Q2
- [X] T058 [US3] Write integration test for session isolation in `backend/tests/integration/test_session_isolation.py`: Create 2 concurrent sessions, verify each has independent history, verify no context leakage between sessions
- [X] T059 [US3] Write E2E test for conversation history endpoint in `backend/tests/e2e/test_conversation_history.py`: POST /chat Q1, POST /chat Q2 with conversationId, verify Q2 references Q1

### Implementation for User Story 3

- [X] T060 [US3] Modify POST /chat endpoint to accept optional conversationId: If provided, retrieve conversation history; if not, create new session (UUID)
- [X] T061 [P] [US3] Enhance Answer/Tutor Agent to include conversation context: Update generate() to prepend last 5 exchanges to Gemini prompt (with compression), maintain awareness across turns
- [X] T062 [P] [US3] Implement conversation history querying in conversation_service.py: get_conversation_history(user_id, conversation_id, limit=5) with 30-day retention, session isolation checks
- [X] T063 [P] [US3] Add conversationId to response JSON: Return conversationId for client to use in follow-up requests, include conversation_count in response
- [X] T064 [US3] Test multi-turn conversation with 5+ exchanges: Verify each response references prior context appropriately, verify accuracy maintained across turns, verify latency < 2s per turn
- [X] T065 [US3] Run pytest on all US3 tests: Verify all pass, confirm 80%+ coverage for conversation history, confirm multi-turn context works correctly

**Checkpoint**: User Story 3 complete - students can have multi-turn conversations with context awareness. Full chatbot functionality operational.

⛔ **TEST GATE (Phase 5/US3)**: `pytest backend/tests/ -v --cov=backend/src/services/conversation_service.py --cov-fail-under=80`

---

## Phase 6: Frontend React Component

**Goal**: Integrate chatbot as React modal/sidebar into Docusaurus site with text selection support

### Tests for Frontend (WRITTEN FIRST - TDD)

- [X] T066 [P] Write unit test for text selection hook in `frontend/tests/useTextSelection.test.tsx`: Mock DOM selection, verify getSelectedText() returns highlighted text
- [X] T067 [P] Write unit test for chat service in `frontend/tests/chatService.test.ts`: Mock POST /chat, verify sends { query, selectedText, tone }, verify parses response JSON
- [X] T068 [P] Write integration test for ChatBot component in `frontend/tests/ChatBot.integration.test.tsx`: Mock API, render component, type question, verify send button submits, verify response displays

### Implementation for Frontend

- [X] T069 [P] Create useTextSelection hook in `frontend/src/hooks/useTextSelection.ts`: Detect user text highlights, return selected text, handle multi-selection edge cases
- [X] T070 [P] Create chat service client in `frontend/src/services/chatService.ts`: HTTP client for POST /chat, error handling, retry logic on 429 (quota exceeded)
- [X] T071 [P] Implement ChatBot component in `frontend/src/components/ChatBot.tsx`: Modal/sidebar UI, question input, response display, tone selector, loading state, error handling
- [X] T072 [P] Create ConversationHistory component in `frontend/src/components/ConversationHistory.tsx`: Display prior Q&A, scroll to latest, timestamp display
- [X] T073 [P] Create ToneSelector component in `frontend/src/components/ToneSelector.tsx`: Dropdown for english/roman_urdu/bro_guide, default to english
- [ ] T074 Integrate ChatBot into Docusaurus: Add React component to docs layout, ensure async loading (non-blocking), verify < 50KB gzipped bundle
- [ ] T075 Add text selection event listener: Capture text highlights in Docosaurus chapters, pass selectedText to chatbot
- [X] T076 Run npm build: Verify bundle size < 50KB gzipped, verify TypeScript compilation, verify no console errors

**Checkpoint**: Frontend integrated, chatbot component live in Docosaurus, text selection working.

---

## Phase 7: Safety Guardian & Hallucination Detection (ADVANCED FEATURES)

> **NOTE**: Core Safety Guardian implementation (T044a-T044e) moved to Phase 3 per FR-003 compliance.
> This phase contains ADVANCED features: clarification flow, response rewriting, transparency notes.

**Goal**: Implement advanced validation features beyond basic hallucination detection

### Tests for Phase 7 (WRITTEN FIRST - TDD)

- [ ] T077 [P] Write clarification flow test in `backend/tests/unit/test_clarification_flow.py`: Simulate Safety Guardian asking clarifying question, simulate user response, verify clarification resolves uncertainty
- [ ] T078 [P] Write response rewriting test in `backend/tests/unit/test_response_rewriting.py`: Inject hallucination, verify rewrite preserves verified content only, verify transparency note added
- [ ] T079 [P] Write transparency note test in `backend/tests/unit/test_transparency_notes.py`: Verify note format "Note: I corrected an inaccuracy in my initial response." appears when correction made

### Implementation for Phase 7 (Advanced Features)

- [ ] T080 [P] Implement response rewriting in safety_guardian.py: If hallucination detected, rewrite response with verified information only, add transparency note
- [ ] T081 [P] Add clarification question generation in safety_guardian.py: If uncertain, generate clarifying question (e.g., "Are you asking about X or Y?"), return to user
- [ ] T082 Implement clarification response handling: Accept user clarification, re-process query with new context, return accurate response
- [ ] T083 Add Safety Guardian metrics: Track hallucinations_detected, clarifications_sent, rewrites_performed for observability (NFR-024)

**Checkpoint**: Advanced Safety Guardian features operational, clarification flow tested.

⛔ **TEST GATE (Phase 7)**: `pytest backend/tests/unit/test_safety*.py backend/tests/unit/test_clarification*.py -v --cov-fail-under=80`

---

## Phase 8: API Key Rotation & Fallback Strategy

**Goal**: Handle Gemini Free API quota limits with multi-key rotation and graceful degradation

### Tests for Phase 8

- [X] T085 [P] Write unit test for API key rotation in `backend/tests/unit/test_api_key_rotation.py`: Mock quota exhaustion, verify auto-rotation to next key, verify requests continue uninterrupted
- [X] T086 [P] Write integration test for fallback message in `backend/tests/integration/test_fallback_message.py`: Exhaust all keys, verify exact fallback message returned: "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
- [X] T087 [P] Write quota reset test in `backend/tests/unit/test_quota_reset.py`: Mock daily quota reset at 00:00 UTC, verify requests resume on new day

### Implementation for Phase 8

- [X] T088 [P] Enhance gemini_service.py with quota management: Track requests per key per day, auto-rotate on quota hit, log all rotations with timestamp and key ID
- [X] T089 [P] Implement /health endpoint quota reporting in api/routes.py: Return { gemini_keys: [active, active, exhausted], api_rotations_today: 42, last_rotation: timestamp }
- [X] T090 [P] Test quota exhaustion scenario: Simulate 15 requests (Gemini Free limit), verify Key1 exhausted, auto-rotate to Key2, Key3, final request returns fallback message
- [X] T091 Add quota metrics to structured logging: Log all API key rotations, track daily exhaustion patterns, alert if pattern suggests need for more keys

**Checkpoint**: API key rotation operational, quota limits respected, fallback strategy tested.

---

## Phase 9: Document Indexing Pipeline

**Goal**: Index all 40 textbook chapters into Qdrant, implement re-indexing without disruption

### Tests for Phase 9

- [ ] T092 [P] Write unit test for document chunking in `backend/tests/unit/test_document_chunking.py`: Mock document, verify 512-token chunks with 50-token overlap, verify metadata attached
- [ ] T093 [P] Write integration test for indexing pipeline in `backend/tests/integration/test_indexing_pipeline.py`: Mock 40 chapters, index all, verify Qdrant contains all chunks, verify < 20MB storage used
- [ ] T094 Write background indexing test in `backend/tests/integration/test_background_indexing.py`: Trigger indexing, immediately send queries, verify "Indexing in progress..." returned, queries queued and processed once ready
- [ ] T094a [P] Write unit test for query queueing in `backend/tests/unit/test_query_queueing.py`: Mock indexing state, verify queries queued when indexing=True, verify queue processes when indexing=False, verify no data loss (FR-024a coverage)

### Implementation for Phase 9

- [ ] T095 Implement background indexing in main.py: On startup, trigger Qdrant indexing in background thread, don't block app initialization
- [ ] T096 [P] Implement query queueing in orchestration_service.py: While indexing, queue chat requests, process when Qdrant ready, return "Indexing in progress..." message
- [ ] T097 [P] Create indexing script at backend root: `python scripts/index_textbook.py` that loads all chapters from Docosaurus, chunks, embeds, stores in Qdrant
- [ ] T098 [P] Implement re-indexing endpoint POST /index: Accept optional force=true, trigger re-indexing without disrupting active conversations, return { status, indexed_chunks, duration }
- [ ] T099 Test full indexing: Load all 40 chapters, verify < 20MB storage, verify search returns relevant chunks, verify latency < 200ms

**Checkpoint**: All textbook chapters indexed, background indexing operational.

⛔ **TEST GATE (Phase 9)**: `pytest backend/tests/unit/test_document*.py backend/tests/integration/test_indexing*.py -v --cov-fail-under=80`

---

## Phase 10: Conversation History Cleanup & Retention

**Goal**: Implement automatic cleanup of conversations older than 30 days to respect Neon free tier limit

### Tests for Phase 10

- [ ] T100 [P] Write unit test for retention policy in `backend/tests/unit/test_conversation_retention.py`: Mock conversations older than 30 days, verify delete_old_conversations() removes them, verify recent ones kept
- [ ] T101 [P] Write integration test for daily cleanup in `backend/tests/integration/test_daily_cleanup.py`: Mock system time, simulate 30-day retention boundary, verify cleanup task runs at scheduled time

### Implementation for Phase 10

- [ ] T102 [P] Implement cleanup task in conversation_service.py: Scheduled job at 00:00 UTC daily, deletes conversations older than 30 days, logs cleanup summary
- [ ] T103 [P] Add cleanup scheduling to main.py: On startup, register daily cleanup task using APScheduler or similar
- [ ] T104 [P] Monitor storage usage: Track current Postgres usage, alert if approaching 0.5GB limit, implement optional compression of old conversations to archive
- [ ] T105 Test retention policy: Simulate 35 days of conversations, verify cleanup runs, verify only last 30 days retained, verify 0.5GB limit respected

**Checkpoint**: Conversation cleanup operational, storage limits respected.

---

## Phase 11: Docosaurus Integration & Text Selection

**Goal**: Embed chatbot into published Docosaurus site, enable text selection from chapters

### Tests for Phase 11

- [ ] T106 Write integration test for Docosaurus embedding in `frontend/tests/docosaurus-integration.test.tsx`: Render Docosaurus page with ChatBot component, verify component loads asynchronously, verify no page load delay
- [ ] T107 Write text selection test in `frontend/tests/text-selection.test.tsx`: Render Docosaurus chapter, select text, trigger chat, verify selectedText passed to API

### Implementation for Phase 11

- [ ] T108 Add ChatBot component to Docosaurus theme: Modify layout to include ChatBot sidebar/modal on all pages, ensure async loading
- [ ] T109 [P] Implement text selection detector in Docosaurus context: Hook into chapter text, capture highlights, make available to ChatBot component
- [ ] T110 [P] Test integration: Open published Docosaurus site, select text from chapter, open chatbot, ask question, verify selectedText bound to query
- [ ] T111 Verify page load performance: Measure page load time, verify < 50ms impact from chatbot async loading

**Checkpoint**: Chatbot live in Docosaurus, text selection working, ready for student access.

---

## Phase 12: Testing, Performance Optimization & Polish

**Purpose**: Comprehensive testing, performance tuning, code quality validation

- [ ] T112 [P] Run full test suite: `pytest backend/tests/ -v --cov=backend/src --cov-report=html`, verify 80%+ coverage, 0 test failures
- [ ] T113 [P] Run code quality checks: `black backend/src backend/tests`, `pylint backend/src` (> 8.0), verify no issues
- [ ] T114 [P] Performance testing: Load test with 100 concurrent requests, verify < 2s p99 latency, verify Qdrant < 200ms, verify agent handoffs < 50ms
- [ ] T115 [P] Test hallucination detection end-to-end: Inject 10 hallucination scenarios, verify all caught by Safety Guardian, verify 0 hallucinations escape to user
- [ ] T116 [P] Test multi-tone responses: Ask same question in all 3 tones, manually verify tone consistency and technical accuracy
- [ ] T117 [P] Test API key rotation end-to-end: Make 16+ requests (exceed 15/min quota), verify Key1 exhausts, auto-rotate to Key2, verify fallback when all exhausted
- [ ] T118 [P] Test conversation history: Complete 5-turn conversation, verify each response references prior context, verify 30-day retention cutoff
- [ ] T119 [P] Test edge cases: Very long input, special characters, ambiguous queries, out-of-scope questions, concurrent sessions - verify all handled gracefully
- [ ] T120 Create comprehensive README at backend root: Installation, setup, running locally, running tests, Docker deployment, troubleshooting
- [ ] T121 [P] Create quickstart guide at `specs/2-multi-agent-rag-chatbot/quickstart.md`: Development environment setup, running backend + frontend locally, connecting to Docosaurus
- [ ] T122 [P] Create API documentation: Auto-generate from FastAPI OpenAPI (http://localhost:8000/docs), verify all endpoints documented with examples
- [ ] T123 [P] Verify Docker build: `docker build -t rag-chatbot . && docker run -p 8000:8000 rag-chatbot`, verify container starts, health check passes
- [ ] T124 [P] Code cleanup: Remove debug prints, add docstrings to all public functions (Google style), ensure 100% type hints on all FastAPI endpoints
- [ ] T125 Final integration test: Full end-to-end flow from textbook highlight → chat query → response with sources, verify all success criteria met

**Checkpoint**: All tests passing, 80%+ coverage, code quality > 8.0, performance targets met, ready for submission.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately ✅
- **Phase 2 (Foundational)**: Depends on Phase 1 COMPLETE - BLOCKS all user stories ⛔
- **Phases 3-5 (User Stories)**: All depend on Phase 2 completion ⛔
  - US1 and US2 can run in parallel after foundational (both P1 priority domain)
  - US3 can start after Phase 2 or alongside US1/US2
- **Phase 6 (Frontend)**: Can start after Phase 2 (independent), but benefits from API endpoints ready (Phase 3+)
- **Phase 7 (Safety Guardian)**: MUST complete before end of Phase 3 (US1 blocks without validation)
- **Phase 8 (API Key Rotation)**: Can start during Phase 2, needed before Phase 3 for quota resilience
- **Phase 9 (Indexing)**: Can start during Phase 2, MUST complete before Phase 3 (no data to retrieve without indexing)
- **Phase 10 (Retention)**: Can start during Phase 2, priority low (cleanup job)
- **Phase 11 (Docosaurus Integration)**: Depends on Phase 6 frontend completion
- **Phase 12 (Testing & Polish)**: Depends on all other phases - final validation pass

### User Story Dependencies

- **US1 (P1)**: Foundation - no dependencies on other stories ✅
- **US2 (P2)**: Can start after Phase 2; no hard dependency on US1 but benefits from API operational ✅
- **US3 (P3)**: Can start after Phase 2; no hard dependency on US1/US2 but benefits from conversation schema ready ✅

### Within Each User Story

1. Tests written FIRST (fail initially)
2. Models created (if needed)
3. Services implemented (agents, orchestration)
4. Endpoints/UI implemented
5. Tests run to verify
6. Integration testing
7. Story marked complete

### Parallel Opportunities

**Phase 1 Parallelization:**
- T003-T007: Create requirements.txt, .env.example, pyproject.toml, Dockerfile, docker-compose.yml - all independent, run in parallel
- T009-T010: Create main.py and utils.py - independent, run in parallel

**Phase 2 Parallelization:**
- T012-T014: Create models (Conversation, EmbeddingsMetadata, APIKeyQuota) - independent, run in parallel
- T015, T017, T020, T022: Create services - independent, run in parallel
- T030-T035: Write all US1 tests - independent, run in parallel

**Phase 3 (US1) Parallelization:**
- T030-T036: Write ALL tests first (can run in parallel while waiting for implementation)
- T037-T039: Implement RAG + Answer + Orchestration agents - independent once services ready, can run in parallel
- Once agents ready: T040-T043 can run in parallel (different endpoints/logic)

**Phase 4 (US2) Parallelization:**
- T045-T048: Write all tone tests in parallel
- T050-T051: Create tone templates and conciseness logic - can run in parallel

**Phase 5 (US3) Parallelization:**
- T055-T059: Write all conversation tests in parallel
- T060-T063: Implement conversation logic in parallel

**Phase 12 Parallelization:**
- T112-T119: All testing/performance/validation tasks - independent, run in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

**Execution Order** (no timeline estimates per planning guidelines):

1. ✅ Complete Phase 1: Setup
2. ✅ Complete Phase 2: Foundational
3. ✅ Complete Phase 3: User Story 1 (includes Safety Guardian per FR-003)
4. Complete Phase 8: API Key Rotation (critical for quota handling)
5. Complete Phase 9: Document Indexing (critical for data availability)
6. Complete Phase 12: Testing & Polish
7. **STOP and VALIDATE**: Run full test suite, verify all success criteria
8. **Deploy MVP**: Publish to GitHub, README, API docs, container ready

**MVP Deliverables**:
- ✅ Students can ask textbook questions and receive accurate, sourced answers
- ✅ Responses verified by Safety Guardian (zero hallucinations)
- ✅ API key rotation handles quota limits gracefully
- ✅ < 2 second latency p99
- ✅ All tests passing, 80%+ coverage
- ✅ Docker image builds and runs
- ✅ Basic integration with Docusaurus

### Incremental Delivery (Add Features)

**After MVP Validation** (order of execution):
1. Add Phase 4: User Story 2 (Multi-tone communication)
2. Add Phase 5: User Story 3 (Multi-turn conversations)
3. Add Phase 6: Frontend React component
4. Add Phase 7: Advanced Safety Guardian features
5. Add Phase 11: Full Docusaurus integration
6. Add Phase 12: Final testing & polish

**Bonus Features** (out of scope for Phase 2):
- Better-Auth integration (Phase 2.5+)
- Content personalization based on skill level (Phase 2.5+)
- Urdu translation layer (Phase 2.5+)

### Parallel Team Strategy

**With 3+ developers** (no timeline estimates):
1. **Developer A**: Phase 1 + Phase 2 (Foundation) → Phase 3 (US1 implementation)
2. **Developer B**: Phase 8 (API Key Rotation) → Phase 9 (Indexing)
3. **Developer C**: Phase 6 (Frontend) + Phase 11 (Docusaurus) → Phase 12 (Testing)

All converge on Phase 12 for comprehensive testing before MVP release.

---

## Task Summary

| Phase | Description | Task Count | Parallel Opportunities |
|-------|-------------|------------|----------------------|
| Phase 1 | Setup | 10 | T003-T007, T009-T010 (8 tasks) |
| Phase 2 | Foundational | 18 | T012-T014, T015/T017/T020/T022, T026-T027 (12 tasks) |
| Phase 3 | US1: Q&A + Safety Guardian | 20 | T030-T044e (tests first, then implementation) |
| Phase 4 | US2: Tone | 10 | T045-T048 (all tone tests), T050-T051 (tone logic) |
| Phase 5 | US3: History | 11 | T055-T065 (all tests + implementation) |
| Phase 6 | Frontend | 11 | T066-T076 (tests + implementation) |
| Phase 7 | Safety Guardian Advanced | 7 | T077-T083 (advanced features) |
| Phase 8 | API Key Rotation | 7 | T085-T091 (tests + implementation) |
| Phase 9 | Indexing | 8 | T092-T099 + T094a (tests + implementation) |
| Phase 10 | Retention | 6 | T100-T105 (tests + implementation) |
| Phase 11 | Docusaurus Integration | 6 | T106-T111 (tests + implementation) |
| Phase 12 | Testing & Polish | 14 | T112-T125 (all testing, can parallelize) |
| **TOTAL** | | **~128 tasks** | **~60+ can run in parallel** |

---

## Notes

- [P] tasks = different files, no inter-task dependencies
- [Story] label = maps task to specific user story for traceability
- Tests written FIRST per TDD principle, must FAIL before implementation
- Each user story independently completable and deployable as MVP increment
- Phase 2 is blocking - NO user story work until foundational complete
- Phase 7 (Safety Guardian) must complete by end of Phase 3 (before US1 release)
- Phase 9 (Indexing) must complete before US1 release (no data without indexing)
- Run `pytest backend/tests/ -v --cov=backend/src` after each phase for validation
- Commit after each task or logical group of parallel tasks
- All endpoints require 100% type hints (FastAPI enforces at runtime)
- All public functions require Google-style docstrings
- Bundle size must be < 50KB gzipped for React component
- Performance targets: < 2s (p99) end-to-end, < 200ms Qdrant, < 50ms agent handoffs
