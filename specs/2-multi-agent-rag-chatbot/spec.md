# Feature Specification: Multi-Agent RAG Chatbot for Textbook

**Feature Branch**: `2-multi-agent-rag-chatbot`
**Created**: 2025-12-19
**Status**: Draft
**Last Amended**: 2025-12-22
**Input**: User description: "Project: Physical AI & Humanoid Robotics Textbook - Integrated RAG Chatbot with Multi-Agent System (Phase 2) Target Audience: Hackathon judges evaluating chatbot functionality, multi-agent coordination, integration quality, and user experience Focus: Build a multi-agent RAG chatbot system that answers questions about the published textbook using Gemini Free API and Qdrant Cloud, with seamless integration into Docusaurus site. Each agent has a specialized role with clear instructions, context awareness, and safety boundaries."

---

## Changelog (Analysis Remediation)

| Date | Issue | Change | Rationale |
|------|-------|--------|-----------|
| 2025-12-22 | M2 | Added session creation logic to Edge Cases | `/sp.analyze` found session handling underspecified |

**Source**: `/sp.analyze` run on 2025-12-22

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Textbook Q&A with Multi-Agent System (Priority: P1)

A student reading the Physical AI & Humanoid Robotics textbook encounters a concept they don't understand and wants an explanation. The student can ask questions about textbook content through the integrated chatbot, which routes the query to the appropriate specialized agent (Content Tutor, Code Explainer, Concept Clarifier, or Safety Guardian) based on the nature of the question.

**Why this priority**: This is the core functionality - enabling students to get accurate, context-aware answers to questions about the textbook content, which is the primary value proposition of the system.

**Independent Test**: A student can ask a question about robotics concepts in the textbook and receive a relevant, accurate response that is sourced from the textbook material. The system should not hallucinate information not present in the textbook.

**Acceptance Scenarios**:

1. **Given** a student has opened a textbook chapter and encountered a concept they don't understand, **When** they ask a question about the concept through the chatbot interface, **Then** the appropriate agent responds with an explanation based only on textbook content.

2. **Given** a student has selected specific text within a textbook chapter, **When** they ask a question about that text, **Then** the response is contextualized to the selected text.

3. **Given** a student submits a query that requires code explanation, **When** the query is processed, **Then** the Code Explainer agent provides a step-by-step breakdown of the relevant code from the textbook.

---

### User Story 2 - Multi-Tone Communication (Priority: P2)

A student wants to receive explanations in their preferred communication style. The system should offer different tone options (English, Roman Urdu, Bro-Guide) that adjust the language and style of responses while maintaining technical accuracy.

**Why this priority**: This enhances accessibility and user engagement by allowing students to receive information in a communication style that resonates with them.

**Independent Test**: A student can select a preferred tone setting and receive responses that match the selected communication style while maintaining technical accuracy.

**Acceptance Scenarios**:

1. **Given** a student has selected a preferred tone setting (English, Roman Urdu, or Bro-Guide), **When** they ask a question, **Then** the response is formatted in the selected tone.

2. **Given** a student changes their tone preference during a conversation, **When** they continue asking questions, **Then** subsequent responses use the new tone.

---

### User Story 3 - Multi-Turn Conversations with History (Priority: P3)

A student wants to have an ongoing conversation with the chatbot about textbook content, building on previous exchanges. The system should maintain context across multiple interactions to enable deeper discussions.

**Why this priority**: This enables more sophisticated learning experiences where students can explore concepts in depth through follow-up questions that reference previous exchanges.

**Independent Test**: A student can ask multiple related questions in succession, and the agents can reference information from previous exchanges in their responses.

**Acceptance Scenarios**:

1. **Given** a student has had a conversation with the chatbot about a particular topic, **When** they ask follow-up questions, **Then** the agents can reference context from previous exchanges.

2. **Given** a student asks a question that references something from an earlier part of the conversation, **When** the query is processed, **Then** the agent provides a contextually appropriate response.

---

### Edge Cases

- What happens when all API keys are exhausted for the day? → System displays fallback message: "Your today's limit of AI Guide is exceeded. Please try again tomorrow."
- How does the system handle queries that aren't covered in the textbook? → Agent redirects with: "This topic isn't covered in our textbook. Would you like me to explain something from the course material instead?"
- What occurs when the vector store is unavailable? → System returns: "I'm temporarily unable to access the textbook database. Please try again in a moment." (No fallback data returned; no hallucination attempted)
- How does the system handle very long user inputs or code snippets? → System truncates inputs gracefully with warning; processes up to reasonable character limit
- What happens when the same user has multiple concurrent sessions? → Conversation history maintained separately per session; no cross-session context leakage
  - **Session Creation**: New session created when: (1) User opens chatbot without existing conversationId, OR (2) User explicitly starts "New Conversation"
  - **Session ID**: UUID generated server-side on first /chat request without conversationId; returned in response for client storage
  - **Session Timeout**: Sessions remain active for 24 hours of inactivity; after timeout, new session created on next request
  - **Session Storage**: Client stores conversationId in localStorage; cleared on "New Conversation" action
- What if query requires external information beyond textbook scope? → Agent asks for clarification or requests rephrasing: "Could you rephrase that?" or "Are you asking about [topic A] or [topic B]?"
- What if agent becomes uncertain about accuracy of response? → Safety Guardian flags uncertainty; agent responds: "I'm not certain about this. Let me verify from the textbook..."

## Requirements *(mandatory)*

### Functional Requirements

**Core Agent System (Orchestrator Pattern with Specialized Sub-Agents):**
- **FR-001**: System MUST implement orchestrator pattern with one Main Coordinator Agent and four specialized sub-agents, each with defined role, tools, and scope:
  - **Main Coordinator Agent**: Routes query through pipeline stages; orchestrates handoffs; maintains overall accuracy
  - **RAG Agent** (specialized for retrieval): Retrieve textbook chapters from Qdrant; handle selectedText context; provide top-k chunks with metadata
  - **Answer/Tutor Agent** (specialized for generation): Generate accurate responses using retrieved context; cite sources; maintain technical accuracy for robotics/ROS 2/Isaac/VLA topics
  - **Tone Agent** (specialized for formatting): Apply user's preferred tone (English, Roman Urdu, Bro-Guide) while preserving accuracy; handle conciseness logic
  - **Safety Guardian Agent** (specialized for validation): Review responses for hallucinations, factual errors, unsourced claims; attempt clarification question if uncertain; rewrite or flag with transparency note
- **FR-002**: System MUST route query through pipeline: Coordinator → RAG Agent (retrieval) → Answer/Tutor Agent (generation) → Tone Agent (formatting) → Safety Guardian (validation) → return to user
- **FR-003**: System MUST NOT bypass Safety Guardian; all responses validated before returning to user (no unreviewed responses)
- **FR-004**: System MUST refuse to answer non-textbook queries with explicit explanation: "This topic isn't covered in our textbook. Would you like me to explain something from the course material instead?"
- **FR-005**: System MUST ask for clarification if query is ambiguous: "Could you rephrase that?" or "Are you asking about [topic A] or [topic B]?"
- **FR-005a**: Safety Guardian MUST attempt clarifying question when detecting potential hallucination (e.g., "Are you asking about X or Y?"). If user's clarification resolves uncertainty → provide accurate response. If unable to clarify → rewrite/fix with transparency note: "Note: I corrected an inaccuracy in my initial response."

**Context & Retrieval:**
- **FR-006**: System MUST retrieve textbook content from Qdrant vector store using query and selectedText as retrieval context
- **FR-007**: System MUST allow users to select text within textbook chapters and pass selectedText as context to agents
- **FR-008**: System MUST ensure selected text influences agent responses in 90%+ of test cases
- **FR-009**: System MUST store and retrieve conversation history from database to maintain multi-turn context
- **FR-010**: System MUST reference prior explanations and conversation context in follow-up responses

**Response Quality:**
- **FR-011**: System MUST provide concise responses by default (1-2 sentences) with prompt for longer explanation: "Ask for longer explanation?" or "Need more details?"
- **FR-012**: System MUST provide full detailed response when user explicitly requests longer explanation
- **FR-013**: System MUST cite or reference textbook chapters/sections in all responses
- **FR-014**: System MUST NEVER hallucinate information; if unsure, agent MUST respond: "I'm not certain about this. Let me verify from the textbook... Actually, I can't find this in our course material."
- **FR-015**: System MUST return response in user's selected tone (English, Roman Urdu, or Bro-Guide) maintaining technical accuracy

**Tone & Personalization:**
- **FR-016**: System MUST support three distinct communication tones:
  - English: Professional, educational, formal tone
  - Roman Urdu: Friendly Karachi-local style with Urdu phrases and casual language
  - Bro-Guide: Casual Karachi slang, colloquial expressions, accessible to local audience
- **FR-017**: System MUST allow users to switch tone preference mid-conversation with immediate effect on subsequent responses
- **FR-018**: System MUST maintain tone consistency across all agents when tone preference changes

**API Management:**
- **FR-019**: System MUST implement automatic API key rotation across multiple Gemini Free API keys when quota exceeded
- **FR-020**: System MUST track quota usage per API key and switch to next available key on quota exhaustion
- **FR-021**: System MUST display exact fallback message when all API keys exhausted: "Your today's limit of AI Guide is exceeded. Please try again tomorrow." (No data returned, no hallucination attempted)
- **FR-022**: System MUST expose API key rotation status and quota information via health check endpoint
- **FR-023**: System MUST log all API key rotations with timestamp, key ID, and reason

**Document Management:**
- **FR-024**: System MUST index all textbook chapters (Docusaurus-generated HTML/Markdown) into vector store; indexing runs in background on startup
- **FR-024a**: While indexing in progress, system MUST accept user queries; return "Indexing in progress..." message and queue queries until vector store ready
- **FR-025**: System MUST split textbook content into semantic chunks with metadata (chapter, section, difficulty level) for retrieval
- **FR-026**: System MUST support document re-indexing on demand without disrupting active conversations

**Integration & Performance:**
- **FR-027**: System MUST integrate chatbot component into Docusaurus website without degrading page load performance
- **FR-028**: System MUST load chatbot asynchronously (non-blocking to page render)
- **FR-029**: System MUST maintain chatbot JavaScript bundle size < 50KB gzipped
- **FR-030**: System MUST provide REST API endpoints for chat operations, indexing, health checks, and vector search

### Non-Functional Requirements

**Performance:**
- **NFR-001**: Response latency MUST be < 2 seconds (p99) including entire orchestrator pipeline (retrieval → generation → formatting → validation)
- **NFR-002**: Vector store search latency MUST be < 200ms for top-k retrieval (RAG Agent)
- **NFR-003**: Agent handoff between pipeline stages MUST complete in < 50ms (Coordinator overhead)
- **NFR-004**: Answer/Tutor Agent generation latency MUST be < 1 second (p99) via Gemini API
- **NFR-005**: Tone Agent formatting MUST complete in < 100ms
- **NFR-006**: Safety Guardian validation MUST complete in < 500ms
- **NFR-007**: API key rotation fallover MUST occur within 100ms

**Reliability & Availability:**
- **NFR-008**: System MUST handle concurrent user sessions without cross-session context leakage
- **NFR-009**: System MUST gracefully degrade when external services (Qdrant, Postgres, Gemini) become unavailable
- **NFR-010**: System MUST provide meaningful error messages to users instead of technical stack traces
- **NFR-011**: System MUST implement health checks for all external dependencies (Qdrant, Gemini keys, Postgres)

**Code Quality:**
- **NFR-012**: Code MUST achieve minimum 80% test coverage (unit tests, integration tests, agent behavior tests)
- **NFR-013**: All Python code MUST comply with Black formatter style requirements
- **NFR-014**: All FastAPI endpoints MUST have 100% type hints
- **NFR-015**: All public functions MUST have Google-style docstrings explaining purpose, parameters, return values
- **NFR-016**: Code MUST pass linting with pylint score > 8.0

**Security & Input Validation:**
- **NFR-017**: System MUST sanitize user queries and selectedText before agent processing
- **NFR-018**: System MUST enforce CORS restrictions to Docusaurus domain only
- **NFR-019**: System MUST store API keys in environment variables only (never in code or config files)
- **NFR-020**: System MUST implement rate limiting at 15 requests/minute per IP address (matching Gemini Free API limit)
- **NFR-021**: System MUST prevent SQL injection through parameterized database queries
- **NFR-022**: System MUST validate all JSON inputs for proper structure before processing

**Observability:**
- **NFR-023**: System MUST implement structured JSON logging including: request_id, endpoint, latency_ms, orchestrator_pipeline_stages, agent_used, api_key_rotations, errors
- **NFR-024**: System MUST track and log all hallucination attempts detected by Safety Guardian with reasoning
- **NFR-025**: System MUST expose metrics endpoint for monitoring response times per agent stage, error rates, and API key exhaustion patterns

**Scalability:**
- **NFR-026**: Vector store usage MUST not exceed 20MB (Qdrant Cloud Free Tier limit)
- **NFR-027**: Database storage usage MUST not exceed 0.5GB (Neon Free Tier limit); conversation history retention policy: Keep only last 30 days; auto-delete older conversations daily at 00:00 UTC
- **NFR-028**: System MUST remain functional under spike to 100 concurrent requests with graceful degradation

### Key Entities

- **User Query**: Textual input from students seeking information about textbook content; includes selectedText context if available
- **Main Coordinator Agent**: Orchestrator that routes query through specialized sub-agents in pipeline: retrieval → generation → formatting → validation
- **Specialized Sub-Agent**: One of four agents with specific role: RAG Agent (retrieval), Answer/Tutor Agent (generation), Tone Agent (formatting), Safety Guardian (validation). Each has defined tools and scope
- **Agent Handoff**: Passing query/context from one agent to next in pipeline; each agent receives output from previous agent as input
- **Conversation**: Sequence of interactions between a user and system; maintains history and context for multi-turn awareness
- **Textbook Content Chunk**: Semantic segment of textbook content (512-token chunks with 50-token overlap) stored in vector database with metadata (chapter, section, difficulty level)
- **API Key Pool**: Collection of multiple Gemini Free API keys with quota tracking and automatic rotation logic
- **Response**: Generated answer from Answer/Tutor Agent, formatted by Tone Agent, validated by Safety Guardian; includes: response text, source references, tone used, conciseness flag, latency metrics, validation status
- **Conversation History**: Database record of user queries, generated responses, tone selection, selected text, agent handoff trace, and validation results for context awareness and audit

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Chatbot accurately answers 90%+ of test queries about textbook content without hallucinated information
- **SC-002**: Selected text from book chapters correctly influences chatbot responses in 90%+ of test cases (context precision > 90%)
- **SC-003**: System responses are delivered within 2 seconds in 99% of cases (p99 latency)
- **SC-004**: All three communication tones (English, Roman Urdu, Bro-Guide) are fully functional and switch correctly across all agents
- **SC-005**: Conversation history enables multi-turn context awareness; agents reference prior exchanges in follow-up responses
- **SC-006**: API key rotation works seamlessly; system automatically switches to next key when individual key quota exceeded
- **SC-007**: System displays exact fallback message when all API keys exhausted (no partial data, no hallucination)
- **SC-008**: Chatbot integration with Docusaurus website causes no observable page load degradation (< 50ms additional load)
- **SC-009**: Students receive concise answers by default; requesting detailed explanation triggers full response without conciseness constraint
- **SC-010**: Safety Guardian successfully catches and blocks/rewrites all hallucinated responses in validation testing
- **SC-011**: All four specialized agents behave according to defined roles in orchestrator pipeline: RAG Agent retrieves accurately, Answer/Tutor Agent explains accurately, Tone Agent maintains tone consistency, Safety Guardian verifies factual accuracy before returning response
- **SC-012**: Document indexing pipeline successfully embeds all textbook chapters into Qdrant without exceeding 20MB storage limit
- **SC-013**: Orchestrator pipeline correctly routes queries through stages: (1) RAG Agent retrieves context, (2) Answer/Tutor Agent generates response, (3) Tone Agent formats, (4) Safety Guardian validates before returning
- **SC-014**: System handles edge cases gracefully: API exhaustion, out-of-scope queries, unavailable services, oversized inputs
- **SC-015**: Code coverage reaches minimum 80% with passing unit tests, integration tests, and agent behavior tests

## Constraints

### Technology Stack (MANDATORY)
- **NO LangChain**: Must use OpenAI Agents SDK exclusively
- **Backend Framework**: FastAPI only (Python 3.10+)
- **LLM Provider**: Google Gemini Free API only (no OpenAI, no alternatives)
- **Vector Database**: Qdrant Cloud Free Tier (no on-premise, no alternatives)
- **Database**: Neon Serverless Postgres only
- **Frontend**: React component (no Vue, Svelte, etc.)
- **Document Source**: Docusaurus-generated HTML/Markdown only (no PDFs, no external sources)

### API & Quota Constraints
- **Gemini Free API**: 15 requests/minute rate limit per key → MUST implement auto-rotation across multiple keys
- **Qdrant Cloud Free**: 20MB maximum vector storage → MUST optimize embeddings for compression
- **Neon Free Tier**: 0.5GB storage → MUST implement conversation history retention policy
- **Rate Limiting**: FastAPI MUST enforce 15 req/min/IP maximum

### Response Quality Constraints (NON-NEGOTIABLE)
- **Accuracy**: 90%+ minimum for factual correctness without hallucinations
- **Conciseness**: Responses brief by default (1-2 sentences); full explanation only on explicit user request
- **Context Binding**: Selected text MUST influence responses (> 90% precision)
- **Latency**: < 2 seconds (p99) including all processing steps
- **Honesty**: Never hallucinate; ask for clarification if unsure
- **Truthfulness**: All answers traceable to textbook content with chapter/section references

## Out of Scope (Phase 2.5+)

- User authentication & profile management (Better-Auth integration) - marked as Bonus Feature
- Content personalization based on user skill level - marked as Bonus Feature
- Urdu translation layer for chapters - marked as Bonus Feature
- Admin dashboard for query monitoring and analytics
- Custom embedding models (using framework defaults only)
- Multi-document RAG beyond Physical AI textbook
- Speech-to-text or text-to-speech capabilities
- Advanced search strategies (hybrid search, reranking, multi-vector retrieval)
- Custom Gemini fine-tuning
- Certification or credential issuance
- Real-time collaborative sessions
- Video content integration

## Assumptions

1. Students have basic Python programming experience
2. Students have access to Ubuntu 22.04 LTS or equivalent development environment
3. Textbook chapters are available as Docusaurus-generated static HTML and/or Markdown files
4. All textbook content is in English (Roman Urdu and Bro-Guide are tone adaptations, not translations)
5. Gemini Free API keys are available and functional; quota is per-key
6. Neon Postgres connection remains stable without unexpected rate limiting
7. Qdrant Cloud Free cluster is accessible and operational
8. Docusaurus site is deployed and accessible for integration
9. User device has JavaScript enabled for chatbot component
10. Textbook content is stable during implementation (no major structural changes expected)

## Dependencies

1. Physical AI textbook (Phase 1, 003-complete-book-content) MUST be complete and deployed to Docusaurus
2. Docusaurus 3.x site must be operational and accessible for integration
3. Gemini Free API documentation and authentication available
4. Qdrant Cloud Free account provisioned with API access
5. Neon Postgres account provisioned with database created