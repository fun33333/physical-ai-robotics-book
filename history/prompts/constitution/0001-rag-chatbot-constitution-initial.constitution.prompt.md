---
id: "0001"
title: "rag-chatbot-constitution-initial"
stage: constitution
date: 2025-12-19
surface: cli
model: claude-opus-4-5
feature: none
branch: main
user: user
command: /sp.constitution
labels: [constitution, rag, chatbot, docusaurus, gemini]
links:
  spec: none
  ticket: none
  adr: none
  pr: none
files:
  - path: .specify/memory/constitution.md
    action: created
tests:
  - none
---

## Prompt

User provided comprehensive project constitution input for "Physical AI & Humanoid Robotics Textbook with Integrated RAG Chatbot" including:

**Core Principles (6):**
1. Content-First RAG Architecture - Chatbot retrieves knowledge exclusively from book chapters
2. Gemini Free API Efficiency - Optimize for 15 req/min rate limits; no premium APIs
3. Modular Integration - Book (Phase 1) and Chatbot (Phase 2) operate independently
4. User Context Binding - Selected text accompanies user queries to chatbot
5. Test-First Development - TDD mandatory with 80% coverage
6. Multi-Tone Responses - English, Roman Urdu, and Bro-Guide style with Karachi flavor

**Technical Stack:**
- Backend: FastAPI (Python 3.10+), Qdrant Cloud Free, Gemini Free API, LangChain, Neon Postgres
- Frontend: Docusaurus v3.x, React chatbot component, JavaScript text selection
- Testing: 80% coverage, <2s latency (p99), <200ms Qdrant search

**Constraints:**
- Gemini: 15 req/min
- Qdrant: 20MB vectors
- Neon: 0.5GB storage
- Zero hallucinations policy

## Response snapshot

Created constitution v1.0.0 at `.specify/memory/constitution.md` with:
- 6 core principles with rules and rationale
- Technical Standards (Backend, Frontend, Data Flow, Testing)
- Constraints & Limits tables (API quotas, code quality, document processing)
- Deployment Standards (Phase 1 book, Phase 2 chatbot backend/frontend)
- Governance section (amendment process, versioning policy, compliance)
- Sync Impact Report embedded as HTML comment
- All dependent templates validated for compatibility

## Outcome

- Impact: Initial constitution established for Physical AI textbook + RAG chatbot project
- Tests: N/A (constitution document)
- Files: 1 file created (.specify/memory/constitution.md)
- Next prompts: /sp.specify for feature specifications, /sp.plan for implementation planning
- Reflection: Comprehensive input enabled detailed constitution without clarification rounds

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): Standard /sp.constitution command
- Next experiment: Test constitution compliance in first feature spec
