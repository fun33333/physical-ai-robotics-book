# Tasks: Complete Physical AI Book Content

**Input**: Design documents from /specs/003-complete-book-content/
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: No automated tests required - validation via npm run build and content review.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: [ID] [P?] [Story] Description

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US6)
- Include exact file paths in descriptions

## Path Conventions

All content files are in book-website/docs/. This is a documentation-only project.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare content structure and verify build system

- [X] T001 Verify Docusaurus build works with existing placeholder content in book-website/
- [X] T002 Review existing file structure matches plan.md content structure
- [X] T003 [P] Create content style guide reference in specs/003-complete-book-content/contracts/content-contract.md

---

## Phase 2: Foundational (Introduction Content)

**Purpose**: Core introduction content that establishes context for ALL user stories

**CRITICAL**: Introduction content sets the stage for the entire course

- [X] T004 Write production content for book-website/docs/intro.md (Physical AI course overview)
- [X] T005 [P] Write production content for book-website/docs/why-physical-ai.md (motivation and industry context)
- [X] T006 [P] Write production content for book-website/docs/learning-outcomes.md (course objectives and prerequisites)
- [X] T007 Run npm run build to validate Phase 2 content

**Checkpoint**: Introduction complete - students understand what they will learn

---

## Phase 3: User Story 1 - ROS 2 Fundamentals Learning Path (Priority: P1)

**Goal**: Enable students to learn ROS 2 from fundamentals to building robot control systems

**Independent Test**: Student with Python experience can complete Module 1 and run their first ROS 2 node

### Implementation for User Story 1

- [X] T008 [US1] Write module overview in book-website/docs/module-1/index.md
- [X] T009 [P] [US1] Write ROS 2 installation and architecture in book-website/docs/module-1/ros2-fundamentals.md
- [X] T010 [P] [US1] Write nodes, topics, services guide in book-website/docs/module-1/ros2-nodes-topics-services.md
- [X] T011 [P] [US1] Write actions and lifecycle guide in book-website/docs/module-1/ros2-actions-lifecycle.md
- [X] T012 [P] [US1] Write navigation stack guide in book-website/docs/module-1/ros2-navigation.md
- [X] T013 [P] [US1] Write manipulation guide in book-website/docs/module-1/ros2-manipulation.md
- [X] T014 [US1] Add Module 1 exercises (minimum 3) across section files
- [X] T015 [US1] Add Module 1 project definition to book-website/docs/assessments/projects.md
- [X] T016 [US1] Run npm run build to validate US1 content

**Checkpoint**: Module 1 complete - students can build basic ROS 2 applications

---

## Phase 4: User Story 2 - Simulation Environment Setup (Priority: P1)

**Goal**: Enable students to test robotics algorithms in simulation without physical hardware

**Independent Test**: Student can run a simulated robot in Gazebo that responds to ROS 2 commands

### Implementation for User Story 2

- [X] T017 [US2] Write module overview in book-website/docs/module-2/index.md
- [X] T018 [P] [US2] Write simulation concepts in book-website/docs/module-2/simulation-fundamentals.md
- [X] T019 [P] [US2] Write Gazebo setup and basics in book-website/docs/module-2/gazebo-basics.md
- [X] T020 [P] [US2] Write ROS 2-Gazebo integration in book-website/docs/module-2/gazebo-ros2-integration.md
- [X] T021 [P] [US2] Write Unity Robotics Hub guide in book-website/docs/module-2/unity-robotics.md
- [X] T022 [P] [US2] Write digital twin workflows in book-website/docs/module-2/digital-twin-workflows.md
- [X] T023 [US2] Add URDF examples with comments to simulation files
- [X] T024 [US2] Add Module 2 exercises (minimum 3) across section files
- [X] T025 [US2] Add Module 2 project definition to book-website/docs/assessments/projects.md
- [X] T026 [US2] Run npm run build to validate US2 content

**Checkpoint**: Module 2 complete - students can simulate robots in Gazebo/Unity

---

## Phase 5: User Story 3 - Hardware Selection and Setup (Priority: P2)

**Goal**: Enable students and institutions to make informed hardware purchasing decisions

**Independent Test**: Student can create a hardware shopping list within their budget

### Implementation for User Story 3

- [X] T027 [US3] Write hardware overview in book-website/docs/hardware/index.md
- [X] T028 [P] [US3] Write required hardware with cost tables in book-website/docs/hardware/required-hardware.md
- [X] T029 [P] [US3] Write optional hardware guide in book-website/docs/hardware/optional-hardware.md
- [X] T030 [P] [US3] Write budget alternatives (sub-500 USD) in book-website/docs/hardware/budget-alternatives.md
- [X] T031 [P] [US3] Write software requirements in book-website/docs/hardware/software-requirements.md
- [X] T032 [US3] Add hardware cost tables with 3 tiers (Budget/Recommended/Premium)
- [X] T033 [US3] Add purchase links and last-updated dates to all hardware tables
- [X] T034 [US3] Run npm run build to validate US3 content

**Checkpoint**: Hardware guide complete - students can plan their hardware purchases

---

## Phase 6: User Story 4 - NVIDIA Isaac Platform Mastery (Priority: P2)

**Goal**: Enable students to use industry-standard NVIDIA tools for advanced robotics

**Independent Test**: Student can run Isaac Sim with a robot model and train a basic RL policy

### Implementation for User Story 4

- [X] T035 [US4] Write Isaac platform overview in book-website/docs/module-3/index.md
- [X] T036 [P] [US4] Write Isaac platform introduction in book-website/docs/module-3/isaac-platform-overview.md
- [X] T037 [P] [US4] Write Isaac Sim installation and usage in book-website/docs/module-3/isaac-sim.md
- [X] T038 [P] [US4] Write Isaac ROS perception in book-website/docs/module-3/isaac-ros.md
- [X] T039 [P] [US4] Write Isaac Lab RL guide in book-website/docs/module-3/isaac-lab.md
- [X] T040 [P] [US4] Write perception pipelines in book-website/docs/module-3/perception-pipelines.md
- [X] T041 [US4] Add GPU requirements and fallback options to Isaac content
- [X] T042 [US4] Add Module 3 exercises (minimum 3) across section files
- [X] T043 [US4] Add Module 3 project definition to book-website/docs/assessments/projects.md
- [X] T044 [US4] Run npm run build to validate US4 content

**Checkpoint**: Module 3 complete - students can use NVIDIA Isaac platform

---

## Phase 7: User Story 5 - Vision-Language-Action Systems (Priority: P2)

**Goal**: Enable students to understand cutting-edge VLA models for intelligent robot behavior

**Independent Test**: Student can explain how RT-2 combines vision, language, and action

### Implementation for User Story 5

- [X] T045 [US5] Write VLA overview in book-website/docs/module-4/index.md
- [X] T046 [P] [US5] Write VLA foundations (transformers) in book-website/docs/module-4/vla-foundations.md
- [X] T047 [P] [US5] Write vision models guide in book-website/docs/module-4/vision-models.md
- [X] T048 [P] [US5] Write language integration in book-website/docs/module-4/language-integration.md
- [X] T049 [P] [US5] Write action generation in book-website/docs/module-4/action-generation.md
- [X] T050 [P] [US5] Write end-to-end VLA (RT-2, Octo) in book-website/docs/module-4/end-to-end-vla.md
- [X] T051 [US5] Add architecture diagrams (Mermaid) to VLA content
- [X] T052 [US5] Add Module 4 exercises (minimum 3) across section files
- [X] T053 [US5] Add Module 4 project definition to book-website/docs/assessments/projects.md
- [X] T054 [US5] Run npm run build to validate US5 content

**Checkpoint**: Module 4 complete - students understand VLA model architectures

---

## Phase 8: User Story 6 - Structured 8-Week Learning (Priority: P3)

**Goal**: Provide structured curriculum with weekly breakdown for pacing

**Independent Test**: Student can follow Week 1-2 content and complete checkpoint assessment

### Implementation for User Story 6

- [X] T055 [US6] Write weekly breakdown overview in book-website/docs/weekly-breakdown/index.md
- [X] T056 [P] [US6] Write Week 1-2 schedule (ROS 2 focus) in book-website/docs/weekly-breakdown/week-01-02.md
- [X] T057 [P] [US6] Write Week 3-4 schedule (Simulation focus) in book-website/docs/weekly-breakdown/week-03-04.md
- [X] T058 [P] [US6] Write Week 5-6 schedule (Isaac focus) in book-website/docs/weekly-breakdown/week-05-06.md
- [X] T059 [P] [US6] Write Week 7-8 schedule (VLA + Capstone) in book-website/docs/weekly-breakdown/week-07-08.md
- [X] T060 [US6] Add daily activities and time estimates to all weekly files
- [X] T061 [US6] Add checkpoint criteria to each weekly file
- [X] T062 [US6] Run npm run build to validate US6 content

**Checkpoint**: Weekly breakdown complete - students have pacing guidance

---

## Phase 9: Assessments (Cross-Cutting)

**Purpose**: Complete assessment materials that span all modules

- [X] T063 Write assessments overview in book-website/docs/assessments/index.md
- [X] T064 [P] Complete all 5 project definitions with rubrics in book-website/docs/assessments/projects.md
- [X] T065 [P] Write 8 weekly quiz outlines in book-website/docs/assessments/quizzes.md
- [X] T066 Add capstone project requirements to assessments/projects.md
- [X] T067 Run npm run build to validate assessment content

---

## Phase 10: Polish and Final Validation

**Purpose**: Final quality checks and cross-cutting improvements

- [X] T068 [P] Add Docusaurus admonitions (tip, note, warning) across all content
- [X] T069 [P] Add platform-specific tabs (Ubuntu/Windows) where applicable
- [X] T070 [P] Verify all code examples have expected output
- [X] T071 [P] Verify all internal links resolve correctly
- [X] T072 Check word count >= 500 for all 40 files
- [X] T073 Verify progressive complexity (beginner to advanced flow)
- [X] T074 Run final npm run build with 0 errors
- [X] T075 Manual review of navigation and table of contents

---

## Dependencies and Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Setup - MUST complete before user stories
- **Phases 3-8 (User Stories)**: All depend on Phase 2 completion
  - US1 and US2 are both P1 - can run in parallel
  - US3, US4, US5 are P2 - can run in parallel after P1 or alongside
  - US6 is P3 - lowest priority, can be done last
- **Phase 9 (Assessments)**: Can start after US1 project definition
- **Phase 10 (Polish)**: Depends on all content phases

### User Story Dependencies

- **US1 (ROS 2)**: Foundation - no dependencies on other stories
- **US2 (Simulation)**: Foundation - no dependencies, builds on ROS 2 concepts
- **US3 (Hardware)**: Can be done independently at any time
- **US4 (Isaac)**: Benefits from US1/US2 but independently testable
- **US5 (VLA)**: Benefits from all prior modules but independently testable
- **US6 (Schedule)**: References all modules, do after content is stable

### Parallel Opportunities

Within each User Story phase, tasks marked [P] can run in parallel:
- All section files within a module can be written simultaneously
- Different user stories can be worked on by different contributors

---

## Implementation Strategy

### MVP First (P1 Stories Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (Introduction)
3. Complete Phase 3: US1 (ROS 2 Fundamentals)
4. Complete Phase 4: US2 (Simulation)
5. **STOP and VALIDATE**: Run build, review content quality
6. Deploy to GitHub Pages - MVP ready for student testing

### Incremental Delivery

1. MVP (US1 + US2) -> Core learning path ready
2. Add US3 (Hardware) -> Students can plan purchases
3. Add US4 (Isaac) -> Advanced simulation ready
4. Add US5 (VLA) -> Cutting-edge content ready
5. Add US6 (Schedule) -> Full curriculum structure
6. Complete Assessments + Polish -> Production ready

---

## Task Summary

| Phase | Description | Task Count |
|-------|-------------|------------|
| Phase 1 | Setup | 3 |
| Phase 2 | Foundational | 4 |
| Phase 3 | US1: ROS 2 | 9 |
| Phase 4 | US2: Simulation | 10 |
| Phase 5 | US3: Hardware | 8 |
| Phase 6 | US4: Isaac | 10 |
| Phase 7 | US5: VLA | 10 |
| Phase 8 | US6: Schedule | 8 |
| Phase 9 | Assessments | 5 |
| Phase 10 | Polish | 8 |
| **Total** | | **75** |

---

## Notes

- [P] tasks = different files, can run in parallel
- [Story] label maps task to specific user story
- Each module should have minimum 500 words per file
- Each module needs at least 3 exercises
- Run npm run build after each phase to catch issues early
- Commit after each task or logical group of parallel tasks
