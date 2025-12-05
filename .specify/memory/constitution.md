<!--
Sync Impact Report:
Version change:  → 1.0.0
List of modified principles: None
Added sections: None
Removed sections: None
Templates requiring updates:
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/*.md: ⚠ pending
Follow-up TODOs:
- TODO(REVIEW_FREQUENCY): The frequency for compliance review is not specified.
-->
# Project Constitution: Physical AI & Humanoid Robotics Book

## 1. Project Overview

This constitution outlines the core principles, standards, and structure for the "Physical AI & Humanoid Robotics" technical book. The book aims to provide comprehensive coverage of Physical AI, from theory to practical deployment, structured as a 13-week learning journey.

## 2. Core Principles

### Principle 1: Comprehensive Coverage
- **Non-negotiable Rule**: The book must cover all aspects of Physical AI, encompassing theoretical foundations, practical implementation, and deployment strategies.
- **Rationale**: To provide a holistic understanding for students transitioning from digital AI to embodied intelligence.

### Principle 2: Structured Learning Journey
- **Non-negotiable Rule**: Content must be organized into a 13-week learning journey, comprising 4 core modules.
- **Rationale**: To facilitate a structured and progressive learning experience.

### Principle 3: Practical Focus
- **Non-negotiable Rule**: The book must include practical details such as hardware specifications, cost breakdowns, and working examples.
- **Rationale**: To enable readers to build and experiment with Physical AI systems.

### Principle 4: Accessibility
- **Non-negotiable Rule**: Explanations must be clear and accessible for students with a background in digital AI, easing their transition to embodied intelligence.
- **Rationale**: To broaden the audience and make complex topics approachable.

## 3. Key Standards

### Standard 1: Content Organization
- **Non-negotiable Rule**: Content organization must strictly follow the 4-module structure: ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA.
- **Rationale**: To maintain a consistent and logical flow throughout the book.

### Standard 2: Hardware Requirements
- **Non-negotiable Rule**: Hardware requirements must include exact models, current prices (2024-2025 USD), and clear justifications.
- **Rationale**: To provide practical and actionable guidance for hardware selection.

### Standard 3: Technical Specifications Verification
- **Non-negotiable Rule**: All technical specifications must be verified against official documentation (ROS 2 Humble, Isaac Sim, Gazebo).
- **Rationale**: To ensure accuracy and reliability of technical information.

### Standard 4: Weekly Breakdown
- **Non-negotiable Rule**: A detailed week-by-week breakdown with clear learning objectives for Weeks 1-13 must be provided.
- **Rationale**: To guide learners through the curriculum effectively.

### Standard 5: Markdown Format
- **Non-negotiable Rule**: Content must be optimized for Docusaurus 3.x Markdown format.
- **Rationale**: To ensure proper rendering and functionality within the Docusaurus framework.

## 4. Content Structure

- **Introduction**: "Why Physical AI Matters" + Course Overview
- **Module 1**: The Robotic Nervous System (ROS 2)
- **Module 2**: The Digital Twin (Gazebo & Unity)
- **Module 3**: The AI-Robot Brain (NVIDIA Isaac)
- **Module 4**: Vision-Language-Action (VLA)
- **Hardware Guide**: Workstation specs, Edge kits, Robot lab options with cost comparisons
- **Weekly Breakdown**: Detailed 13-week curriculum map
- **Capstone Project**: Autonomous Humanoid specification

## 5. Constraints

- **Deployment**: Must deploy successfully to GitHub Pages via GitHub Actions.
- **Responsiveness**: Mobile-responsive design required.
- **Cost Estimates**: All cost estimates in USD with 2024-2025 pricing.
- **Hardware Separation**: Hardware requirements clearly separated into Required, Optional, and Budget alternatives.
- **Content Completeness**: No placeholder content—all sections must be complete.

## 6. Success Criteria

- **Learning Outcomes**: Book covers all learning outcomes listed in the course details.
- **Hardware Tiers**: Hardware section includes 3 tiers: Digital Twin Workstation, Edge Kit, Robot Lab.
- **Navigation Flow**: Navigation flows logically through modules and weekly progression.
- **Capstone Project Definition**: Capstone project clearly defines: voice command → planning → navigation → manipulation.
- **Deployment Pipeline**: Deployment pipeline works end-to-end (commit → build → deploy).

## 7. Bonus Objectives (for extra marks)

- **Claude Code Subagents**: Create Claude Code Subagents for:
  * Module content template generation
  * Hardware comparison table generation
  * Week-by-week content scaffolding
- **Agent Skills**: Create Agent Skills for:
  * Technical spec validation (checking ROS 2/Isaac compatibility)
  * Cost calculation and hardware recommendation logic
  * Docusaurus sidebar auto-generation from content structure

## 8. Governance

### 8.1 Amendment Procedure
Any proposed amendments to this constitution must be submitted as a pull request to the project repository and require approval from the project maintainers.

### 8.2 Versioning Policy
This constitution adheres to semantic versioning (MAJOR.MINOR.PATCH).
- MAJOR: Backward incompatible governance/principle removals or redefinitions.
- MINOR: New principle/section added or materially expanded guidance.
- PATCH: Clarifications, wording, typo fixes, non-semantic refinements.

### 8.3 Compliance Review
The project maintainers will review compliance with this constitution TODO(REVIEW_FREQUENCY): The frequency for compliance review is not specified. to ensure ongoing adherence to its principles and standards.

## 9. Version History

- **CONSTITUTION_VERSION**: 1.0.0
- **RATIFICATION_DATE**: 2025-12-05
- **LAST_AMENDED_DATE**: 2025-12-05
