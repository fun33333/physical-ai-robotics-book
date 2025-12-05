---
PHR_ID: 20251205120000
STAGE: constitution
TITLE: Initial project constitution
DATE: 2025-12-05
---

## Prompt Text

```text
/sp.constitution
Project: "Physical AI & Humanoid Robotics" technical book using Docusaurus, deployed to GitHub Pages

Core principles:
- Comprehensive coverage of Physical AI from theory to deployment
- Structured as a 13-week learning journey with 4 core modules
- Practical focus: hardware specs, cost breakdowns, and working examples
- Accessibility: Clear explanations for students transitioning from digital AI to embodied intelligence

Key standards:
- Content organization follows the 4-module structure: ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA
- Hardware requirements must include exact models, prices, and justifications
- All technical specifications verified against official docs (ROS 2 Humble, Isaac Sim, Gazebo)
- Week-by-week breakdown with clear learning objectives (Weeks 1-13)
- Markdown format optimized for Docusaurus 3.x

Content structure:
- Introduction: "Why Physical AI Matters" + Course Overview
- Module 1: The Robotic Nervous System (ROS 2)
- Module 2: The Digital Twin (Gazebo & Unity)
- Module 3: The AI-Robot Brain (NVIDIA Isaac)
- Module 4: Vision-Language-Action (VLA)
- Hardware Guide: Workstation specs, Edge kits, Robot lab options with cost comparisons
- Weekly Breakdown: Detailed 13-week curriculum map
- Capstone Project: Autonomous Humanoid specification

Constraints:
- Must deploy successfully to GitHub Pages via GitHub Actions
- Mobile-responsive design required
- All cost estimates in USD with 2024-2025 pricing
- Hardware requirements clearly separated: Required vs. Optional vs. Budget alternatives
- No placeholder content—all sections must be complete

Success criteria:
- Book covers all learning outcomes listed in the course details
- Hardware section includes 3 tiers: Digital Twin Workstation, Edge Kit, Robot Lab
- Navigation flows logically through modules and weekly progression
- Capstone project clearly defines: voice command → planning → navigation → manipulation
- Deployment pipeline works end-to-end (commit → build → deploy)

Bonus objectives (for extra marks):
- Create Claude Code Subagents for:
  * Module content template generation
  * Hardware comparison table generation
  * Week-by-week content scaffolding
- Agent Skills for:
  * Technical spec validation (checking ROS 2/Isaac compatibility)
  * Cost calculation and hardware recommendation logic
  * Docusaurus sidebar auto-generation from content structure/
```

## Response Text

The project constitution has been created and updated to version 1.0.0.

**New version:** 1.0.0
**Bump rationale:** Initial creation of the constitution.

**Files flagged for manual follow-up:**
- .specify/templates/plan-template.md: ⚠ pending
- .specify/templates/spec-template.md: ⚠ pending
- .specify/templates/tasks-template.md: ⚠ pending
- .specify/templates/commands/*.md: ⚠ pending

**Follow-up TODOs:**
- TODO(REVIEW_FREQUENCY): The frequency for compliance review is not specified.

**Suggested commit message:**
```
docs: Initial project constitution v1.0.0
```