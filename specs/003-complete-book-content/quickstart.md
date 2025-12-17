# Quickstart: Complete Physical AI Book Content

**Date**: 2025-12-12
**Feature**: 003-complete-book-content

## Overview

This quickstart guide provides the fastest path to contributing content to the Physical AI book. Follow these steps to update placeholder content with production-ready material.

---

## Prerequisites

1. **Git**: Clone the repository
2. **Node.js 20+**: For Docusaurus development
3. **Text Editor**: VS Code recommended with Markdown extensions

---

## Quick Setup

```bash
# Clone and setup
git clone https://github.com/fun33333/physical-ai-robotics-book.git
cd physical-ai-robotics-book/book-website

# Install dependencies
npm install

# Start development server
npm run start
```

The site will be available at http://localhost:3000/physical-ai-robotics-book/

---

## Content Structure

All content files are in `book-website/docs/`:

```
docs/
├── intro.md                    # Edit: Course introduction
├── why-physical-ai.md          # Edit: Motivation section
├── learning-outcomes.md        # Edit: Course objectives
├── module-1/                   # ROS 2 content
├── module-2/                   # Simulation content
├── module-3/                   # Isaac content
├── module-4/                   # VLA content
├── hardware/                   # Hardware guides
├── weekly-breakdown/           # Schedule
└── assessments/                # Projects/quizzes
```

---

## Validation Checklist

Before submitting content:

- [ ] Frontmatter complete (title, sidebar_position, description)
- [ ] Minimum 500 words
- [ ] At least 1 code example with expected output
- [ ] At least 1 exercise
- [ ] All links tested
- [ ] Build passes: `npm run build`

---

## Build and Test

```bash
# Local development (hot reload)
npm run start

# Production build test
npm run build

# Serve production build locally
npm run serve
```

---

## Priority Order

Implement content in this order for maximum value:

1. **P1 - Module 1**: ROS 2 Fundamentals (foundation for all other content)
2. **P1 - Module 2**: Simulation (enables learning without hardware)
3. **P2 - Hardware**: Hardware guide (helps students plan purchases)
4. **P2 - Module 3**: Isaac Platform (advanced simulation)
5. **P2 - Module 4**: VLA Systems (cutting-edge AI)
6. **P2 - Assessments**: Projects and quizzes
7. **P3 - Weekly Breakdown**: Schedule (organizational aid)

---

## Getting Help

- Docusaurus docs: https://docusaurus.io/docs
- ROS 2 docs: https://docs.ros.org/en/humble/
- Feature spec: `specs/003-complete-book-content/spec.md`
- Research notes: `specs/003-complete-book-content/research.md`
