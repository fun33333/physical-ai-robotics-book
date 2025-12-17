# Content Contract: Physical AI Book

## Purpose

This contract defines the structure and requirements for all content in the Physical AI book. Contributors MUST follow this contract to ensure consistency.

---

## Section Contract

Every markdown file MUST include:

### 1. Frontmatter (Required)

```yaml
---
title: "Section Title"
sidebar_position: 1
description: "Brief description for SEO (max 160 chars)"
---
```

### 2. Structure (Required)

| Section | Required | Description |
|---------|----------|-------------|
| Title (H1) | YES | Same as frontmatter title |
| Overview | YES | 1-2 paragraphs introducing topic |
| Prerequisites | IF APPLICABLE | List of required knowledge |
| Core Content | YES | Main educational material |
| Code Examples | YES (technical) | Runnable code with output |
| Exercises | YES | At least 1 hands-on activity |
| Summary | RECOMMENDED | Key takeaways |
| Further Reading | OPTIONAL | External resources |

### 3. Content Requirements

| Requirement | Minimum |
|-------------|---------|
| Word count | 500 words |
| Code examples (technical sections) | 1 |
| Exercises | 1 |
| Internal links | 1 (to related content) |

---

## Code Example Contract

```markdown
```python title="filename.py"
# Clear comments explaining the code
import rclpy
from rclpy.node import Node

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        # Initialize publisher
```

**Expected Output**:
```
[INFO] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [minimal_publisher]: Publishing: "Hello World: 1"
```

**Explanation**: Brief explanation of what the code does.
```

---

## Exercise Contract

```markdown
:::tip Exercise N: Exercise Title
**Objective**: What the student will learn/accomplish

**Time Estimate**: X minutes

**Steps**:
1. First step with clear instruction
2. Second step with expected result
3. Third step...

**Expected Result**: What success looks like

**Hints** (optional):
- Hint 1
- Hint 2
:::
```

---

## Hardware Table Contract

```markdown
| Component | Budget | Recommended | Premium |
|-----------|--------|-------------|---------|
| GPU | GTX 1660 (~$200) | RTX 3060 (~$350) | RTX 4090 (~$1600) |
| RAM | 16GB (~$50) | 32GB (~$100) | 64GB (~$200) |

*Prices in USD, last updated: December 2024*
```

---

## Project Contract

```markdown
## Project: Project Title

### Overview
Brief description of what the project accomplishes.

### Learning Objectives
- Objective 1
- Objective 2
- Objective 3

### Requirements
1. Functional requirement 1
2. Functional requirement 2
3. Functional requirement 3

### Deliverables
- [ ] Deliverable 1
- [ ] Deliverable 2
- [ ] Deliverable 3

### Rubric

| Criterion | Points | Description |
|-----------|--------|-------------|
| Functionality | 40 | Code works as specified |
| Code Quality | 30 | Clean, documented code |
| Documentation | 20 | README and comments |
| Creativity | 10 | Extra features or improvements |

**Total**: 100 points
**Passing**: 70 points
```

---

## Quiz Contract

```markdown
## Quiz: Week N

### Topics Covered
- Topic 1
- Topic 2
- Topic 3

### Question Types
- Multiple choice: 5 questions
- Short answer: 3 questions
- Code review: 2 questions

### Sample Questions

**Q1 (Multiple Choice)**: Question text?
- A) Option A
- B) Option B
- C) Option C (correct)
- D) Option D

**Q2 (Short Answer)**: Question text?
Expected answer: Brief answer text

### Passing Score
70% (7/10 questions)
```
