---
title: "Language Model Integration"
sidebar_position: 4
description: "Connecting large language models to robotics: instruction following, planning, and reasoning."
---

# Language Model Integration

Large language models (LLMs) offer capabilities that dramatically enhance robot intelligence - from understanding complex instructions to reasoning about plans and explaining decisions. This chapter explores how to effectively integrate LLMs into robotics systems, covering different integration patterns, prompt engineering techniques, and strategies for handling uncertainties in safety-critical applications.

## Overview

In this section, you will:

- Understand different LLM integration patterns for robotics
- Implement instruction parsing and grounding
- Build LLM-based task planners
- Design effective prompts for robot control
- Handle LLM uncertainties and failures safely
- Create conversational robot interfaces

## Prerequisites

- Familiarity with transformer architecture concepts
- Python programming experience
- Understanding of ROS 2 basics
- API access to an LLM (OpenAI, Anthropic, or local models)
- Completed [VLA Foundations](/docs/module-4/vla-foundations)

---

## Integration Patterns

### Overview of Approaches

```
┌─────────────────────────────────────────────────────────────────┐
│                LLM Integration Patterns                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Pattern 1: High-Level Planner                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Natural   │────▶│    LLM      │────▶│   Skill     │      │
│   │   Language  │     │   Planner   │     │   Sequence  │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
│   Pattern 2: Code Generation                                    │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Task      │────▶│    LLM      │────▶│   Python    │      │
│   │   Description│    │   Coder     │     │   Code      │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
│   Pattern 3: Reasoning Engine                                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Scene +   │────▶│    LLM      │────▶│   Decision  │      │
│   │   Query     │     │   Reasoner  │     │   + Explain │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
│   Pattern 4: End-to-End VLA                                     │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Image +   │────▶│  Multimodal │────▶│   Actions   │      │
│   │   Language  │     │    VLM      │     │   Directly  │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern Comparison

| Pattern | Latency | Flexibility | Safety | Use Case |
|---------|---------|-------------|--------|----------|
| **High-Level Planner** | High (1-5s) | High | Good | Complex multi-step tasks |
| **Code Generation** | High (2-10s) | Very High | Requires review | Novel tasks, scripting |
| **Reasoning Engine** | Medium (0.5-2s) | Medium | Good | Decision support, QA |
| **End-to-End VLA** | Low (50-200ms) | Low | Learned | Real-time control |

---

## High-Level Planning

### Task Decomposition

LLMs excel at breaking complex tasks into manageable steps:

```python title="task_planner.py"
"""LLM-based task planner for robotics."""
import json
from dataclasses import dataclass
from typing import List, Optional
from openai import OpenAI

@dataclass
class RobotSkill:
    """Available robot primitive skill."""
    name: str
    description: str
    parameters: dict
    preconditions: List[str]
    effects: List[str]

@dataclass
class PlanStep:
    """Single step in task plan."""
    skill: str
    parameters: dict
    description: str

class TaskPlanner:
    """LLM-based task planner."""

    def __init__(self, model: str = "gpt-4"):
        self.client = OpenAI()
        self.model = model
        self.skills = self._define_skills()

    def _define_skills(self) -> List[RobotSkill]:
        """Define available robot skills."""
        return [
            RobotSkill(
                name="navigate_to",
                description="Move robot base to a location",
                parameters={"location": "string - named location or coordinates"},
                preconditions=["robot is not carrying fragile item"],
                effects=["robot is at location"]
            ),
            RobotSkill(
                name="pick_object",
                description="Pick up an object with the gripper",
                parameters={"object": "string - object name"},
                preconditions=["robot is near object", "gripper is empty"],
                effects=["robot is holding object", "object is not on surface"]
            ),
            RobotSkill(
                name="place_object",
                description="Place held object at a location",
                parameters={"location": "string - where to place"},
                preconditions=["robot is holding object", "robot is near location"],
                effects=["object is at location", "gripper is empty"]
            ),
            RobotSkill(
                name="open_gripper",
                description="Open the robot gripper",
                parameters={},
                preconditions=[],
                effects=["gripper is open"]
            ),
            RobotSkill(
                name="close_gripper",
                description="Close the robot gripper",
                parameters={},
                preconditions=["gripper is open"],
                effects=["gripper is closed"]
            ),
            RobotSkill(
                name="look_at",
                description="Point camera at target",
                parameters={"target": "string - what to look at"},
                preconditions=[],
                effects=["camera is viewing target"]
            ),
        ]

    def _build_system_prompt(self) -> str:
        """Build system prompt with skill definitions."""
        skills_desc = "\n".join([
            f"- {s.name}: {s.description}\n"
            f"  Parameters: {json.dumps(s.parameters)}\n"
            f"  Preconditions: {s.preconditions}\n"
            f"  Effects: {s.effects}"
            for s in self.skills
        ])

        return f"""You are a robot task planner. Given a natural language instruction,
decompose it into a sequence of robot skills.

Available skills:
{skills_desc}

Output a JSON array of steps, each with:
- "skill": skill name
- "parameters": parameter values
- "description": human-readable description

Only use the skills listed above. Ensure preconditions are met before each step.
If the task is impossible with available skills, explain why."""

    def plan(self, instruction: str, scene_context: str = "") -> List[PlanStep]:
        """Generate plan from natural language instruction."""
        user_prompt = f"Instruction: {instruction}"
        if scene_context:
            user_prompt += f"\n\nCurrent scene: {scene_context}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Low temperature for consistent planning
        )

        result = json.loads(response.choices[0].message.content)

        steps = []
        for step_data in result.get("steps", []):
            steps.append(PlanStep(
                skill=step_data["skill"],
                parameters=step_data.get("parameters", {}),
                description=step_data.get("description", "")
            ))

        return steps


# Example usage
planner = TaskPlanner()

instruction = "Put the red cup on the kitchen table"
scene = "Robot is at charging station. Red cup is on the counter. Kitchen table is 3m away."

plan = planner.plan(instruction, scene)
for i, step in enumerate(plan):
    print(f"{i+1}. {step.skill}({step.parameters}) - {step.description}")

# Output:
# 1. navigate_to({'location': 'counter'}) - Move to the counter where the cup is
# 2. look_at({'target': 'red cup'}) - Locate the red cup visually
# 3. pick_object({'object': 'red cup'}) - Pick up the red cup
# 4. navigate_to({'location': 'kitchen table'}) - Move to the kitchen table
# 5. place_object({'location': 'kitchen table'}) - Place the cup on the table
```

### Scene-Aware Planning

```python title="scene_aware_planner.py"
"""Scene-aware planning with visual context."""
import base64
from pathlib import Path

class SceneAwarePlanner(TaskPlanner):
    """Planner that incorporates visual scene understanding."""

    def __init__(self, model: str = "gpt-4-vision-preview"):
        super().__init__(model)

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def plan_with_image(self, instruction: str,
                        image_path: str) -> List[PlanStep]:
        """Generate plan using image context."""
        image_b64 = self._encode_image(image_path)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Instruction: {instruction}\n\n"
                                   "The image shows the current scene. "
                                   "Identify relevant objects and plan accordingly."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        # Parse response and return plan steps
        # (Implementation similar to base class)
        pass
```

---

## Code Generation

### Robot Code Synthesis

LLMs can generate executable robot code for novel tasks:

```python title="code_generator.py"
"""LLM-based code generation for robot control."""
import ast
import textwrap
from typing import Callable, Dict

class RobotCodeGenerator:
    """Generate and execute robot control code."""

    def __init__(self, robot_api: object):
        self.client = OpenAI()
        self.robot = robot_api
        self.safe_globals = self._build_safe_globals()

    def _build_safe_globals(self) -> Dict:
        """Build sandboxed execution environment."""
        return {
            # Math operations
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "round": round,

            # Robot API (read-only introspection)
            "robot": self.robot,
            "get_position": self.robot.get_position,
            "get_joint_states": self.robot.get_joint_states,
            "get_gripper_state": self.robot.get_gripper_state,

            # Safe robot commands (with limits)
            "move_to": self._safe_move_to,
            "pick": self._safe_pick,
            "place": self._safe_place,
            "wait": self._safe_wait,

            # Perception
            "detect_objects": self.robot.detect_objects,
            "get_object_pose": self.robot.get_object_pose,
        }

    def _safe_move_to(self, x: float, y: float, z: float):
        """Move with workspace limits."""
        # Enforce workspace bounds
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        z = max(0.0, min(1.5, z))
        return self.robot.move_to(x, y, z)

    def _safe_pick(self, object_name: str):
        """Pick with validation."""
        if not self.robot.is_object_reachable(object_name):
            raise ValueError(f"Object {object_name} not reachable")
        return self.robot.pick(object_name)

    def _safe_place(self, location):
        """Place with validation."""
        if not self.robot.is_holding_object():
            raise ValueError("Not holding any object")
        return self.robot.place(location)

    def _safe_wait(self, seconds: float):
        """Wait with maximum limit."""
        seconds = min(seconds, 30.0)  # Max 30 second wait
        return self.robot.wait(seconds)

    def generate_code(self, task: str, scene_description: str) -> str:
        """Generate Python code for task."""
        system_prompt = """You are a robot programming assistant. Generate Python code
to accomplish the given task using the available robot API.

Available functions:
- get_position() -> (x, y, z): Get current end-effector position
- get_joint_states() -> dict: Get all joint positions
- detect_objects() -> list: Get detected objects with positions
- get_object_pose(name) -> (x, y, z, qx, qy, qz, qw): Get object pose
- move_to(x, y, z): Move end-effector to position
- pick(object_name): Pick up named object
- place(location): Place held object at location
- wait(seconds): Wait for specified time

Generate clean, safe Python code. Include error handling.
Only use the functions listed above. Do not import any modules.
Wrap your code in ```python``` blocks."""

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Task: {task}\n\nScene: {scene_description}"}
            ],
            temperature=0.1
        )

        # Extract code from response
        content = response.choices[0].message.content
        code_blocks = content.split("```python")
        if len(code_blocks) > 1:
            code = code_blocks[1].split("```")[0]
            return code.strip()
        return ""

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate generated code for safety."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Check for forbidden operations
        forbidden = {"import", "exec", "eval", "open", "__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                return False, "Import statements not allowed"
            if isinstance(node, ast.Name) and any(f in node.id for f in forbidden):
                return False, f"Forbidden identifier: {node.id}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["exec", "eval", "compile"]:
                        return False, f"Forbidden function: {node.func.id}"

        return True, "Code validated successfully"

    def execute_code(self, code: str, dry_run: bool = False) -> dict:
        """Execute validated code."""
        is_valid, message = self.validate_code(code)
        if not is_valid:
            return {"success": False, "error": message}

        if dry_run:
            return {"success": True, "message": "Dry run - code validated", "code": code}

        try:
            exec(code, self.safe_globals)
            return {"success": True, "message": "Execution completed"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Example usage
generator = RobotCodeGenerator(robot_api)

task = "Stack the blocks by size, largest on bottom"
scene = "Three blocks detected: small_block at (0.3, 0.1, 0.05), medium_block at (0.4, 0.2, 0.05), large_block at (0.2, 0.3, 0.05)"

code = generator.generate_code(task, scene)
print("Generated code:")
print(code)

# Validate before execution
result = generator.execute_code(code, dry_run=True)
if result["success"]:
    result = generator.execute_code(code)
```

---

## Prompt Engineering for Robotics

### Effective Prompt Design

```python title="prompt_templates.py"
"""Prompt templates for robotics applications."""

# Task planning prompt
PLANNING_PROMPT = """You are planning actions for a robot assistant.

## Robot Capabilities
{capabilities}

## Current State
- Location: {robot_location}
- Gripper: {gripper_state}
- Holding: {held_object}

## Scene Objects
{scene_objects}

## Task
{task}

## Instructions
1. Break the task into steps using only available capabilities
2. Ensure each step's preconditions are met
3. Consider failure cases and recovery
4. Output as JSON with format: {{"steps": [...]}}

Generate a safe, efficient plan:"""

# Object grounding prompt
GROUNDING_PROMPT = """Given a natural language description and a list of detected objects,
identify which object(s) the description refers to.

## Detected Objects
{objects}

## Description
"{description}"

## Instructions
Return a JSON object with:
- "matches": list of object IDs that match the description
- "confidence": float 0-1 indicating certainty
- "reasoning": brief explanation

If no objects match, return empty matches list with explanation."""

# Error recovery prompt
RECOVERY_PROMPT = """A robot action has failed. Help diagnose and recover.

## Attempted Action
{action}

## Error
{error}

## Current State
{state}

## Available Recovery Options
{recovery_options}

Analyze the failure and suggest recovery steps. Consider:
1. What likely caused the failure?
2. Can the action be retried?
3. Are alternative approaches available?
4. Should the task be aborted?

Respond with JSON: {{"diagnosis": "...", "recovery_plan": [...], "should_abort": bool}}"""


class PromptBuilder:
    """Build prompts for robot LLM queries."""

    @staticmethod
    def build_planning_prompt(
        capabilities: list,
        robot_state: dict,
        scene: dict,
        task: str
    ) -> str:
        """Build task planning prompt."""
        cap_str = "\n".join(f"- {c['name']}: {c['description']}" for c in capabilities)
        obj_str = "\n".join(f"- {o['name']} at {o['position']}" for o in scene.get("objects", []))

        return PLANNING_PROMPT.format(
            capabilities=cap_str,
            robot_location=robot_state.get("location", "unknown"),
            gripper_state=robot_state.get("gripper", "unknown"),
            held_object=robot_state.get("holding", "nothing"),
            scene_objects=obj_str,
            task=task
        )

    @staticmethod
    def build_grounding_prompt(objects: list, description: str) -> str:
        """Build object grounding prompt."""
        obj_str = "\n".join(
            f"- ID: {o['id']}, Type: {o['type']}, "
            f"Color: {o.get('color', 'unknown')}, "
            f"Position: {o['position']}"
            for o in objects
        )

        return GROUNDING_PROMPT.format(
            objects=obj_str,
            description=description
        )
```

### Few-Shot Prompting

```python title="few_shot_examples.py"
"""Few-shot examples for robotics tasks."""

PICK_PLACE_EXAMPLES = [
    {
        "instruction": "Put the apple in the bowl",
        "scene": "apple at (0.3, 0.1, 0.05), bowl at (0.5, 0.2, 0.02)",
        "plan": [
            {"skill": "navigate_to", "params": {"location": "apple"}},
            {"skill": "pick_object", "params": {"object": "apple"}},
            {"skill": "navigate_to", "params": {"location": "bowl"}},
            {"skill": "place_object", "params": {"location": "bowl"}}
        ]
    },
    {
        "instruction": "Move all the cups to the tray",
        "scene": "cup_1 at (0.2, 0.1, 0.05), cup_2 at (0.4, 0.1, 0.05), tray at (0.6, 0.3, 0.01)",
        "plan": [
            {"skill": "navigate_to", "params": {"location": "cup_1"}},
            {"skill": "pick_object", "params": {"object": "cup_1"}},
            {"skill": "navigate_to", "params": {"location": "tray"}},
            {"skill": "place_object", "params": {"location": "tray"}},
            {"skill": "navigate_to", "params": {"location": "cup_2"}},
            {"skill": "pick_object", "params": {"object": "cup_2"}},
            {"skill": "navigate_to", "params": {"location": "tray"}},
            {"skill": "place_object", "params": {"location": "tray"}}
        ]
    }
]

def build_few_shot_prompt(examples: list, new_instruction: str, new_scene: str) -> str:
    """Build few-shot prompt with examples."""
    prompt = "Here are examples of robot task planning:\n\n"

    for i, ex in enumerate(examples):
        prompt += f"Example {i+1}:\n"
        prompt += f"Instruction: {ex['instruction']}\n"
        prompt += f"Scene: {ex['scene']}\n"
        prompt += f"Plan: {json.dumps(ex['plan'], indent=2)}\n\n"

    prompt += "Now plan for this task:\n"
    prompt += f"Instruction: {new_instruction}\n"
    prompt += f"Scene: {new_scene}\n"
    prompt += "Plan:"

    return prompt
```

---

## Safety and Uncertainty Handling

### Confidence Estimation

```python title="confidence_estimation.py"
"""Estimate and handle LLM uncertainty."""
import numpy as np
from typing import List, Tuple

class ConfidenceEstimator:
    """Estimate confidence in LLM outputs."""

    def __init__(self, client, model: str = "gpt-4"):
        self.client = client
        self.model = model

    def get_multiple_responses(self, prompt: str, n: int = 5) -> List[str]:
        """Get multiple responses for consistency check."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=0.7
        )
        return [choice.message.content for choice in response.choices]

    def estimate_plan_confidence(self, prompt: str) -> Tuple[dict, float]:
        """Estimate confidence through self-consistency."""
        responses = self.get_multiple_responses(prompt, n=5)

        # Parse all responses as plans
        plans = []
        for resp in responses:
            try:
                plan = json.loads(resp)
                plans.append(plan)
            except json.JSONDecodeError:
                continue

        if not plans:
            return None, 0.0

        # Check consistency of first steps
        first_steps = [p.get("steps", [{}])[0].get("skill") for p in plans]
        most_common = max(set(first_steps), key=first_steps.count)
        consistency = first_steps.count(most_common) / len(first_steps)

        # Return most consistent plan
        for plan in plans:
            if plan.get("steps", [{}])[0].get("skill") == most_common:
                return plan, consistency

        return plans[0], consistency

    def get_explicit_confidence(self, plan: dict) -> dict:
        """Ask LLM to rate its own confidence."""
        prompt = f"""Rate your confidence in each step of this robot plan.

Plan: {json.dumps(plan, indent=2)}

For each step, rate:
- feasibility (0-1): Can the robot physically do this?
- safety (0-1): Is this step safe?
- likelihood (0-1): Will this achieve the intended effect?

Output JSON with confidence ratings for each step."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)


class SafetyChecker:
    """Verify safety of LLM-generated plans."""

    def __init__(self, workspace_bounds: dict, forbidden_regions: list):
        self.bounds = workspace_bounds
        self.forbidden = forbidden_regions

    def check_plan_safety(self, plan: List[dict]) -> Tuple[bool, List[str]]:
        """Check if plan is safe to execute."""
        issues = []

        for i, step in enumerate(plan):
            skill = step.get("skill")
            params = step.get("parameters", {})

            # Check workspace bounds
            if skill == "move_to":
                pos = params.get("position", [0, 0, 0])
                if not self._in_workspace(pos):
                    issues.append(f"Step {i}: Position {pos} outside workspace")

            # Check forbidden regions
            if skill in ["navigate_to", "move_to"]:
                location = params.get("location") or params.get("position")
                if self._in_forbidden_region(location):
                    issues.append(f"Step {i}: Location in forbidden region")

            # Check for dangerous operations
            if skill == "pick_object":
                obj = params.get("object", "")
                if "human" in obj.lower() or "person" in obj.lower():
                    issues.append(f"Step {i}: Cannot pick up humans")

        return len(issues) == 0, issues

    def _in_workspace(self, position: list) -> bool:
        """Check if position is within workspace."""
        x, y, z = position
        return (self.bounds["x_min"] <= x <= self.bounds["x_max"] and
                self.bounds["y_min"] <= y <= self.bounds["y_max"] and
                self.bounds["z_min"] <= z <= self.bounds["z_max"])

    def _in_forbidden_region(self, location) -> bool:
        """Check if location is in forbidden region."""
        # Simplified check - implement actual geometry
        return False
```

### Human-in-the-Loop Verification

```python title="human_verification.py"
"""Human-in-the-loop verification for critical actions."""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String

class HumanVerificationNode(Node):
    """Request human verification for uncertain actions."""

    def __init__(self):
        super().__init__('human_verification')

        # Parameters
        self.declare_parameter('confidence_threshold', 0.8)
        self.declare_parameter('always_verify_skills', ['pick_object', 'place_object'])

        # Service for verification requests
        self.verify_srv = self.create_service(
            Trigger, '/robot/request_verification',
            self.verification_callback
        )

        # Publisher for verification requests to UI
        self.request_pub = self.create_publisher(
            String, '/ui/verification_request', 10
        )

        # Subscriber for verification responses
        self.response_sub = self.create_subscription(
            String, '/ui/verification_response',
            self.response_callback, 10
        )

        self.pending_verification = None
        self.verification_result = None

    def request_verification(self, action: dict, confidence: float) -> bool:
        """Request human verification for an action."""
        threshold = self.get_parameter('confidence_threshold').value
        always_verify = self.get_parameter('always_verify_skills').value

        needs_verification = (
            confidence < threshold or
            action.get('skill') in always_verify
        )

        if not needs_verification:
            return True  # Auto-approve

        # Send verification request
        request_msg = String()
        request_msg.data = json.dumps({
            'action': action,
            'confidence': confidence,
            'reason': 'Low confidence' if confidence < threshold else 'Requires verification'
        })
        self.request_pub.publish(request_msg)

        # Wait for response (with timeout)
        self.pending_verification = action
        self.verification_result = None

        timeout = 30.0  # seconds
        start = self.get_clock().now()
        while self.verification_result is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.get_clock().now() - start).nanoseconds > timeout * 1e9:
                self.get_logger().warn('Verification timeout - rejecting action')
                return False

        return self.verification_result

    def response_callback(self, msg):
        """Handle verification response from human."""
        response = json.loads(msg.data)
        self.verification_result = response.get('approved', False)

        if not self.verification_result:
            self.get_logger().info(
                f"Action rejected: {response.get('reason', 'No reason given')}"
            )
```

---

## ROS 2 Integration

### LLM Service Node

```python title="llm_service_node.py"
"""ROS 2 service node for LLM queries."""
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String
from example_interfaces.srv import SetBool

class LLMServiceNode(Node):
    """ROS 2 node providing LLM services."""

    def __init__(self):
        super().__init__('llm_service')

        self.callback_group = ReentrantCallbackGroup()

        # Initialize planner
        self.planner = TaskPlanner()
        self.confidence_estimator = ConfidenceEstimator(
            self.planner.client, self.planner.model
        )

        # Planning service
        self.plan_srv = self.create_service(
            PlanTask, '/llm/plan_task',
            self.plan_callback,
            callback_group=self.callback_group
        )

        # Grounding service
        self.ground_srv = self.create_service(
            GroundObject, '/llm/ground_object',
            self.ground_callback,
            callback_group=self.callback_group
        )

        # Scene context subscription
        self.scene_sub = self.create_subscription(
            SceneGraph, '/perception/scene_graph',
            self.scene_callback, 10
        )

        self.current_scene = None
        self.get_logger().info('LLM service node ready')

    def scene_callback(self, msg):
        """Update current scene context."""
        self.current_scene = msg

    async def plan_callback(self, request, response):
        """Handle planning request."""
        try:
            # Build scene context string
            scene_ctx = self._scene_to_string(self.current_scene)

            # Generate plan with confidence
            plan, confidence = self.confidence_estimator.estimate_plan_confidence(
                PromptBuilder.build_planning_prompt(
                    capabilities=self.planner.skills,
                    robot_state=request.robot_state,
                    scene={"objects": self._parse_scene(self.current_scene)},
                    task=request.instruction
                )
            )

            response.success = plan is not None
            response.plan = json.dumps(plan) if plan else ""
            response.confidence = confidence

        except Exception as e:
            self.get_logger().error(f'Planning failed: {e}')
            response.success = False
            response.error = str(e)

        return response

    def _scene_to_string(self, scene) -> str:
        """Convert scene graph to text description."""
        if scene is None:
            return "No scene information available"

        lines = []
        for obj in scene.objects:
            lines.append(f"{obj.label} at ({obj.pose.position.x:.2f}, "
                        f"{obj.pose.position.y:.2f}, {obj.pose.position.z:.2f})")
        return "\n".join(lines)


def main():
    rclpy.init()
    node = LLMServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Action Client Integration

```python title="llm_action_client.py"
"""Integration with ROS 2 action system."""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from control_msgs.action import GripperCommand

class LLMActionExecutor(Node):
    """Execute LLM-generated plans using ROS 2 actions."""

    def __init__(self):
        super().__init__('llm_executor')

        # Action clients
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.gripper_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        # Skill to action mapping
        self.skill_executors = {
            'navigate_to': self._execute_navigate,
            'pick_object': self._execute_pick,
            'place_object': self._execute_place,
            'open_gripper': self._execute_open_gripper,
            'close_gripper': self._execute_close_gripper,
        }

    async def execute_plan(self, plan: List[dict]) -> bool:
        """Execute a complete plan."""
        for i, step in enumerate(plan):
            self.get_logger().info(f"Executing step {i+1}: {step['skill']}")

            executor = self.skill_executors.get(step['skill'])
            if executor is None:
                self.get_logger().error(f"Unknown skill: {step['skill']}")
                return False

            success = await executor(step.get('parameters', {}))
            if not success:
                self.get_logger().error(f"Step {i+1} failed")
                return False

        return True

    async def _execute_navigate(self, params: dict) -> bool:
        """Execute navigation action."""
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'

        # Get pose from location name or coordinates
        location = params.get('location')
        if isinstance(location, str):
            pose = self._location_to_pose(location)
        else:
            pose = location

        goal.pose.pose.position.x = pose[0]
        goal.pose.pose.position.y = pose[1]
        goal.pose.pose.orientation.w = 1.0

        self.nav_client.wait_for_server()
        result = await self.nav_client.send_goal_async(goal)

        return result.status == 4  # SUCCEEDED

    def _location_to_pose(self, location_name: str) -> tuple:
        """Convert location name to coordinates."""
        # Would typically look up from semantic map
        locations = {
            'kitchen': (1.0, 2.0),
            'living_room': (3.0, 1.0),
            'charging_station': (0.0, 0.0),
        }
        return locations.get(location_name, (0.0, 0.0))
```

---

## Exercise 1: Build a Task Planner

:::tip Exercise 1: LLM Task Planner
**Objective**: Implement an LLM-based task planner for a robot.

**Steps**:

1. Define 5+ robot skills with preconditions/effects
2. Create a planning prompt with skill descriptions
3. Implement plan generation with JSON output
4. Add confidence estimation through sampling
5. Test with 5 different task instructions

**Verification**:
```python
planner = TaskPlanner()
plan = planner.plan("Clean up the table")
assert len(plan) > 0
assert all(step.skill in valid_skills for step in plan)
```

**Time Estimate**: 60 minutes
:::

---

## Exercise 2: Object Grounding

:::tip Exercise 2: Language-to-Object Grounding
**Objective**: Ground natural language descriptions to detected objects.

**Steps**:

1. Create object grounding prompt template
2. Handle ambiguous descriptions ("the cup" vs "the red cup")
3. Return confidence scores for matches
4. Handle cases with no matches
5. Test with various description styles

**Test Cases**:
- "the red ball" → should match red_ball object
- "something to drink from" → should match cup objects
- "the unicorn" → should return no matches

**Time Estimate**: 45 minutes
:::

---

## Exercise 3: Safe Execution Pipeline

:::tip Exercise 3: Safety-First Execution
**Objective**: Build a safe LLM-to-robot execution pipeline.

**Steps**:

1. Implement plan validation (workspace bounds, forbidden regions)
2. Add confidence thresholding for auto-approval
3. Create human verification flow for uncertain actions
4. Test with intentionally ambiguous commands
5. Log all decisions for audit

**Safety Requirements**:
- Never execute plans with confidence < 0.6 without verification
- Reject any plan touching forbidden regions
- Log all LLM queries and responses

**Time Estimate**: 90 minutes
:::

---

## Summary

In this chapter, you learned:

- **Integration Patterns**: High-level planning, code generation, reasoning, and end-to-end VLA
- **Task Planning**: Decomposing instructions into robot skill sequences
- **Code Generation**: Synthesizing executable robot code from descriptions
- **Prompt Engineering**: Designing effective prompts for robotics tasks
- **Safety**: Confidence estimation, validation, and human-in-the-loop verification
- **ROS 2 Integration**: Building LLM services and action execution

LLMs provide powerful reasoning and language understanding that complements learned robot policies. The key is choosing the right integration pattern for your latency and reliability requirements, while maintaining safety through validation and human oversight.

Next, explore [Action Generation](/docs/module-4/action-generation) to understand how to convert high-level plans into low-level robot motions.

## Further Reading

- [SayCan Paper](https://arxiv.org/abs/2204.01691) - Grounding Language in Robotic Affordances
- [Code as Policies](https://arxiv.org/abs/2209.07753) - LLM Code Generation for Robotics
- [RT-2 Paper](https://arxiv.org/abs/2307.15818) - Vision-Language-Action Models
- [Language Models Meet Robotics](https://arxiv.org/abs/2306.17842) - Survey of LLM+Robotics
- [PaLM-E](https://arxiv.org/abs/2303.03378) - Embodied Multimodal Language Model
