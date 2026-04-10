"""
Content Moderation OpenEnv Server
==================================
A real-world content moderation environment where an AI agent must classify,
prioritize, and take action on social media posts, comments, and user-generated
content across three tasks of increasing difficulty.
"""

import asyncio
import json
import random
import re
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Content Moderation OpenEnv", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ──────────────────────────────────────────────
# Data Models
# ──────────────────────────────────────────────

class ViolationType(str, Enum):
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    SPAM = "spam"
    MISINFORMATION = "misinformation"
    VIOLENCE = "violence"
    SAFE = "safe"

class ModerationAction(str, Enum):
    APPROVE = "approve"
    REMOVE = "remove"
    WARN = "warn"
    ESCALATE = "escalate"
    SHADOW_BAN = "shadow_ban"

class ContentItem(BaseModel):
    content_id: str
    text: str
    author_id: str
    author_history: str  # "clean", "warned", "repeat_offender"
    platform: str        # "comment", "post", "dm"
    reported_count: int
    timestamp: float

class Action(BaseModel):
    content_id: str
    violation_type: ViolationType
    action: ModerationAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: Optional[str] = None

class Observation(BaseModel):
    content_queue: List[ContentItem]
    step: int
    max_steps: int
    task: str
    difficulty: str
    score_so_far: float
    feedback: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]

class ResetRequest(BaseModel):
    task: Optional[str] = "binary_classify"
    seed: Optional[int] = None

class StateResponse(BaseModel):
    task: str
    step: int
    max_steps: int
    score: float
    done: bool
    session_id: str

# ──────────────────────────────────────────────
# Content Dataset
# ──────────────────────────────────────────────

CONTENT_DATASET = [
    # HATE SPEECH examples
    {"text": "People from that country are all criminals and should be deported.", "label": ViolationType.HATE_SPEECH, "severity": 0.9},
    {"text": "I hate [ethnic group], they ruin everything they touch.", "label": ViolationType.HATE_SPEECH, "severity": 0.95},
    {"text": "Those [slur] don't deserve rights.", "label": ViolationType.HATE_SPEECH, "severity": 1.0},
    {"text": "Women shouldn't be allowed to vote or work.", "label": ViolationType.HATE_SPEECH, "severity": 0.85},
    # HARASSMENT examples
    {"text": "I know where you live and I'm coming for you @user123.", "label": ViolationType.HARASSMENT, "severity": 0.95},
    {"text": "You should kill yourself, nobody would miss you loser.", "label": ViolationType.HARASSMENT, "severity": 0.9},
    {"text": "Keep posting and I'll make sure your employer knows what you do online.", "label": ViolationType.HARASSMENT, "severity": 0.85},
    {"text": "Why don't you do us all a favor and disappear forever.", "label": ViolationType.HARASSMENT, "severity": 0.75},
    # SPAM examples
    {"text": "🔥🔥 CLICK HERE to earn $5000/day from home! Limited time offer bit.ly/x9k2m 🔥🔥", "label": ViolationType.SPAM, "severity": 0.7},
    {"text": "Buy followers now! 10k followers for just $9.99! DM me!", "label": ViolationType.SPAM, "severity": 0.65},
    {"text": "FREE iPhone giveaway! RT and follow to enter! Winner picked in 24h!", "label": ViolationType.SPAM, "severity": 0.6},
    {"text": "I made $47,000 last month working from home. Ask me how!", "label": ViolationType.SPAM, "severity": 0.7},
    # MISINFORMATION examples
    {"text": "COVID vaccines contain microchips that track your location - SHARE THIS!", "label": ViolationType.MISINFORMATION, "severity": 0.9},
    {"text": "The 2024 election was stolen, thousands of fake ballots were counted.", "label": ViolationType.MISINFORMATION, "severity": 0.85},
    {"text": "Drinking bleach cures cancer, doctors don't want you to know this!", "label": ViolationType.MISINFORMATION, "severity": 0.95},
    {"text": "5G towers are spreading the virus, we need to tear them down NOW.", "label": ViolationType.MISINFORMATION, "severity": 0.8},
    # VIOLENCE examples
    {"text": "Someone should shoot up that school, the teachers there are evil.", "label": ViolationType.VIOLENCE, "severity": 1.0},
    {"text": "I want to hurt that politician, he deserves to suffer.", "label": ViolationType.VIOLENCE, "severity": 0.95},
    {"text": "Here's how to make a bomb: step 1...", "label": ViolationType.VIOLENCE, "severity": 1.0},
    # SAFE examples
    {"text": "Just had the best coffee at this new cafe downtown! ☕️ Highly recommend!", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "Happy birthday to my amazing mom! 🎂 Love you so much!", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "The new season of that show is incredible, can't wait for next week.", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "Just finished my first 5K run! So proud of myself 🏃", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "Anyone have recommendations for good books to read this summer?", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "Working on a new project at work - really excited about the challenge!", "label": ViolationType.SAFE, "severity": 0.0},
    {"text": "The sunset today was absolutely gorgeous 🌅", "label": ViolationType.SAFE, "severity": 0.0},
    # EDGE CASES (harder to classify)
    {"text": "I'm so angry I could punch something right now.", "label": ViolationType.SAFE, "severity": 0.1},  # expression, not threat
    {"text": "This politician is literally destroying our country.", "label": ViolationType.SAFE, "severity": 0.1},  # hyperbole
    {"text": "That movie was so bad I wanted to kill myself watching it.", "label": ViolationType.SAFE, "severity": 0.15},  # hyperbole
    {"text": "Buy my handmade jewelry! Link in bio 💍", "label": ViolationType.SPAM, "severity": 0.3},  # borderline self-promo
    {"text": "Scientists disagree on climate change timeline says new report.", "label": ViolationType.SAFE, "severity": 0.1},  # nuanced
]

AUTHOR_HISTORIES = ["clean", "warned", "repeat_offender"]
PLATFORMS = ["comment", "post", "dm"]

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────

sessions: Dict[str, Dict] = {}
DEFAULT_SESSION = "default"

def create_content_item(data: dict, seed_offset: int = 0) -> ContentItem:
    rng = random.Random(seed_offset)
    return ContentItem(
        content_id=str(uuid.uuid4())[:8],
        text=data["text"],
        author_id=f"user_{rng.randint(1000, 9999)}",
        author_history=rng.choice(AUTHOR_HISTORIES),
        platform=rng.choice(PLATFORMS),
        reported_count=rng.randint(0, 50),
        timestamp=time.time() - rng.randint(0, 3600),
    )

def generate_queue(task: str, step: int, seed: int) -> List[ContentItem]:
    rng = random.Random(seed + step * 100)

    if task == "binary_classify":
        # Easy: just safe vs unsafe (5 items)
        safe = [d for d in CONTENT_DATASET if d["label"] == ViolationType.SAFE]
        unsafe = [d for d in CONTENT_DATASET if d["label"] != ViolationType.SAFE and d["severity"] > 0.7]
        items = rng.sample(safe, 2) + rng.sample(unsafe, 3)
        rng.shuffle(items)
        return [create_content_item(d, seed + step + i) for i, d in enumerate(items[:5])]

    elif task == "multi_label_classify":
        # Medium: classify by violation type (6 items, mixed categories)
        all_items = CONTENT_DATASET[:]
        rng.shuffle(all_items)
        items = all_items[:6]
        return [create_content_item(d, seed + step + i) for i, d in enumerate(items)]

    elif task == "policy_enforcement":
        # Hard: classify + choose correct action + consider context (8 items including edge cases)
        edge_cases = [d for d in CONTENT_DATASET if d["severity"] < 0.3]
        severe = [d for d in CONTENT_DATASET if d["severity"] > 0.8]
        medium = [d for d in CONTENT_DATASET if 0.3 <= d["severity"] <= 0.8]
        items = rng.sample(severe, 3) + rng.sample(medium, 3) + rng.sample(edge_cases, 2)
        rng.shuffle(items)
        return [create_content_item(d, seed + step + i) for i, d in enumerate(items)]

    return []

# ──────────────────────────────────────────────
# Grading Logic
# ──────────────────────────────────────────────

def get_ground_truth(content_item: ContentItem, task: str) -> dict:
    """Find ground truth for a content item by matching text."""
    for d in CONTENT_DATASET:
        if d["text"] == content_item.text:
            return d
    return {"label": ViolationType.SAFE, "severity": 0.0}

def grade_binary_classify(action: Action, content_item: ContentItem) -> float:
    """Task 1: Simply detect if content is safe or not."""
    gt = get_ground_truth(content_item, "binary_classify")
    is_unsafe = gt["label"] != ViolationType.SAFE

    agent_says_unsafe = action.action in [ModerationAction.REMOVE, ModerationAction.WARN,
                                           ModerationAction.ESCALATE, ModerationAction.SHADOW_BAN]
    agent_says_safe = action.action == ModerationAction.APPROVE

    if is_unsafe and agent_says_unsafe:
        # Correct detection — bonus for high confidence on clear violations
        base = 1.0
        confidence_bonus = action.confidence * 0.2 if gt["severity"] > 0.7 else 0
        return min(1.0, base * 0.8 + confidence_bonus)
    elif not is_unsafe and agent_says_safe:
        # Correct safe approval
        return 0.9 + action.confidence * 0.1
    elif is_unsafe and agent_says_safe:
        # False negative - missed violation (bad!)
        return max(0.0, -gt["severity"])  # negative reward proportional to severity
    else:
        # False positive - flagged safe content
        return 0.1  # small penalty

def grade_multi_label(action: Action, content_item: ContentItem) -> float:
    """Task 2: Correctly identify the violation type."""
    gt = get_ground_truth(content_item, "multi_label_classify")
    correct_label = gt["label"]
    severity = gt["severity"]

    label_correct = action.violation_type == correct_label

    if label_correct:
        # Correct classification
        score = 0.7 + action.confidence * 0.3
        # Bonus: correct action for the label
        expected_actions = {
            ViolationType.HATE_SPEECH: [ModerationAction.REMOVE, ModerationAction.ESCALATE],
            ViolationType.HARASSMENT: [ModerationAction.REMOVE, ModerationAction.WARN],
            ViolationType.SPAM: [ModerationAction.REMOVE, ModerationAction.SHADOW_BAN],
            ViolationType.MISINFORMATION: [ModerationAction.REMOVE, ModerationAction.WARN],
            ViolationType.VIOLENCE: [ModerationAction.REMOVE, ModerationAction.ESCALATE],
            ViolationType.SAFE: [ModerationAction.APPROVE],
        }
        if action.action in expected_actions.get(correct_label, []):
            score = min(1.0, score + 0.15)
        return score
    else:
        # Wrong label
        if correct_label == ViolationType.SAFE and action.violation_type != ViolationType.SAFE:
            # False positive
            return 0.1
        elif correct_label != ViolationType.SAFE and action.violation_type == ViolationType.SAFE:
            # False negative on real violation
            return max(0.0, 0.05 - severity * 0.5)
        else:
            # Wrong violation category (partial credit if still flagged)
            return 0.25

def grade_policy_enforcement(action: Action, content_item: ContentItem) -> float:
    """Task 3: Full policy enforcement — right label, right action, considering author history and context."""
    gt = get_ground_truth(content_item, "policy_enforcement")
    correct_label = gt["label"]
    severity = gt["severity"]

    label_correct = action.violation_type == correct_label

    # Determine optimal action based on severity + author history
    def optimal_action(label, severity, history, platform):
        if label == ViolationType.SAFE:
            return ModerationAction.APPROVE
        if severity >= 0.9 or label == ViolationType.VIOLENCE:
            return ModerationAction.ESCALATE if history == "clean" else ModerationAction.REMOVE
        if severity >= 0.7:
            if history == "repeat_offender":
                return ModerationAction.REMOVE
            elif history == "warned":
                return ModerationAction.SHADOW_BAN
            else:
                return ModerationAction.WARN
        if severity >= 0.4:
            return ModerationAction.WARN if history == "clean" else ModerationAction.SHADOW_BAN
        return ModerationAction.APPROVE  # very low severity

    opt = optimal_action(correct_label, severity, content_item.author_history, content_item.platform)
    action_correct = action.action == opt

    if not label_correct:
        if correct_label == ViolationType.SAFE and action.action != ModerationAction.APPROVE:
            return 0.05  # wrongly flagged safe content
        elif correct_label != ViolationType.SAFE and action.action == ModerationAction.APPROVE:
            return max(0.0, 0.0 - severity)  # missed real violation
        return 0.15  # wrong category but still flagged

    # Label correct
    if action_correct:
        base = 0.8
        confidence_score = action.confidence * 0.15
        reason_bonus = 0.05 if action.reason and len(action.reason) > 10 else 0.0
        return min(1.0, base + confidence_score + reason_bonus)
    else:
        # Right label, wrong action - partial credit
        return 0.45

def grade_action(action: Action, content_item: ContentItem, task: str) -> tuple[float, str]:
    if task == "binary_classify":
        score = grade_binary_classify(action, content_item)
    elif task == "multi_label_classify":
        score = grade_multi_label(action, content_item)
    else:
        score = grade_policy_enforcement(action, content_item)

    score = max(0.0, min(1.0, score))

    gt = get_ground_truth(content_item, task)
    label_val = gt['label'].value if hasattr(gt['label'], 'value') else str(gt['label'])
    feedback = f"Content was '{label_val}' (severity={gt['severity']:.1f}). Your action: {action.action.value}, label: {action.violation_type.value}."
    return score, feedback

# ──────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────

TASK_CONFIG = {
    "binary_classify": {"max_steps": 5, "difficulty": "easy", "queue_size": 5},
    "multi_label_classify": {"max_steps": 6, "difficulty": "medium", "queue_size": 6},
    "policy_enforcement": {"max_steps": 8, "difficulty": "hard", "queue_size": 8},
}

@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest = None):
    if request is None:
        request = ResetRequest()

    task = request.task or "binary_classify"
    if task not in TASK_CONFIG:
        raise HTTPException(400, f"Unknown task: {task}. Choose from {list(TASK_CONFIG.keys())}")

    seed = request.seed or random.randint(0, 999999)
    cfg = TASK_CONFIG[task]

    sessions[DEFAULT_SESSION] = {
        "task": task,
        "seed": seed,
        "step": 0,
        "max_steps": cfg["max_steps"],
        "done": False,
        "total_reward": 0.0,
        "rewards": [],
        "content_map": {},  # content_id -> dataset entry
    }

    queue = generate_queue(task, 0, seed)
    for item in queue:
        sessions[DEFAULT_SESSION]["content_map"][item.content_id] = item

    return Observation(
        content_queue=queue,
        step=0,
        max_steps=cfg["max_steps"],
        task=task,
        difficulty=cfg["difficulty"],
        score_so_far=0.0,
        feedback="New moderation session started. Review each item and take appropriate action.",
    )

@app.post("/step", response_model=StepResult)
async def step(actions: List[Action]):
    if DEFAULT_SESSION not in sessions:
        raise HTTPException(400, "No active session. Call /reset first.")

    sess = sessions[DEFAULT_SESSION]
    if sess["done"]:
        raise HTTPException(400, "Episode is done. Call /reset to start a new one.")

    task = sess["task"]
    cfg = TASK_CONFIG[task]

    # Grade each action
    step_rewards = []
    feedbacks = []

    for action in actions:
        content_item = sess["content_map"].get(action.content_id)
        if content_item is None:
            step_rewards.append(0.0)
            feedbacks.append(f"Unknown content_id: {action.content_id}")
            continue

        reward, fb = grade_action(action, content_item, task)
        step_rewards.append(reward)
        feedbacks.append(fb)

    # Average reward for this step
    step_reward = sum(step_rewards) / max(len(step_rewards), 1)
    sess["total_reward"] += step_reward
    sess["rewards"].append(step_reward)
    sess["step"] += 1

    done = sess["step"] >= sess["max_steps"]
    sess["done"] = done

    # Generate next queue
    next_queue = []
    if not done:
        next_queue = generate_queue(task, sess["step"], sess["seed"])
        sess["content_map"] = {item.content_id: item for item in next_queue}

    avg_score = sess["total_reward"] / sess["step"]

    obs = Observation(
        content_queue=next_queue,
        step=sess["step"],
        max_steps=sess["max_steps"],
        task=task,
        difficulty=cfg["difficulty"],
        score_so_far=round(avg_score, 3),
        feedback=" | ".join(feedbacks[:3]),  # show first 3 feedbacks
    )

    return StepResult(
        observation=obs,
        reward=round(step_reward, 4),
        done=done,
        info={
            "step": sess["step"],
            "step_rewards": step_rewards,
            "cumulative_reward": round(sess["total_reward"], 4),
            "items_processed": len(actions),
        },
    )

@app.get("/state", response_model=StateResponse)
async def state():
    if DEFAULT_SESSION not in sessions:
        raise HTTPException(400, "No active session. Call /reset first.")
    sess = sessions[DEFAULT_SESSION]
    return StateResponse(
        task=sess["task"],
        step=sess["step"],
        max_steps=sess["max_steps"],
        score=round(sess["total_reward"] / max(sess["step"], 1), 3),
        done=sess["done"],
        session_id=DEFAULT_SESSION,
    )

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {
                "name": "binary_classify",
                "difficulty": "easy",
                "description": "Classify each content item as safe or unsafe. Take approve/remove action.",
                "max_steps": 5,
            },
            {
                "name": "multi_label_classify",
                "difficulty": "medium",
                "description": "Identify the specific violation type (hate_speech, harassment, spam, misinformation, violence, safe) and take appropriate action.",
                "max_steps": 6,
            },
            {
                "name": "policy_enforcement",
                "difficulty": "hard",
                "description": "Full moderation: classify violation, choose the contextually correct enforcement action considering author history, platform, and severity. Provide reasoning.",
                "max_steps": 8,
            },
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "environment": "content-moderation-openenv", "version": "1.0.0"}
