# 🛡️ Content Moderation OpenEnv

A real-world **content moderation environment** for training and evaluating AI agents on trust & safety decisions. Built for the OpenEnv standard — the same decision pipeline used by platform safety teams at scale.

---

## 🌍 Environment Description & Motivation

Content moderation is one of the most consequential real-world tasks performed at scale today. Platforms like Twitter, Meta, YouTube, and TikTok process billions of posts daily, requiring accurate and contextually-aware moderation decisions. Errors in either direction have real consequences:

- **False negatives** (missing harmful content) → real-world harm, user trauma, regulatory risk
- **False positives** (wrongly removing safe content) → free speech concerns, user frustration, chilling effects

This environment simulates a moderation queue where an agent must review content items and decide: what violation is present (if any), and what enforcement action to take — considering the severity, platform context, and the author's history.

**Why this domain fills a gap:**
- Unlike toy NLP benchmarks, this task requires *contextual reasoning*, not just pattern matching
- The action space is non-trivial: 5 actions × 6 violation types × 3 author history states = rich decision space
- Partial credit rewards enable meaningful learning signal at every step
- Hard task includes deliberate edge cases (hyperbole, borderline self-promotion, nuanced political speech) that challenge frontier models

---

## 📐 Action Space

Each step, the agent receives a queue of content items and must submit one decision per item.

```json
[
  {
    "content_id": "a1b2c3d4",
    "violation_type": "harassment",
    "action": "remove",
    "confidence": 0.92,
    "reason": "Direct threat with personal identifying information"
  }
]
```

| Field | Type | Values |
|-------|------|--------|
| `content_id` | string | From observation queue |
| `violation_type` | enum | `hate_speech`, `harassment`, `spam`, `misinformation`, `violence`, `safe` |
| `action` | enum | `approve`, `remove`, `warn`, `escalate`, `shadow_ban` |
| `confidence` | float | 0.0 – 1.0 |
| `reason` | string | Optional. Boosts score on hard task. |

---

## 👁️ Observation Space

Each step returns a queue of content items plus session state:

```json
{
  "content_queue": [
    {
      "content_id": "a1b2c3d4",
      "text": "Sample post content here",
      "author_id": "user_4821",
      "author_history": "warned",
      "platform": "comment",
      "reported_count": 12,
      "timestamp": 1712600000.0
    }
  ],
  "step": 2,
  "max_steps": 6,
  "task": "multi_label_classify",
  "difficulty": "medium",
  "score_so_far": 0.73,
  "feedback": "Content was 'harassment' (severity=0.9). Your action: warn, label: harassment."
}
```

| Field | Description |
|-------|-------------|
| `content_queue` | Items to moderate this step |
| `author_history` | `clean`, `warned`, or `repeat_offender` — affects optimal action |
| `platform` | `comment`, `post`, or `dm` — affects visibility/severity |
| `reported_count` | Community reports — higher = more urgent |
| `score_so_far` | Running average score this episode |
| `feedback` | Ground truth feedback from previous step |

---

## 📋 Tasks

### Task 1: `binary_classify` — Easy
**Goal:** Determine if each piece of content is safe or unsafe. Take `approve` or `remove`.

- 5 steps, 5 items/step
- Only severe violations (severity > 0.7) + clearly safe content
- Score: based on correct safe/unsafe split + confidence calibration
- **Baseline score:** ~0.78

---

### Task 2: `multi_label_classify` — Medium
**Goal:** Identify the *specific* violation type across 6 categories. Choose action appropriate to that category.

- 6 steps, 6 items/step (mixed categories)
- Score: label accuracy (0.7 base) + action alignment (0.15 bonus)
- **Baseline score:** ~0.61

---

### Task 3: `policy_enforcement` — Hard
**Goal:** Full moderation pipeline. Correct label + contextually correct enforcement action, considering:
- Content severity
- Author history (`clean` → `warned` → `repeat_offender`)
- Platform (`dm` vs `comment` vs `post`)
- Edge cases: hyperbole, borderline self-promotion, nuanced speech

Providing a `reason` gives additional score bonus.

- 8 steps, 8 items/step (includes edge cases)
- Score: label + contextual action match + reasoning bonus
- **Baseline score:** ~0.48

---

## 🏆 Reward Function

Rewards are **continuous and shaped** — not binary end-of-episode.

| Situation | Reward |
|-----------|--------|
| Correct label + correct action + good confidence | 0.85–1.0 |
| Correct label + wrong action | 0.25–0.45 |
| Wrong label, still flagged (not safe) | 0.15–0.25 |
| False positive (flagged safe content) | 0.05–0.10 |
| False negative on severe violation | 0.0 (penalized) |

**Key design choices:**
- Missed violent/extreme content gets 0.0 (safety-first)
- Partial credit for catching violations even if category is wrong
- Confidence calibration matters — overconfident wrong answers are penalized
- Reasoning text bonus on hard task encourages explainability

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone or download the repo
cd content-moderation-env/server

# Build Docker image
docker build -t content-moderation-env .

# Run the server
docker run -p 7860:7860 content-moderation-env

# Verify it's running
curl http://localhost:7860/health
```

### Running Inference

```bash
# Install dependencies
pip install openai httpx

# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_token_here"
export ENV_BASE_URL="http://localhost:7860"   # or your HF Space URL

# Run all 3 tasks
export MODERATION_TASK="binary_classify,multi_label_classify,policy_enforcement"
python inference.py
```

### API Quick Reference

```bash
# Reset / start new episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "binary_classify"}'

# Take a step (submit decisions)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '[{"content_id":"abc123","violation_type":"spam","action":"remove","confidence":0.9}]'

# Get current state
curl http://localhost:7860/state

# List all tasks
curl http://localhost:7860/tasks
```

### Validation

```bash
pip install openenv-core
openenv validate
```

---

## 📊 Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Perfect Agent | LLM Baseline | Success Rate |
|------|-----------|--------------|-------------|-------------|
| binary_classify | Easy | 0.987 | ~0.78 | ~94% |
| multi_label_classify | Medium | 1.000 | ~0.65 | ~75% |
| policy_enforcement | Hard | 0.948 | ~0.50 | ~55% |

> **Perfect agent** = oracle with full ground truth access. LLM baseline uses `Qwen2.5-72B-Instruct` zero-shot.

---

## 📁 Project Structure

```
content-moderation-env/
├── inference.py                  # Baseline inference script (required)
├── content_moderation_env.py     # OpenEnv client library
├── openenv.yaml                  # OpenEnv metadata spec
├── README.md                     # This file
└── server/
    ├── main.py                   # FastAPI environment server
    └── Dockerfile                # Container build
```

---

## 🔬 Design Notes

**Why content moderation?**
- Immediately applicable — every major platform needs this
- Rich enough for difficulty progression
- Non-trivial edge cases separate good agents from great ones

**What makes this challenging for frontier models?**
- Edge cases like "I want to kill that movie" vs real threats
- Contextual policy: same text, different action based on author history
- Confidence calibration tested directly in scoring
- Hard task rewards reasoning — pure pattern matching scores lower

**Future extensions:**
- Image + text multimodal content
- Appeal handling (reviewer disagreement)
- Live streaming moderation (time pressure)
- Cross-language content
