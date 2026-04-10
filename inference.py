"""
Inference Script — Content Moderation OpenEnv
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_BASE_URL   The base URL of your deployed HF Space (e.g. https://your-space.hf.space)
    IMAGE_NAME     (optional) Docker image name if running locally

- Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
    [START] task=<task_name> env=content-moderation model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from content_moderation_env import ContentModerationAction, ContentModerationEnv, StepResult

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf-dummy")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASK_NAME = os.getenv("MODERATION_TASK", "binary_classify")
BENCHMARK = "content-moderation"
MAX_STEPS = 8
TEMPERATURE = 0.2   # Low temperature for consistent moderation decisions
MAX_TOKENS = 1000
SUCCESS_SCORE_THRESHOLD = 0.6

# ─────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ─────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert content moderation AI for a major social media platform.
Your job is to review content items and make accurate moderation decisions.

VIOLATION TYPES:
- hate_speech: Content targeting people based on race, religion, gender, nationality, etc.
- harassment: Direct threats, bullying, doxxing, or targeted attacks on individuals
- spam: Unsolicited commercial content, scams, fake giveaways, bot-like behavior
- misinformation: Demonstrably false health/political/scientific claims presented as fact
- violence: Explicit threats of physical harm, instructions for violence/weapons
- safe: Content that does not violate any policies

ACTIONS:
- approve: No violation found, content is safe
- remove: Clear, severe violation — immediately remove
- warn: Borderline or first-offense violation — issue a warning
- escalate: Extremely severe (violence, CSAM risk) — send to senior reviewer
- shadow_ban: Repeat offender or spam — hide without notifying user

CONTEXT CLUES:
- author_history: "clean" (no history), "warned" (has prior warning), "repeat_offender" (multiple violations)
- platform: "dm" (private, less visible), "comment" (semi-public), "post" (fully public)
- reported_count: higher = more community concern

RULES:
- Prioritize user safety — missing a real violation is worse than a false positive
- Consider author history when choosing action severity
- Never escalate safe content
- Repeat offenders get stricter actions for the same violation level

Respond ONLY with a valid JSON array of decisions. No markdown, no explanation outside the JSON.
Format:
[
  {
    "content_id": "<id>",
    "violation_type": "<type>",
    "action": "<action>",
    "confidence": <0.0-1.0>,
    "reason": "<brief explanation>"
  }
]
""").strip()


def build_user_prompt(step: int, result: StepResult, history: List[str]) -> str:
    obs = result.observation
    items_text = ""
    for item in obs.content_queue:
        items_text += textwrap.dedent(f"""
        ---
        content_id: {item.content_id}
        text: "{item.text}"
        author_history: {item.author_history}
        platform: {item.platform}
        reported_count: {item.reported_count}
        """).strip() + "\n\n"

    history_block = "\n".join(history[-3:]) if history else "None yet."

    return textwrap.dedent(f"""
    STEP {step} / {obs.max_steps}  |  Task: {obs.task} ({obs.difficulty})  |  Score so far: {obs.score_so_far:.2f}

    {f"Previous feedback: {obs.feedback}" if obs.feedback else ""}

    CONTENT QUEUE ({len(obs.content_queue)} items to moderate):
    {items_text}
    Recent history:
    {history_block}

    Respond with a JSON array of {len(obs.content_queue)} decisions — one per content_id above.
    """).strip()

# ─────────────────────────────────────────────────
# Model call
# ─────────────────────────────────────────────────

def get_model_decisions(
    client: OpenAI,
    step: int,
    result: StepResult,
    history: List[str],
) -> List[Dict[str, Any]]:
    user_prompt = build_user_prompt(step, result, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        decisions = json.loads(raw)
        if isinstance(decisions, list):
            return decisions
        return []

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback: approve everything with low confidence
        return [
            {
                "content_id": item.content_id,
                "violation_type": "safe",
                "action": "approve",
                "confidence": 0.3,
                "reason": "fallback decision",
            }
            for item in result.observation.content_queue
        ]

# ─────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────

async def run_task(client: OpenAI, env: ContentModerationEnv, task: str) -> tuple[bool, int, float, List[float]]:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[str] = []

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)

        for step in range(1, MAX_STEPS + 1):
            if result.done or not result.observation.content_queue:
                break

            decisions = get_model_decisions(client, step, result, history)
            action = ContentModerationAction(decisions=decisions)

            error_msg = None
            try:
                result = await env.step(action)
                reward = result.reward
                done = result.done
            except Exception as exc:
                error_msg = str(exc)[:80]
                reward = 0.0
                done = True

            rewards.append(reward)
            steps_taken = step

            # Compact action summary for logging
            action_summary = f"moderated_{len(decisions)}_items"
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error_msg)

            history.append(
                f"Step {step}: processed {len(decisions)} items, reward={reward:.2f}"
            )

            if done or error_msg:
                break

        score = sum(rewards) / max(len(rewards), 1)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        pass  # env closed by caller

    return success, steps_taken, score, rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks_to_run = os.getenv("MODERATION_TASK", "binary_classify,multi_label_classify,policy_enforcement").split(",")

    for task in tasks_to_run:
        task = task.strip()

        if IMAGE_NAME:
            env = await ContentModerationEnv.from_docker_image(IMAGE_NAME)
        else:
            env = await ContentModerationEnv.from_url(ENV_BASE_URL)

        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        try:
            success, steps_taken, score, rewards = await run_task(client, env, task)
        finally:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
