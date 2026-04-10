"""
Content Moderation OpenEnv Client
===================================
Client library that wraps the HTTP API into a clean async interface
matching the OpenEnv standard: reset(), step(), state(), close()
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx


# ─────────────────────────────────────────────────
# Action / Observation / Result Models
# ─────────────────────────────────────────────────

@dataclass
class ContentModerationAction:
    """
    A list of moderation decisions — one per content item in the queue.

    Each decision is a dict with:
        content_id    : str   — ID from the observation queue
        violation_type: str   — one of: hate_speech, harassment, spam,
                                        misinformation, violence, safe
        action        : str   — one of: approve, remove, warn,
                                        escalate, shadow_ban
        confidence    : float — 0.0–1.0
        reason        : str   — optional explanation (boosts score on hard task)
    """
    decisions: List[Dict[str, Any]]


@dataclass
class ContentItem:
    content_id: str
    text: str
    author_id: str
    author_history: str   # "clean" | "warned" | "repeat_offender"
    platform: str         # "comment" | "post" | "dm"
    reported_count: int
    timestamp: float


@dataclass
class ModerationObservation:
    content_queue: List[ContentItem]
    step: int
    max_steps: int
    task: str
    difficulty: str
    score_so_far: float
    feedback: Optional[str] = None


@dataclass
class StepResult:
    observation: ModerationObservation
    reward: float
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────
# Environment Client
# ─────────────────────────────────────────────────

class ContentModerationEnv:
    """
    Async OpenEnv-compatible client for the Content Moderation environment.

    Usage:
        env = await ContentModerationEnv.from_url("http://localhost:7860")
        result = await env.reset(task="binary_classify")
        result = await env.step(action)
        await env.close()
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    async def from_url(cls, base_url: str) -> "ContentModerationEnv":
        env = cls(base_url)
        env._client = httpx.AsyncClient(base_url=base_url, timeout=30.0)
        return env

    @classmethod
    async def from_docker_image(cls, image_name: str, port: int = 7860) -> "ContentModerationEnv":
        """
        Start a Docker container from image_name and connect to it.
        Falls back to localhost if Docker is unavailable.
        """
        import subprocess
        try:
            container_id = subprocess.check_output([
                "docker", "run", "-d", "-p", f"{port}:{port}", image_name
            ]).decode().strip()
            # Wait for server to be ready
            await asyncio.sleep(3)
            env = await cls.from_url(f"http://localhost:{port}")
            env._container_id = container_id
            return env
        except Exception:
            # Fallback to env var URL
            url = os.getenv("ENV_BASE_URL", f"http://localhost:{port}")
            return await cls.from_url(url)

    async def reset(self, task: str = "binary_classify", seed: Optional[int] = None) -> StepResult:
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed

        r = await self._client.post("/reset", json=payload)
        r.raise_for_status()
        data = r.json()

        obs = self._parse_observation(data)
        return StepResult(observation=obs, reward=0.0, done=False, info={})

    async def step(self, action: ContentModerationAction) -> StepResult:
        r = await self._client.post("/step", json=action.decisions)
        r.raise_for_status()
        data = r.json()

        obs = self._parse_observation(data["observation"])
        return StepResult(
            observation=obs,
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )

    async def state(self) -> Dict[str, Any]:
        r = await self._client.get("/state")
        r.raise_for_status()
        return r.json()

    async def close(self):
        if self._client:
            await self._client.aclose()
        if hasattr(self, "_container_id"):
            try:
                import subprocess
                subprocess.run(["docker", "stop", self._container_id], capture_output=True)
                subprocess.run(["docker", "rm", self._container_id], capture_output=True)
            except Exception:
                pass

    def _parse_observation(self, data: dict) -> ModerationObservation:
        queue = [
            ContentItem(
                content_id=item["content_id"],
                text=item["text"],
                author_id=item["author_id"],
                author_history=item["author_history"],
                platform=item["platform"],
                reported_count=item["reported_count"],
                timestamp=item["timestamp"],
            )
            for item in data.get("content_queue", [])
        ]
        return ModerationObservation(
            content_queue=queue,
            step=data["step"],
            max_steps=data["max_steps"],
            task=data["task"],
            difficulty=data["difficulty"],
            score_so_far=data["score_so_far"],
            feedback=data.get("feedback"),
        )
