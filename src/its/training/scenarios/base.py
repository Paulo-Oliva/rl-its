"""
What a scenario is, meaning how a session starts, how it is scored and when it is done.
"""

import random
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from its.config import TUTOR_ACTIONS
from its.training.reward import Judge, RewardBreakdown
from its.training.rollout import Session
from its.training.settings import ScenarioStart


class Scenario(ABC):
    name: str
    # The tool whose call may be forced to break the cold start, where None means none is
    force_tool: str | None = None
    # The tutor's action vocabulary for this scenario
    actions: Mapping[str, str] = TUTOR_ACTIONS
    # The tag that is this scenario's job, which the rules reward directly
    signature_action: str | None = None

    @abstractmethod
    def start(self, problem: str, answer: str, difficulty: str, rng: random.Random) -> ScenarioStart:
        ...

    @abstractmethod
    def score(
        self,
        session: Session,
        meta: dict[str, Any],
        judge: Judge | None,
        weights: dict[str, float] | None = None,
        student_level: str = "",
        difficulty: str = "",
    ) -> RewardBreakdown:
        ...

    def is_complete(self, transcript: list[dict[str, Any]], answer: str) -> bool:
        """
        Whether the goal has been reached, checked after each student reply.

        Args:
            transcript (list[dict[str, Any]]): The session so far, in order.
            answer (str): The gold answer.

        Returns:
            bool: True to end the session here. Returns False by default.
        """
        return False
