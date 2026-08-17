"""
Settings for the rollout and training.

ScenarioStart is how a session opens, DecodeSettings is how the tutor generates, and
SessionSettings is how the session itself is run.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from its.config import MAX_TURNS, TUTOR_ACTIONS
from its.llm import ChatModel, HFChatModel, VLLMChatModel

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
    from trl.generation.vllm_generation import VLLMGeneration

    from its.llm import GenerativeModel
    from its.training.scenarios.base import Scenario


@dataclass
class ScenarioStart:
    """
    Settings for how a scenario starts, which the rollout uses to generate the first
    student message and the tutor's system prompt.
    """
    opening: str | None = None
    solve_terminates: bool = True
    tutor_sees_answer: bool = True
    student_directive: str | None = None
    tutor_system: str | None = None
    show_problem: bool = True
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeSettings:
    """
    Settings for how the tutor generates its messages.
    """
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 256
    action_temperature: float | None = None
    length_norm_actions: bool = False

    def build_tutor(self,
                    model: "GenerativeModel",
                    tokenizer: "PreTrainedTokenizerBase",
                    vllm_generation: "VLLMGeneration | None" = None) -> ChatModel:
        """
        Build a tutor model for this scenario, using the given model and tokenizer.
        """
        kwargs: dict[str, Any] = dict(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            action_temperature=self.action_temperature,
            length_norm_actions=self.length_norm_actions,
        )

        if vllm_generation is not None:
            return VLLMChatModel(vllm_generation=vllm_generation, **kwargs)

        return HFChatModel(**kwargs)


@dataclass(frozen=True)
class SessionSettings:
    """
    Settings for how a session is run, which the rollout uses to determine when to stop
    and how many tools the tutor may call in a turn.
    """
    student_level: str = "medium"
    max_turns: int = MAX_TURNS
    # How many tools the tutor may call in a turn before it has to finish its move.
    max_tool_calls: int = 2
    # The action menu for the tutor, which is scenario-specific.
    ped_actions: Mapping[str, str] = field(default_factory=lambda: TUTOR_ACTIONS)
    # A function that takes the current transcript and returns whether the session should terminate.
    terminate_fn: Callable[[list[dict[str, Any]]], bool] | None = None
    force_tool: str | None = None

    def for_scenario(self, scenario: "Scenario", answer: str, force_tool: str | None) -> "SessionSettings":
        """
        Return a new SessionSettings object with the given scenario's actions and termination function.
        """
        return replace(
            self,
            ped_actions=scenario.actions,
            force_tool=force_tool,
            terminate_fn=lambda tr: scenario.is_complete(tr, answer),
        )
