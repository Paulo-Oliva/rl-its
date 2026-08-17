"""
The chat models the tutor and student are driven through.

ChatModel is the abstract base class, and HFChatModel and VLLMChatModel are concrete implementations.
They share the same interface, so the rollout code can be agnostic to which one is being used.
The rollout code only needs to know that it can hand a message list to the model and get text back,
and that it can set the decoding parameters (max_new_tokens, temperature, top_p, do_sample) once for the whole rollout.

VLLMChatModel is a wrapper around TRL's vLLM engine. It generates on the engine, but keeps all action scoring and
logprob recomputation on the HF model. This avoids any train/inference mismatch, since vLLM only ever produces tokens,
never probabilities.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizerBase
    from transformers.generation.utils import GenerateOutput
    from trl.generation.vllm_generation import VLLMGeneration

log = logging.getLogger(__name__)


def apply_template(tokenizer: "PreTrainedTokenizerBase",
                   messages: list[dict[str, Any]],
                   add_generation_prompt: bool,
                   tools: list[dict[str, Any]] | None = None) -> list[int]:
    """
    Tokenize a message list with the chat template, returning a flat id list.

    Args:
        tokenizer (PreTrainedTokenizerBase): HuggingFace tokenizer whose chat template is applied.
        messages (list[dict[str, Any]]): The conversation, each entry a {"role", "content"} dict.
        add_generation_prompt (bool): Append the assistant header, so the model starts a new reply instead of
            continuing the last message.
        tools (list[dict[str, Any]] | None, optional): Tool schemas to advertise to the model.
            Defaults to None, which leaves the prompt untouched.

    Returns:
        list[int]: The token ids. Some tokenizer versions return a dict here, so the
            ids are unwrapped before returning.
    """
    result: Any = tokenizer.apply_chat_template(messages,
                                                add_generation_prompt=add_generation_prompt,
                                                tokenize=True,
                                                tools=tools)  # type: ignore
    return result if isinstance(result, list) else result["input_ids"]


# For type-checking the HF model
class GenerativeModel(Protocol):

    @property
    def device(self) -> "torch.device":
        ...

    def generate(self, **kwargs: Any) -> "GenerateOutput | torch.Tensor":
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass
class ChatModel(ABC):
    """
    Messages in, text out - the interface the rollout code is written against.

    The decoding parameters are fields rather than call arguments, so a rollout sets them
    once at construction and every later turn inherits them. `model` and `tokenizer` are
    the HF pair even on the vLLM backend, since scoring always needs the HF forward pass.
    """

    model: GenerativeModel
    tokenizer: "PreTrainedTokenizerBase"
    max_new_tokens: int
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    action_temperature: float | None = None
    length_norm_actions: bool = False

    @abstractmethod
    def __call__(self, messages: list[dict[str, Any]]) -> str:
        pass

    def template_ids(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> list[int]:
        return apply_template(self.tokenizer, messages, add_generation_prompt=True, tools=tools)


@dataclass
class HFChatModel(ChatModel):
    """
    Generates locally with HuggingFace `generate`, one call per tutor or student turn.
    """

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        import torch

        # Apply the template and move the inputs to the model device
        inputs: Any = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        # Override the generation config with the rollout's decoding parameters
        gen_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            gen_kwargs.update(temperature=self.temperature, top_p=self.top_p)
        else:
            # Qwen checkpoints ship sampling defaults in generation_config.json, which the model will use if we don't
            # override them and we get a warning. Setting to None to silence the warning, since we don't need them.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        # A dict-like comes back only when return_dict_in_generate is asked for, which it is not
        assert isinstance(output_ids, torch.Tensor)
        return self.tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()


@dataclass
class VLLMChatModel(ChatModel):
    """
    Generates on TRL's vLLM engine.

    Only the text generation moves. `model` still holds the HF policy, because action
    scoring and logprob recomputation need probabilities and the engine only hands back
    tokens. TRL merges the LoRA adapter into the engine at every step, so the weights
    generating and the weights being scored are the same ones.
    """

    # The vLLMGeneration object is created by the trainer and passed to the model
    vllm_generation: "VLLMGeneration | None" = None
    # A list of strings that, if generated, will cause the generation to stop
    stop: list[str] | None = None

    def generate_ids_batch(self, prompts_ids: list[list[int]]) -> list[list[int]]:
        """
        Generate one completion per prompt in a single vLLM call.

        Args:
            prompts_ids (list[list[int]]): One templated prompt per session, as token ids.

        Returns:
            list[list[int]]: The generated ids per prompt, in the order they were given.
                Each one stops early if it hits a string in `self.stop`.
        """
        from vllm import SamplingParams

        temp = self.temperature if self.do_sample else 0.0
        vg = self.vllm_generation
        if vg is None:
            raise RuntimeError("VLLMChatModel needs a vllm_generation (trainer.vllm_generation)")
        if vg.mode != "colocate":
            raise RuntimeError(f"Only the colocated vLLM engine is supported, got mode {vg.mode!r}")

        params = SamplingParams(temperature=temp,
                                top_p=self.top_p,
                                max_tokens=self.max_new_tokens,
                                stop=self.stop,
                                include_stop_str_in_output=True)
        # Hand over token ids because the prompt is already tokenized, and letting vLLM do it
        # again could give a different one
        outs = vg.llm.generate([{"prompt_token_ids": p} for p in prompts_ids], sampling_params=params, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]

    def generate_ids(self, prompt_ids: list[int]) -> list[int]:
        """
        Generate one completion for a single prompt, the sequential rollout's path.

        Args:
            prompt_ids (list[int]): The templated prompt as token ids.

        Returns:
            list[int]: The generated token ids.
        """
        return self.generate_ids_batch([prompt_ids])[0]

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        prompt_ids = apply_template(self.tokenizer, messages, add_generation_prompt=True)
        ids = self.generate_ids(prompt_ids)
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


def load_model(model_id: str, device: str, max_new_tokens: int = 200) -> HFChatModel:
    """
    Load a tokenizer and model from the Hub or a local path, ready to chat with.

    Args:
        model_id (str): Hub repo ID (e.g. "Qwen/Qwen2.5-Math-7B-Instruct") or a local
            checkpoint directory.
        device (str): Where to place the weights, passed to `device_map` (e.g. "cuda", "cpu").
        max_new_tokens (int, optional): Generation cap for every turn. Defaults to 200.

    Returns:
        HFChatModel: The loaded model, decoding greedily until `do_sample` is set on it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device)
    return HFChatModel(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
