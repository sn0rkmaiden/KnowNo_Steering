from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

import re
from functools import partial

def _steering_hook(activations, hook, steering_vector, steering_strength: float, max_act: float):
    # activations: [batch, seq, d_model]
    return activations + max_act * steering_strength * steering_vector

@torch.no_grad()
def generate_with_steering(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    steering_feature: int,
    max_act: float,
    steering_strength: float,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    steering_vector = sae.W_dec[steering_feature].to(model.cfg.device)
    steering_hook = partial(
        _steering_hook,
        steering_vector=steering_vector,
        steering_strength=float(steering_strength),
        max_act=float(max_act),
    )
    prepend_bos = bool(getattr(sae.cfg.metadata, "prepend_bos", False))

    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    with model.hooks(fwd_hooks=[(sae.cfg.metadata.hook_name, steering_hook)]):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_at_eos=False if model.cfg.device == "mps" else True,
            prepend_bos=prepend_bos,
        )

    # decode to text
    return model.tokenizer.decode(out[0])

@dataclass
class SteeringConfig:
    enabled: bool = False
    sae_release: Optional[str] = None
    sae_id: Optional[str] = None
    feature: Optional[int] = None
    strength: float = 0.0
    max_act: Optional[float] = None
    compute_max_per_prompt: bool = False

class GemmaHookedModel:
    """
    Minimal wrapper around TransformerLens HookedTransformer with optional SAE steering.
    Uses a plain-text chat format:
      User: ...
      Assistant:
    """
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        steering: Optional[SteeringConfig] = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = int(max_new_tokens)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype, torch.bfloat16)

        self.model = HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
            dtype=self.dtype,
        )

        self.steering = steering or SteeringConfig(enabled=False)
        self.sae: Optional[SAE] = None
        if self.steering.enabled:
            if self.steering.sae_release is None or self.steering.sae_id is None or self.steering.feature is None:
                raise ValueError("Steering enabled but sae_release/sae_id/feature not provided.")
            self.sae = SAE.from_pretrained(
                release=self.steering.sae_release,
                sae_id=self.steering.sae_id,
                device=self.device,
            )

    def _build_text_from_messages(self, messages: List[Dict[str, str]], force_json: bool) -> str:
        parts: List[str] = []
        for m in messages:
            role = (m.get("role") or "").strip().lower()
            content = (m.get("content") or "").strip()
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"{role.capitalize()}: {content}")
        parts.append("Assistant:")
        text = "\n".join(parts)
        if force_json:
            text += "\nReturn **only** a valid JSON object without any extra text."
        return text

    def _extract_assistant_answer(self, decoded: str, prompt_text: str) -> str:
        # remove prompt echo if present
        if decoded.startswith(prompt_text):
            decoded = decoded[len(prompt_text):]
        # heuristics: take content after last 'Assistant:' marker
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1]
        return decoded.strip()

    @torch.no_grad()
    def _estimate_feature_max_for_prompt(self, prompt_text: str, feature_idx: int) -> float:
        if self.sae is None:
            return 1.0
        hook_name = self.sae.cfg.metadata.hook_name
        prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))
        toks = self.model.to_tokens(prompt_text, prepend_bos=prepend_bos)
        _, cache = self.model.run_with_cache(toks, names_filter=[hook_name])
        sae_in = cache[hook_name]
        feats = self.sae.encode(sae_in)
        # feats: [batch, seq, n_features]
        return float(feats[..., feature_idx].max().item())

    def request(
        self,
        prompt: str,
        previous_messages: Optional[List[Dict[str, str]]] = None,
        json_format: bool = False,
    ) -> Tuple[str, List[Dict[str, str]]]:
        messages = (list(previous_messages) if previous_messages else []) + [{"role": "user", "content": prompt}]
        prompt_text = self._build_text_from_messages(messages, force_json=bool(json_format))

        if self.steering.enabled:
            assert self.sae is not None
            turn_max = self.steering.max_act
            if self.steering.compute_max_per_prompt or turn_max is None:
                turn_max = self._estimate_feature_max_for_prompt(prompt_text, int(self.steering.feature))
                # avoid tiny/zero
                turn_max = max(1e-6, float(turn_max))

            decoded = generate_with_steering(
                model=self.model,
                sae=self.sae,
                prompt=prompt_text,
                steering_feature=int(self.steering.feature),
                max_act=float(turn_max),
                steering_strength=float(self.steering.strength),
                max_new_tokens=self.max_new_tokens,
            )
        else:
            prepend_bos = False
            toks = self.model.to_tokens(prompt_text, prepend_bos=prepend_bos)
            out = self.model.generate(
                toks,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=False if self.model.cfg.device == "mps" else True,
                prepend_bos=prepend_bos,
            )
            decoded = self.model.tokenizer.decode(out[0])

        assistant = self._extract_assistant_answer(decoded, prompt_text)
        messages = messages + [{"role": "assistant", "content": assistant}]
        return assistant, messages
