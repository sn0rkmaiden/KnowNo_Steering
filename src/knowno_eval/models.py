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
    desired_prepend_bos = bool(getattr(sae.cfg.metadata, "prepend_bos", False))

    # NOTE: we decode only the newly generated tokens (not the prompt echo).
    # Also: avoid double-BOS when the prompt (e.g. from a chat template) already includes BOS.
    bos_id = getattr(model.tokenizer, "bos_token_id", None)
    if desired_prepend_bos and bos_id is not None:
        toks_no_bos = model.to_tokens(prompt, prepend_bos=False)
        if toks_no_bos.numel() > 0 and int(toks_no_bos[0, 0].item()) == int(bos_id):
            prepend_bos = False
            input_ids = toks_no_bos
        else:
            prepend_bos = True
            input_ids = model.to_tokens(prompt, prepend_bos=True)
    else:
        prepend_bos = False
        input_ids = model.to_tokens(prompt, prepend_bos=False)
    with model.hooks(fwd_hooks=[(sae.cfg.metadata.hook_name, steering_hook)]):
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_at_eos=False if model.cfg.device == "mps" else True,
            prepend_bos=prepend_bos,
        )

    gen = out[0, input_ids.shape[1]:]
    return model.tokenizer.decode(gen.tolist(), skip_special_tokens=True).strip()

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
        """Render a chat prompt.

        For Gemma Instruct models, using the tokenizer's chat template is critical.
        If we fall back to a plain-text format, some models will immediately emit <eos>.
        """
        tok = getattr(self.model, "tokenizer", None)

        # Put the JSON constraint into a system message so it is strongly applied
        if force_json:
            json_sys = {
                "role": "system",
                "content": "Return ONLY a valid JSON object and nothing else.",
            }
            # keep any existing system message first
            if messages and (messages[0].get("role") or "").strip().lower() == "system":
                messages = messages[:1] + [json_sys] + messages[1:]
            else:
                messages = [json_sys] + messages

        # Prefer HF chat template when available
        if tok is not None and hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Fallback: plain text (kept for non-chat models)
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
        return "\n".join(parts)

    def _extract_assistant_answer(self, decoded: str, prompt_text: str) -> str:
        # Kept for backward compatibility with the fallback prompt path.
        if decoded.startswith(prompt_text):
            decoded = decoded[len(prompt_text):]
        if "Assistant:" in decoded:
            decoded = decoded.split("Assistant:")[-1]
        return decoded.strip()

    def _choose_prepend_bos(self, prompt_text: str, desired: bool) -> bool:
        """Avoid double-BOS when the chat template already inserted it."""
        if not desired:
            return False
        bos_id = getattr(self.model.tokenizer, "bos_token_id", None)
        if bos_id is None:
            return bool(desired)
        toks = self.model.to_tokens(prompt_text, prepend_bos=False)
        if toks.numel() > 0 and int(toks[0, 0].item()) == int(bos_id):
            return False
        return True

    @torch.no_grad()
    def _estimate_feature_max_for_prompt(self, prompt_text: str, feature_idx: int) -> float:
        if self.sae is None:
            return 1.0
        hook_name = self.sae.cfg.metadata.hook_name
        prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))
        prepend_bos = self._choose_prepend_bos(prompt_text, prepend_bos)
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

            prepend_bos = bool(getattr(self.sae.cfg.metadata, "prepend_bos", False))
            prepend_bos = self._choose_prepend_bos(prompt_text, prepend_bos)

            decoded = generate_with_steering(
                model=self.model,
                sae=self.sae,
                prompt=prompt_text,
                steering_feature=int(self.steering.feature),
                max_act=float(turn_max),
                steering_strength=float(self.steering.strength),
                max_new_tokens=self.max_new_tokens,
                # generation params are set inside generate_with_steering
            )
        else:
            # Decode only newly generated tokens to avoid prompt-echo / <eos> artifacts.
            toks = self.model.to_tokens(prompt_text, prepend_bos=False)
            out = self.model.generate(
                toks,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=False if self.model.cfg.device == "mps" else True,
                prepend_bos=False,
            )
            gen = out[0, toks.shape[1]:]
            decoded = self.model.tokenizer.decode(gen.tolist(), skip_special_tokens=True).strip()

        assistant = decoded if decoded else ""
        messages = messages + [{"role": "assistant", "content": assistant}]
        return assistant, messages
