# agent_utils/inference_vllm_gemma3.py
"""
Optional vLLM-based inference for Gemma-3 (simple_gemma3) for maximum throughput.

Use this for large-scale annotation (e.g. 10k+ posts) when you want:
  - Batched inference (many prompts per GPU forward)
  - Built-in prefix caching (codebook processed once)
  - Faster decode kernels than HuggingFace generate()

Requirements:
  - pip install vllm
  - A saved model: either merged (base + PEFT adapter) or base path + adapter path
    (vLLM supports some LoRA loading; see vLLM docs).

Usage:
  - From Python: call run_inference_vllm(prompts, model_path, system_prompt, ...)
  - Or run this file as a script with argparse (see __main__).

Prompt format must match training: same system_prompt and user template with "{}"
filled by the post text. We use the same _build_chat_text_simple logic so outputs
are comparable to run_simple_val_inference().
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Optional

# Optional dependency
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


def _build_prompt_for_vllm(system_prompt: Optional[str], user_instruction: str) -> str:
    """
    Build a single prompt string for vLLM. vLLM will apply the model's chat
    template if we use the chat API, but for generate() we pass raw text.
    Gemma 3 uses <start_of_turn>user ... <end_of_turn> etc.
    We mirror the tokenizer's apply_chat_template output so prefix caching
    matches: system content folded into first user turn for Gemma 3.
    """
    if system_prompt:
        # Gemma 3 style: system is often folded into first user turn
        content = f"{system_prompt.strip()}\n\n{user_instruction}"
    else:
        content = user_instruction
    # Minimal Gemma 3 chat format (no BOS here; vLLM may add it)
    return f"<start_of_turn>user\n{content}\n<end_of_turn>\n<start_of_turn>model\n"


def run_inference_vllm(
    prompts: List[str],
    model_path: str,
    system_prompt: Optional[str] = None,
    max_tokens: int = 400,
    temperature: float = 0.0,
    batch_size: int = 8,
    max_model_len: int = 4096,
    enable_prefix_caching: bool = True,
    dtype: str = "bfloat16",
) -> List[str]:
    """
    Run batched inference with vLLM. prompts should be the full prompt strings
    (system + user content already formatted, or pass user-only and we prepend system).

    If each prompt is "user instruction only", set system_prompt and we build
    full prompts with _build_prompt_for_vllm(system_prompt, user_instruction).
    If each prompt is already the full string (system + user), set system_prompt=None.

    Returns list of generated text (one per prompt).
    """
    if not VLLM_AVAILABLE:
        raise RuntimeError(
            "vLLM is not installed. Install with: pip install vllm\n"
            "Then use this module for fast batched inference."
        )
    if system_prompt is not None:
        full_prompts = [_build_prompt_for_vllm(system_prompt, p) for p in prompts]
    else:
        full_prompts = prompts
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        dtype=dtype,
        enable_prefix_caching=enable_prefix_caching,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
    )
    outputs = llm.generate(full_prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="vLLM inference for Gemma-3 (simple_gemma3)")
    parser.add_argument("model_path", help="Path to merged model or base model")
    parser.add_argument("--prompts-file", help="Text file: one prompt per line (user instruction only)")
    parser.add_argument("--system-prompt-file", help="Text file with system/codebook (optional)")
    parser.add_argument("--output-file", default="vllm_outputs.jsonl", help="Output JSONL path")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    system_prompt = None
    if args.system_prompt_file and os.path.isfile(args.system_prompt_file):
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    with open(args.prompts_file, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    texts = run_inference_vllm(
        prompts,
        args.model_path,
        system_prompt=system_prompt,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
    )
    with open(args.output_file, "w", encoding="utf-8") as f:
        for p, t in zip(prompts, texts):
            f.write(json.dumps({"prompt": p[:200], "generated": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts)} outputs to {args.output_file}")


if __name__ == "__main__":
    main()
