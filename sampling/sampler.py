"""
Core sampling engine using vLLM.
Generates N independent solution paths for a given problem.
"""

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.answer_utils import extract_final_answer

SYSTEM_PROMPT  = """
You are an expert math olympiad solver.
Solve the problem step by step, showing clear reasoning.

At the end, state your final answer as a single integer on a new line
prefixed exactly with 'Answer:'.

Example 1:
Problem: What is the sum of the first 10 positive integers?

Reasoning:
The sum of the first n positive integers is given by the formula:
n(n + 1) / 2.

For n = 10:
10 × 11 / 2 = 55.

Answer: 55


Example 2:
Problem: If a rectangle has length 12 and width 5, what is its area?

Reasoning:
The area of a rectangle is length × width.
So:
12 × 5 = 60.

Answer: 60


Now solve the following problem step by step.
"""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SolutionPath:
    """One sampled solution path for a problem."""
    problem:    str
    response:   str
    answer:     Optional[str]       # extracted final answer
    tokens:     int = 0             # number of generated tokens
    score:      float = 0.0         # filled in by reranker


@dataclass
class SamplingResult:
    """All N sampled paths for one problem + metadata."""
    problem:    str
    paths:      list[SolutionPath] = field(default_factory=list)
    n_valid:    int = 0             # paths where answer was extracted


# ── Sampler ───────────────────────────────────────────────────────────────────

class SelfConsistencySampler:
    """
    Generates N solution paths per problem using vLLM.

    Args:
        model_path:    Path to fine-tuned model (merged SFT or GRPO checkpoint)
        n_samples:     Number of solution paths to sample per problem
        temperature:   Sampling temperature (0.6–0.8 recommended)
        top_p:         Nucleus sampling cutoff
        max_new_tokens: Max tokens per solution path
        gpu_memory_utilization: vLLM GPU memory fraction
    """

    def __init__(
        self,
        model_path: str,
        n_samples: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 2048,
        gpu_memory_utilization: float = 0.90,
    ):
        self.model_path  = model_path
        self.n_samples   = n_samples
        self.temperature = temperature
        self.top_p       = top_p
        self.max_new_tokens = max_new_tokens

        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
        )

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.sampling_params = SamplingParams(
            n=self.n_samples,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        print(f"Sampler ready — {n_samples} paths per problem")

    def _build_prompt(self, problem: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": problem.strip()},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def sample_one(self, problem: str) -> SamplingResult:
        """Sample N paths for a single problem."""
        prompt  = self._build_prompt(problem)
        outputs = self.llm.generate([prompt], self.sampling_params)
        result  = SamplingResult(problem=problem)

        for completion in outputs[0].outputs:
            text   = completion.text
            answer = extract_final_answer(text)
            path   = SolutionPath(
                problem=problem,
                response=text,
                answer=answer,
                tokens=len(completion.token_ids),
            )
            result.paths.append(path)
            if answer is not None:
                result.n_valid += 1

        return result

    def sample_batch(self, problems: list[str]) -> list[SamplingResult]:
        """
        Sample N paths for each problem in a batch.
        More efficient than calling sample_one in a loop.
        """
        prompts = [self._build_prompt(p) for p in problems]
        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for problem, output in zip(problems, outputs):
            result = SamplingResult(problem=problem)
            for completion in output.outputs:
                text   = completion.text
                answer = extract_final_answer(text)
                path   = SolutionPath(
                    problem=problem,
                    response=text,
                    answer=answer,
                    tokens=len(completion.token_ids),
                )
                result.paths.append(path)
                if answer is not None:
                    result.n_valid += 1
            results.append(result)

        return results