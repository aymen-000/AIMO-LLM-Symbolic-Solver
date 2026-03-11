#  AIMO-LLM-Symbolic-Solver (in dev progress)

A full end-to-end system for solving competition math problems (AIME / AMC / Olympiad style).
Built for the [AIMO Kaggle competition](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize).

**Stack:** Qwen2.5-Math-7B · SFT + GRPO · Self-Consistency Sampling · Tool Execution · Multi-Verifier · Last-Resort Waterfall

---

## Architecture

```
                        ┌─────────────────────┐
                        │   Math Problem      │
                        └────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Fine-tuned LLM        │
                    │   Qwen2.5-Math-7B       │
                    │   SFT → GRPO            │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Self-Consistency       │
                    │  Sampling  (N=32)       │
                    │  temp=0.7, top_p=0.95   │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Code Extraction        │
                    │  + Sandbox Execution    │
                    │  (per path)             │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │   Symbolic   │  │   Numeric    │  │  Theorem     │
      │   Verifier   │  │   Verifier   │  │  Provers     │
      │   (SymPy)    │  │   (numpy)    │  │  (Z3 / Lean) │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             └──────────────┬──┘──────────────────┘
                            │
                ┌───────────▼───────────┐
                │   Verdict Aggregator  │
                │   PASS / FAIL /       │
                │   UNCERTAIN           │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Voting              │
                │   majority | weighted │
                │   best_of_n           │
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Post-vote Verify    │
                │   + Fallback cascade  │
                │   + Last-resort (S1-S6)│
                └───────────┬───────────┘
                            │
                ┌───────────▼───────────┐
                │   Final Answer        │
                └───────────────────────┘
```

---

## Repository Structure

```
aimo/
│
├── inference.py                    ← MAIN ENTRY POINT — full inference pipeline
│
├── configs/
│   ├── sft_config.yaml             ← SFT hyperparameters
│   ├── grpo_config.yaml            ← GRPO hyperparameters
│   └── sampling_config.yaml        ← Sampling & voting config
│
├── data/
│   ├── download_datasets.py        ← Pull NuminaMath, MATH, AIME from HuggingFace
│   ├── filter_and_clean.py         ← Quality filter, dedup, format into chat template
│   └── build_grpo_dataset.py       ← Select calibrated-difficulty problems for GRPO
│
├── sft/
│   ├── train_sft.py                ← LoRA SFT training (TRL SFTTrainer)
│   └── merge_lora.py               ← Merge LoRA adapter into full model checkpoint
│
├── grpo/
│   ├── reward.py                   ← Reward function (binary + optional partial)
│   └── train_grpo.py               ← GRPO training loop (TRL GRPOTrainer)
│
├── sampling/
│   ├── sampler.py                  ← vLLM-based self-consistency sampler
│   ├── voting.py                   ← MajorityVoter, WeightedVoter, BestOfNVoter
│   ├── scorers.py                  ← Heuristic path scoring (no ORM needed)
│   └── pipeline.py                 ← Standalone sampling CLI
│
├── tools/
│   ├── extractor.py                ← Extract code blocks + LaTeX from CoT text
│   ├── executor.py                 ← Sandboxed Python execution (subprocess + AST check)
│   ├── router.py                   ← Classify problem type → assign verifiers
│   ├── aggregator.py               ← Combine verifier verdicts with confidence weights
│   ├── pipeline.py                 ← Standalone verification CLI
│   ├── verifiers/
│   │   ├── symbolic.py             ← SymPy: equation substitution, simplification
│   │   ├── numeric.py              ← Execute code, compare output to claimed answer
│   │   ├── combinatorial.py        ← Brute-force integer/modular/GCD/digit checks
│   │   └── geometric.py            ← Pythagorean, Heron, circle, distance formula
│   └── provers/
│       ├── z3_prover.py            ← Z3 SMT: integer constraints, modular arithmetic
│       └── lean_prover.py          ← Lean 4: formal proof verification (optional)
│
├── utils/
│   ├── answer_utils.py             ← Answer extraction, normalization, SymPy verify
│   └── vram_utils.py               ← VRAM monitoring, VRAMGuard context manager
│
└── requirements.txt
└── README.md
```

---

## Hardware Requirements

All training and inference designed for **1x H100 (80 GB SXM)**.

| Stage | VRAM used | Time estimate |
|---|---|---|
| SFT (LoRA r64, batch 32) | ~65 GB | ~8 h on 300K samples |
| LoRA merge | ~30 GB | ~5 min |
| GRPO (G=8, ref in 8-bit) | ~75 GB | ~3 h on 1500 steps |
| Inference (N=32, vLLM) | ~70 GB | ~90 s per problem |

---

## Setup

```bash
git clone 
cd AIMO-LLM-Symbolic-Solve
pip install -r requirements.txt

pip install z3-solver

curl https://elan.lean-lang.org/install.sh | sh
```

---

## Training

### Step 1 — Download and clean data

```bash
python data/download_datasets.py
python data/filter_and_clean.py
```

Downloads NuminaMath-CoT (~860K), MATH (12.5K), AIME (~900), and NuminaMath-TIR.
After filtering and deduplication, caps at 300K samples for single-GPU training.
Outputs `data/processed/sft_train.jsonl` and `data/processed/sft_eval.jsonl`.

### Step 2 — SFT

```bash
python sft/train_sft.py
```

Fine-tunes `Qwen2.5-Math-7B-Instruct` with LoRA (rank 64) for 2 epochs.
Computes loss on assistant tokens only. Logs to Weights and Biases.
Checkpoint saved to `outputs/sft/final/`.

### Step 3 — Merge LoRA

```bash
python sft/merge_lora.py
```

Merges the LoRA adapter into the base model weights.
Output: `outputs/sft_merged/` — a standard HuggingFace model directory ready for GRPO.

### Step 4 — Build GRPO dataset

```bash
python data/build_grpo_dataset.py
```

Runs the SFT model on AIME + MATH Level 4/5 problems with N=8 rollouts each.
Keeps only problems where solve rate is **20%–80%**. Too easy means no gradient.
Too hard means no positive reward signal.
Output: `data/processed/grpo_hard.jsonl`.

### Step 5 — GRPO

```bash
python grpo/train_grpo.py
```

GRPO with group size G=8. Reward = 1.0 for correct answer, 0.0 for wrong.
Reference model is loaded in 8-bit to save VRAM.
Final checkpoint saved to `outputs/grpo/final/`.

Key GRPO settings:

| Parameter | Value | Reason |
|---|---|---|
| Group size G | 8 | 8 rollouts per problem per step |
| KL coefficient | 0.02 | Prevents policy from drifting too far from SFT |
| Learning rate | 1e-6 | Much lower than SFT |
| Max steps | 1500 | Converges fast on math |
| Reference model | 8-bit | Saves ~8 GB VRAM |

---


## Inference

### Quick start

```bash
# Single problem — prints full breakdown to stdout
python inference.py \
  --model ./outputs/grpo/final \
  --problem "Find the sum of all positive integers n < 1000 where n squared is congruent to 1 mod 7"

# Batch from file
python inference.py \
  --model ./outputs/grpo/final \
  --input  problems.jsonl \
  --output predictions.jsonl \
  --n 32 --strategy majority

```

### Inference pipeline stages

```
Stage 1     Sample N paths              vLLM, temp=0.7
Stage 1.5   Execute code blocks         Sandbox per path
            If code answer != text answer → override with code answer
Stage 2     Verify each path            SymPy / numeric / Z3 / combinatorial / geometric
            PASS  → boost path score +2.0
            FAIL  → penalize path score -1.5
Stage 3     Vote                        majority | weighted | best_of_n
Stage 4     Re-verify winner            Full verifier stack on chosen answer
Stage 5     Last-resort waterfall       Fires if winner still FAILs:
              S1 Sweep all candidates   Try every unique answer by vote rank
              S2 Best UNCERTAIN         Not disproved beats actively wrong
              S3 Code-only answers      Pure arithmetic, ignore text reasoning
              S4 Highest-score path     Best heuristic score regardless of verdict
              S5 Greedy resample        temp=0 single deterministic generation + verify
              S6 Abstain                Return None (scores 0, not wrong)
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `./outputs/grpo/final` | Model checkpoint path |
| `--n` | 32 | Paths sampled per problem |
| `--strategy` | `majority` | majority / weighted / best_of_n |
| `--temperature` | 0.7 | Sampling temperature |
| `--verify` | on | Run tool verification |
| `--no_verify` | off | Disable all verification |
| `--lean` | off | Enable Lean 4 prover (slow) |
| `--max_verifiers` | 4 | Max verifiers per problem |
| `--no_fallback` | off | Disable last-resort waterfall |
| `--gpu_util` | 0.88 | vLLM GPU memory fraction |

---

## Verification System

The verifier stack is routed by problem type. Not every verifier runs on every problem.

| Problem type | Primary verifiers |
|---|---|
| Algebraic | SymPy symbolic → numeric executor |
| Combinatorial | Brute-force enumeration → Z3 |
| Geometric | Heron / Pythagorean / distance → numeric |
| Number theory | Z3 SMT → combinatorial → SymPy |
| Has Python code | Numeric executor always runs first |

Verdict weights — higher means more trusted. One FAIL from weight >= 0.65 triggers fallback.

| Verifier / method | Weight |
|---|---|
| Lean 4 formal proof | 1.00 |
| Z3 SMT solver | 0.95 |
| GCD / LCM direct computation | 0.90 |
| Integer brute-force enumeration | 0.85 |
| SymPy equation substitution | 0.85 |
| Code execution match | 0.70 |
| Heron formula / Pythagorean | 0.70 |
| Numeric expression evaluation | 0.60 |

---

## Code Execution Sandbox

Enforced per execution:

- Subprocess isolation — code runs in a separate process, not via `eval`
- Static AST scan before execution — blocks `os`, `sys`, `subprocess`, `open`, `__import__`, `eval`, `exec`
- Allowed imports whitelist: `sympy`, `numpy`, `math`, `fractions`, `itertools`, `collections`, `re`, `decimal`, `scipy`
- 512 MB memory limit via `resource.setrlimit`
- Hard 15-second timeout via `subprocess.run(timeout=...)`

---

## Key Design Decisions

**Why GRPO over PPO?** No critic or value network needed. GRPO normalizes rewards within a group of G responses, which is more stable for math where reward is binary (correct / wrong).

**Why LoRA for SFT then full merge?** LoRA is safer during SFT experimentation. Merging before GRPO avoids adapter overhead during the rollout generation phase which runs thousands of times.

**Why code execution before voting (Stage 1.5)?** The model's mental arithmetic is unreliable for large numbers. Running the code corrects computation errors before they propagate into the vote. A path that reasoned correctly but computed 347 times 892 wrong gets its answer fixed before it votes.

**Why UNCERTAIN beats FAIL in the last-resort waterfall?** UNCERTAIN means a verifier could not confirm or deny the answer. FAIL means a verifier found an active contradiction. Only the latter is a disproof. Routing UNCERTAIN answers above FAIL candidates prevents the fallback from discarding correct answers that the verifier simply could not check.

**Why greedy resample as S5?** Stochastic sampling at temp=0.7 can cluster all N paths around a wrong answer when the model has a strong but incorrect prior on the problem. Temperature=0 gives the single highest-probability response which often avoids the sampling artifact that caused all stochastic paths to fail.

---

## Output Format

Full JSONL audit trail (default):

```json
{
  "problem_id": "0",
  "problem": "Find all integers n where ...",
  "final_answer": "286",
  "confidence": 0.75,
  "verdict": "PASS",
  "verdict_conf": 0.85,
  "verdict_summary": "[code_execution] output 286 matches claimed answer",
  "strategy": "majority",
  "vote_counts": {"286": 24, "143": 6, "572": 2},
  "n_total": 32,
  "n_valid": 31,
  "fallback_used": false,
  "elapsed_sec": 87.3,
  "paths": [
    {
      "answer": "286",
      "tokens": 412,
      "score": 0.83,
      "verdict": "PASS",
      "had_code": true,
      "code_executed": true,
      "code_answer": "286",
      "answer_source": "code_confirmed"
    }
  ]
}
```



## Acknowledgements

- [Qwen2.5-Math](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) — base model
- [NuminaMath](https://huggingface.co/AI-MO/NuminaMath-CoT) — training data
- [TRL](https://github.com/huggingface/trl) — SFTTrainer and GRPOTrainer
- [vLLM](https://github.com/vllm-project/vllm) — fast inference engine
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — GRPO training recipe
- [Z3](https://github.com/Z3Prover/z3) — SMT solver
- [Lean 4 / Mathlib](https://leanprover-community.github.io) — formal verification