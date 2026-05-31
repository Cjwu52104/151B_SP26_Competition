# CSE 151B SP26 Competition

## GPU & Inference Time

| Item | Detail |
|---|---|
| GPU | NVIDIA A100 80 GB (single GPU) |
| Approximate total inference time | ~45–60 minutes for the full private set |

## Model Weights

We use **`Qwen/Qwen3-4B-Thinking-2507`** (a public HuggingFace model — no fine-tuning checkpoint to download).

The model is loaded automatically from HuggingFace Hub when `run_inference()` is called. No manual download is required. If you want to pre-cache it:

```bash
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507
```

Place the private dataset at `data/private.jsonl` (same directory as `run_inference.py`) before running.

## How to Run

### Prerequisites

```bash
pip install vllm transformers tqdm
```

### Option 1 — command line

```bash
python run_inference.py
```

### Option 2 — from Python

```python
from run_inference import run_inference
run_inference()
```

Both options write the final submission CSV to `results/private_submission.csv`.

### Optional arguments

```python
run_inference(
    data_path="data/private.jsonl",       # path to private dataset
    output_path="results/private_submission.csv",  # output CSV path
    model_id="Qwen/Qwen3-4B-Thinking-2507",
    gpu_id="0",                           # CUDA_VISIBLE_DEVICES
    max_tokens=16384,
)
```

## Pipeline Summary

1. **Load** `Qwen3-4B-Thinking-2507` via vLLM (bfloat16, prefix caching enabled).
2. **Classify** each question as MCQ or free-form and select the appropriate system prompt.
3. **Generate** answers with chain-of-thought reasoning (`temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_tokens=16384`).
4. **Post-process**: extract `\boxed{}` answers; for MCQs where extraction fails, run a lightweight logprob calibration pass to recover the answer letter.
5. **Output** `results/private_submission.csv` with columns `id, response`.

## Repository Contents

| File | Description |
|---|---|
| `run_inference.py` | Single entry point — call `run_inference()` to reproduce results |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output CSV files written at runtime |
