"""
CSE 151B Competition — Inference Pipeline

Usage:
    python run_inference.py
    # or from another script:
    from run_inference import run_inference
    run_inference()
"""

import csv
import json
import os
import re
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = "0"
DATA_PATH   = "data/private.jsonl"
OUTPUT_PATH = "results/private_submission.csv"
MAX_TOKENS  = 16384


# ---------------------------------------------------------------------------
# LaTeX repair — only fixes tokens inside $...$ blocks
# ---------------------------------------------------------------------------
_LATEX_SUBS = [
    (r'(?<!\\)\bfrac\b',    r'\\frac'),
    (r'(?<!\\)\binfty\b',   r'\\infty'),
    (r'(?<!\\)\bint\b',     r'\\int'),
    (r'(?<!\\)\bsum\b',     r'\\sum'),
    (r'(?<!\\)\bprod\b',    r'\\prod'),
    (r'(?<!\\)\blim\b',     r'\\lim'),
    (r'(?<!\\)\bsqrt\b',    r'\\sqrt'),
    (r'(?<!\\)\bsin\b',     r'\\sin'),
    (r'(?<!\\)\bcos\b',     r'\\cos'),
    (r'(?<!\\)\btan\b',     r'\\tan'),
    (r'(?<!\\)\bcot\b',     r'\\cot'),
    (r'(?<!\\)\bsec\b',     r'\\sec'),
    (r'(?<!\\)\bcsc\b',     r'\\csc'),
    (r'(?<!\\)\blog\b',     r'\\log'),
    (r'(?<!\\)\bln\b',      r'\\ln'),
    (r'(?<!\\)\bexp\b',     r'\\exp'),
    (r'(?<!\\)\bpi\b',      r'\\pi'),
    (r'(?<!\\)\btheta\b',   r'\\theta'),
    (r'(?<!\\)\balpha\b',   r'\\alpha'),
    (r'(?<!\\)\bbeta\b',    r'\\beta'),
    (r'(?<!\\)\bgamma\b',   r'\\gamma'),
    (r'(?<!\\)\bdelta\b',   r'\\delta'),
    (r'(?<!\\)\bsigma\b',   r'\\sigma'),
    (r'(?<!\\)\blambda\b',  r'\\lambda'),
    (r'(?<!\\)\bmu\b',      r'\\mu'),
    (r'(?<!\\)\bepsilon\b', r'\\epsilon'),
    (r'(?<!\\)\bphi\b',     r'\\phi'),
    (r'(?<!\\)\bomega\b',   r'\\omega'),
    (r'(?<!\\)\bcdot\b',    r'\\cdot'),
    (r'(?<!\\)\bpm\b',      r'\\pm'),
    (r'(?<!\\)\bleq\b',     r'\\leq'),
    (r'(?<!\\)\bgeq\b',     r'\\geq'),
    (r'(?<!\\)\bneq\b',     r'\\neq'),
    (r'(?<!\\)\btimes\b',   r'\\times'),
    (r'(?<!\\)\bpartial\b', r'\\partial'),
    (r'(?<!\\)\bnabla\b',   r'\\nabla'),
]
_COMPILED_SUBS = [(re.compile(p), r) for p, r in _LATEX_SUBS]
_MATH_BLOCK = re.compile(r'\$+[^$]*?\$+')


def _repair_math_block(match: re.Match) -> str:
    body = match.group(0)
    for pattern, replacement in _COMPILED_SUBS:
        body = pattern.sub(replacement, body)
    return body


def repair_latex(text: str) -> str:
    """Restore missing backslashes before LaTeX commands, but only inside $...$."""
    return _MATH_BLOCK.sub(_repair_math_block, text)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician solving a multiple-choice problem.\n\n"
    "Inside <think>, work through these steps:\n"
    "  1. CLASSIFY — Identify the mathematical domain (e.g., calculus, linear algebra, "
         "complex analysis) and the specific concept being tested.\n"
    "  2. SETUP — Write the relevant formula, theorem, or method from first principles. "
         "Do not skip steps.\n"
    "  3. SOLVE — Carry out all computations in full. Show every intermediate result. "
         "Keep exact forms (fractions, radicals, limits) rather than approximating.\n"
    "  4. MATCH — Compare your result to every listed option. "
         "Check for algebraic equivalence (e.g., rationalized denominators, "
         "factored vs. expanded, trigonometric identities) before concluding no match.\n"
    "  5. ELIMINATE — Briefly rule out each wrong option and state why.\n"
    "  6. CONFIRM — Restate the correct letter and the core reason it is right.\n\n"
    "Options are labeled A, B, C, … and may go up to J (10 options). "
    "Read the option labels carefully before selecting.\n\n"
    "After </think>, output EXACTLY ONE LINE and nothing else:\n"
    "\\boxed{X}\n"
    "where X is the single capital letter of the correct option. "
    "No prose, no punctuation, no blank lines after </think>."
)

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician.\n\n"
    "Inside <think>, work through these steps:\n"
    "  1. COUNT — Scan the problem and count every [ANS] placeholder. "
         "Label them [ANS]_1, [ANS]_2, … in the order they appear. "
         "State what quantity each one asks for.\n"
    "  2. PLAN — Identify the relevant formulas, theorems, or definitions "
         "you will need. Write them explicitly.\n"
    "  3. SOLVE — Compute each answer in the order [ANS]_1, [ANS]_2, …, "
         "showing every algebraic or calculus step. "
         "Prefer exact forms (reduced fractions, simplified radicals, factored expressions) "
         "unless the problem explicitly asks for a decimal.\n"
    "  4. VERIFY — Check each answer independently: substitute back into the original "
         "equation, test boundary/special cases, or use an alternative method.\n"
    "  5. COLLECT — List the final answers in order before writing \\boxed{}.\n\n"
    "After </think>, output EXACTLY ONE LINE and nothing else:\n"
    "\\boxed{a_1, a_2, ...}\n\n"
    "Formatting rules (strictly enforced):\n"
    "  • ONE \\boxed{} total — never multiple \\boxed{} expressions.\n"
    "  • Single answer  →  \\boxed{42}\n"
    "  • Multiple answers  →  \\boxed{ans1, ans2, ans3}  in [ANS]_1, [ANS]_2, … order.\n"
    "  • No units, words, or explanation inside \\boxed{}.\n"
    "  • Fully simplify: reduce fractions, rationalize denominators, "
         "collect like terms, expand products where simpler.\n"
    "  • If an answer is an expression, write it in standard mathematical notation "
         "(e.g., x**2 + 3*x - 1 or x^2 + 3x - 1)."
)


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt); applies LaTeX repair to question text."""
    question = repair_latex(question)
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(
            f"{lbl}. {repair_latex(opt.strip())}" for lbl, opt in zip(labels, options)
        )
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
_ANSWER_PATTERNS = [
    r"\\boxed\{([A-Ja-j])\}",
    r"(?:the\s+)?answer\s+is\s+[\(\[]?([A-J])[\)\]]?",
    r"(?:correct\s+)?answer\s*[:\-]\s*[\(\[]?([A-J])[\)\]]?",
    r"(?:therefore|thus|so),?\s+(?:the\s+)?(?:answer|choice|option)\s+is\s+[\(\[]?([A-J])[\)\]]?",
    r"option\s+([A-J])\s+is\s+correct",
]
_OPTION_MARKER = re.compile(r"^([A-J])[.)]\s", re.MULTILINE)


def extract_letter(text: str) -> str:
    after_think = text.split("</think>", 1)[-1] if "</think>" in text else text
    for source in (after_think, text):
        for pat in _ANSWER_PATTERNS:
            m = re.search(pat, source, re.IGNORECASE)
            if m:
                return m.group(1).upper()
    markers = _OPTION_MARKER.findall(after_think)
    return markers[-1].upper() if markers else ""


def score_mcq_logprobs(calib_out, options_count: int, letter_token_ids: dict) -> str:
    gen_text = calib_out.outputs[0].text.strip()
    valid = [chr(65 + i) for i in range(options_count)]
    if gen_text and gen_text[0].upper() in valid:
        return gen_text[0].upper()
    lps = calib_out.outputs[0].logprobs
    if lps:
        first_pos = lps[0]
        best_letter, best_lp = "", float("-inf")
        for letter in valid:
            tid = letter_token_ids.get(letter)
            if tid and tid in first_pos:
                lp = first_pos[tid].logprob
                if lp > best_lp:
                    best_lp, best_letter = lp, letter
        if best_letter:
            return best_letter
    return extract_letter(gen_text)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_inference(
    data_path: str = DATA_PATH,
    output_path: str = OUTPUT_PATH,
    model_id: str = MODEL_ID,
    gpu_id: str = GPU_ID,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """
    Run the full inference pipeline end-to-end.

    Loads the model, runs generation on data_path, applies all post-processing,
    and writes the final submission CSV to output_path.

    Returns the path to the written CSV.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from tqdm import tqdm

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Load dataset
    data = [json.loads(line) for line in open(data_path)]
    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = sum(not d.get("options")   for d in data)
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        max_model_len=max_tokens,
        trust_remote_code=True,
        max_num_seqs=64,
    )
    print("Model loaded.")

    _BASE = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0,
                 presence_penalty=0.0, repetition_penalty=1.0)
    main_sampling = SamplingParams(n=1, max_tokens=max_tokens, **_BASE)
    logprob_sampling = SamplingParams(
        n=1, max_tokens=10, temperature=0.01, top_p=1.0, logprobs=20,
    )

    # Build prompts
    def make_prompt(item: dict) -> str:
        system, user = build_prompt(item["question"], item.get("options"))
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def make_calib_prompt(item: dict) -> str:
        _, user = build_prompt(item["question"], item.get("options"))
        try:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": "Output only the answer letter (A-J), nothing else."},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                [{"role": "system", "content": "Output only the answer letter (A-J), nothing else."},
                 {"role": "user",   "content": user}],
                tokenize=False, add_generation_prompt=True,
            )

    prompts = [make_prompt(item) for item in data]

    # Main generation pass
    print(f"Running main generation for {len(prompts)} questions...")
    outputs = llm.generate(prompts, main_sampling)
    responses = [out.outputs[0].text.strip() for out in outputs]
    print("Main generation done.")

    # Build letter token ID map for logprobs fallback
    letter_token_ids = {}
    for ch in "ABCDEFGHIJ":
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if ids:
            letter_token_ids[ch] = ids[0]

    # Lazy calibration pass for MCQs where extract_letter abstains
    mcq_indices = [i for i, item in enumerate(data) if item.get("options")]
    need_calib = [i for i in mcq_indices if not extract_letter(responses[i])]

    calib_map = {}
    if need_calib:
        print(f"Running fallback calibration for {len(need_calib)} MCQ(s)...")
        calib_prompts = [make_calib_prompt(data[i]) for i in need_calib]
        calib_outs = llm.generate(calib_prompts, logprob_sampling)
        calib_map = {need_calib[k]: calib_outs[k] for k in range(len(calib_outs))}

    # Collect final responses — override abstained MCQs with calibration result
    results = []
    for idx, (item, response) in tqdm(enumerate(zip(data, responses)),
                                       total=len(data), desc="Collecting"):
        if idx in calib_map:
            n_opts = len(item.get("options", []))
            letter = score_mcq_logprobs(calib_map[idx], n_opts, letter_token_ids)
            if letter:
                # Rewrite response to contain the calibrated letter in \boxed{}
                response = f"<think></think>\n\\boxed{{{letter}}}"
        results.append({"id": item["id"], "response": response})

    print(f"Collected {len(results)} responses.")

    # Write CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for r in results:
            writer.writerow([r["id"], r["response"]])

    print(f"Saved {len(results)} rows to {out_path}")
    print("Done! Submit", output_path)
    return str(out_path)


if __name__ == "__main__":
    run_inference()
