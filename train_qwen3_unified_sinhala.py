from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen3-1.7B"
SYSTEM_PROMPT = (
    "You are a Sinhala transcript and text checker. Decide whether the input has a "
    "spelling, grammar, spacing, repeated-word, omitted-word, Unicode, "
    "punctuation, or realistic transcription error. Preserve the original "
    "meaning and preserve formal, neutral, or colloquial wording. Make only "
    "necessary corrections. Reply using exactly two lines:\n"
    "STATUS: CORRECT or STATUS: INCORRECT\n"
    "CORRECTION: corrected sentence\n"
    "/no_think"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one Qwen3-1.7B adapter on unified Sinhala data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            Path(__file__).parents[1]
            / "data"
            / "sinhala_unified_correction_500.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/qwen3-1.7b-unified-sinhala-corrector"),
    )
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Permit rows that have not yet been approved by a Sinhala reviewer.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest checkpoint in output-dir.",
    )
    return parser.parse_args()


def load_rows(
    path: Path,
    allow_pending: bool,
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path.resolve())
    with path.open("r", encoding="utf-8-sig") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    required = {
        "example_id",
        "base_phrase_id",
        "input_text",
        "status",
        "correct_text",
        "error_type",
        "review_status",
        "source",
        "split",
    }
    if not rows:
        raise ValueError("Dataset is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")

    pairs = [(row["input_text"], row["correct_text"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate input/correction pairs")

    for row in rows:
        if row["status"] not in {"CORRECT", "INCORRECT"}:
            raise ValueError(f"Invalid status in {row['example_id']}")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split in {row['example_id']}")
        should_be_correct = row["input_text"] == row["correct_text"]
        is_labelled_correct = row["status"] == "CORRECT"
        if should_be_correct != is_labelled_correct:
            raise ValueError(f"Incorrect label in {row['example_id']}")
        review_status = row["review_status"].strip().lower()
        if review_status not in {"approved", "revised"} and not allow_pending:
            raise ValueError(
                f"{row['example_id']} is not reviewed. Mark rows approved/revised "
                "or pass --allow-pending for a synthetic pilot."
            )

    phrase_splits: dict[str, set[str]] = {}
    for row in rows:
        phrase_splits.setdefault(row["base_phrase_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in phrase_splits.values()):
        raise ValueError("A base phrase appears in more than one split")
    return rows


def to_training_record(row: dict[str, str]) -> dict:
    return {
        "example_id": row["example_id"],
        "base_phrase_id": row["base_phrase_id"],
        "error_type": row["error_type"],
        "source": row["source"],
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["input_text"]},
        ],
        "completion": [
            {
                "role": "assistant",
                "content": (
                    f"STATUS: {row['status']}\n"
                    f"CORRECTION: {row['correct_text']}"
                ),
            }
        ],
    }


def balance_training_rows(
    rows: list[dict[str, str]],
    seed: int,
) -> list[dict[str, str]]:
    correct = [row for row in rows if row["status"] == "CORRECT"]
    incorrect = [row for row in rows if row["status"] == "INCORRECT"]
    if not correct or not incorrect:
        raise ValueError("Training split requires both CORRECT and INCORRECT rows")

    rng = random.Random(seed)
    balanced_correct = [
        copy
        for copy in (
            rng.choice(correct) for _ in range(len(incorrect))
        )
    ]
    balanced = incorrect + balanced_correct
    rng.shuffle(balanced)
    return balanced


def newest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and path.is_dir():
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None))[1]


def generate_result(model, tokenizer, input_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_result(text: str) -> tuple[str, str]:
    status = re.search(r"STATUS:\s*(CORRECT|INCORRECT)", text, re.I)
    correction = re.search(r"CORRECTION:\s*(.*)", text, re.I | re.S)
    return (
        status.group(1).upper() if status else "INVALID",
        correction.group(1).strip() if correction else "",
    )


def accuracy(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(bool(row[field]) for row in rows) / len(rows)


def verify_hardware() -> None:
    print("Operating system:", platform.platform())
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA:", torch.version.cuda)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a CUDA-enabled PyTorch build."
        )

    gpu_name = torch.cuda.get_device_name(0)
    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    capability = torch.cuda.get_device_capability(0)

    print("GPU:", gpu_name)
    print("VRAM:", round(vram_gb, 2), "GB")
    print("Compute capability:", capability)

    if vram_gb < 7.5:
        raise RuntimeError(
            "This configuration expects approximately 8 GB VRAM or more."
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    verify_hardware()

    rows = load_rows(args.dataset, args.allow_pending)
    split_rows = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    print("Split sizes:", {key: len(value) for key, value in split_rows.items()})
    print("Training sources:", Counter(row["source"] for row in split_rows["train"]))
    print("Training errors:", Counter(row["error_type"] for row in split_rows["train"]))

    balanced_train_rows = balance_training_rows(
        split_rows["train"],
        args.seed,
    )
    print(
        "Balanced training statuses:",
        Counter(row["status"] for row in balanced_train_rows),
    )
    train_dataset = Dataset.from_list(
        [to_training_record(row) for row in balanced_train_rows]
    )
    validation_dataset = Dataset.from_list(
        [to_training_record(row) for row in split_rows["validation"]]
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    torch.backends.cuda.matmul.allow_tf32 = True

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=args.max_length,
        packing=True,
        completion_only_loss=True,
        report_to="none",
        dataloader_num_workers=0,
        dataset_num_proc=1,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    resume_checkpoint = newest_checkpoint(args.output_dir) if args.resume else None
    if args.resume and resume_checkpoint is None:
        print("No checkpoint found; starting from the base model.")

    trainer.train(
        resume_from_checkpoint=(
            str(resume_checkpoint) if resume_checkpoint is not None else None
        )
    )

    adapter_dir = args.output_dir / "final-adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    trainer.model.eval()
    predictions = []
    for row in split_rows["test"]:
        generated = generate_result(
            trainer.model,
            tokenizer,
            row["input_text"],
        )
        predicted_status, predicted_correction = parse_result(generated)
        predictions.append(
            {
                **row,
                "model_output": generated,
                "predicted_status": predicted_status,
                "predicted_correction": predicted_correction,
                "status_correct": predicted_status == row["status"],
                "correction_exact": (
                    predicted_correction == row["correct_text"]
                ),
            }
        )

    prediction_path = args.output_dir / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(predictions[0]),
        )
        writer.writeheader()
        writer.writerows(predictions)

    metrics = {
        "test_rows": len(predictions),
        "status_accuracy": accuracy(predictions, "status_correct"),
        "exact_correction_accuracy": accuracy(predictions, "correction_exact"),
        "unchanged_input_accuracy": accuracy(
            [row for row in predictions if row["status"] == "CORRECT"],
            "correction_exact",
        ),
        "by_source": {},
        "by_error_type": {},
    }
    for source in sorted({row["source"] for row in predictions}):
        source_rows = [row for row in predictions if row["source"] == source]
        metrics["by_source"][source] = {
            "rows": len(source_rows),
            "status_accuracy": accuracy(source_rows, "status_correct"),
            "exact_correction_accuracy": accuracy(
                source_rows,
                "correction_exact",
            ),
        }
    for error_type in sorted({row["error_type"] for row in predictions}):
        error_rows = [
            row for row in predictions if row["error_type"] == error_type
        ]
        metrics["by_error_type"][error_type] = {
            "rows": len(error_rows),
            "status_accuracy": accuracy(error_rows, "status_correct"),
            "exact_correction_accuracy": accuracy(
                error_rows,
                "correction_exact",
            ),
        }
    metrics_path = args.output_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(adapter_dir),
        "zip",
        adapter_dir,
    )
    print("Adapter:", adapter_dir.resolve())
    print("Adapter ZIP:", Path(archive).resolve())
    print("Predictions:", prediction_path.resolve())
    print("Metrics:", metrics_path.resolve())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
