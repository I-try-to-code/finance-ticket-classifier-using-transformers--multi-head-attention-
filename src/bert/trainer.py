"""Fine-tuning engine for BERT-base on BANKING77 with validation Macro F1 model selection."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

from src.bert.config import BertConfig
from src.bert.dataset import create_bert_data_loaders
from src.distilbert.trainer import EpochMetric, TrainingSummary, set_seed


def evaluate_bert_epoch(
    model: PreTrainedModel,
    val_loader: DataLoader,
    device: torch.device,
    fp16: bool = True,
) -> Tuple[float, float, float]:
    """Execute validation pass and compute (val_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []

    criterion = nn.CrossEntropyLoss()
    use_cuda_amp = fp16 and device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(val_loader.dataset)
    acc = float(accuracy_score(all_targets, all_preds))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return avg_loss, acc, macro_f1


def train_bert(
    train_df,
    val_df,
    id_to_label: Dict[int, str],
    config: BertConfig = BertConfig(),
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, TrainingSummary]:
    """Fine-tune BERT-base end-to-end on BANKING77 with best-model checkpointing based on Macro F1."""
    set_seed(config.seed)
    config.ensure_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"      Training on hardware device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Construct bidirectional mappings
    id2label = {int(k): str(v) for k, v in id_to_label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # Initialize model with classification head for 77 classes
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Prepare DataLoaders
    train_loader, val_loader = create_bert_data_loaders(
        train_df=train_df,
        val_df=val_df,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.max_length,
    )

    # Optimizer with weight decay exclusion for LayerNorm and biases
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_cuda_amp = config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)
    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = -1
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[EpochMetric] = []

    start_total_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_cuda_amp):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item() * len(labels)

        train_epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_macro_f1 = evaluate_bert_epoch(
            model=model,
            val_loader=val_loader,
            device=device,
            fp16=config.fp16,
        )
        epoch_duration = time.perf_counter() - epoch_start

        epoch_record = EpochMetric(
            epoch=epoch,
            train_loss=round(train_epoch_loss, 4),
            val_loss=round(val_loss, 4),
            val_macro_f1=round(val_macro_f1, 4),
            val_accuracy=round(val_acc, 4),
            duration_seconds=round(epoch_duration, 2),
        )
        history.append(epoch_record)

        print(
            f"      Epoch {epoch}/{config.epochs} | "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Track best model strictly by Validation Macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_training_duration = time.perf_counter() - start_total_time

    # Load best checkpoint weights
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Save best model artifact to disk
    model.save_pretrained(config.model_output_dir)
    tokenizer.save_pretrained(config.model_output_dir)

    summary = TrainingSummary(
        best_epoch=best_epoch,
        best_macro_f1=best_macro_f1,
        total_training_duration=total_training_duration,
        history=history,
        device=str(device),
        num_parameters=total_params,
        trainable_parameters=trainable_params,
    )

    return model, tokenizer, summary
