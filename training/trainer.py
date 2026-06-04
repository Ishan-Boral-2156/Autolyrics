"""AutoLyrics — Seq2SeqTrainer wrapper."""

from __future__ import annotations

from pathlib import Path

from core.logging import get_logger
from evaluation.metrics import compute_wer_cer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

logger = get_logger(__name__)

WHISPER_KEYS = {"input_features", "labels", "decoder_input_ids", "attention_mask"}


WHISPER_KEYS = {"input_features", "labels", "decoder_input_ids", "attention_mask"}


class WhisperSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        clean = {k: v for k, v in inputs.items() if k in WHISPER_KEYS}
        outputs = model(**clean)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        clean = {k: v for k, v in inputs.items() if k in WHISPER_KEYS}
        return super().prediction_step(model, clean, prediction_loss_only, ignore_keys=ignore_keys)


class AutoLyricsTrainer:
    def __init__(
        self,
        model,
        processor,
        train_dataset,
        eval_dataset,
        data_collator,
        training_cfg,
        callbacks=None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.training_cfg = training_cfg
        self.callbacks = callbacks or []
        self._trainer = None

    def _build_training_args(self):
        cfg = self.training_cfg
        return Seq2SeqTrainingArguments(
            output_dir=cfg.get("output_dir", "./runs/default"),
            num_train_epochs=cfg.get("num_train_epochs", 10),
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 8),
            per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 8),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 2),
            learning_rate=cfg.get("learning_rate", 1e-4),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            bf16=False,
            fp16=True,
            optim=cfg.get("optim", "adamw_torch"),
            dataloader_num_workers=0,
            predict_with_generate=True,
            generation_max_length=cfg.get("generation_max_length", 225),
            eval_strategy=cfg.get("evaluation_strategy", "steps"),
            eval_steps=cfg.get("eval_steps", 200),
            save_strategy=cfg.get("save_strategy", "steps"),
            save_steps=cfg.get("save_steps", 200),
            save_total_limit=3,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            report_to=["tensorboard"],
            remove_unused_columns=True,
            label_names=["labels"],
        )

    def _compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer, cer = compute_wer_cer(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    def build(self):
        training_args = self._build_training_args()
        self._trainer = WhisperSeq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
            processing_class=self.processor.feature_extractor,
            callbacks=self.callbacks,
        )
        return self._trainer

    def train(self):
        if self._trainer is None:
            self.build()
        logger.info("Starting training...")
        result = self._trainer.train()
        output_dir = Path(self.training_cfg.get("output_dir", "./runs/default"))
        best_dir = output_dir / "best"
        self._trainer.save_model(str(best_dir))
        self.processor.save_pretrained(str(best_dir))
        logger.info("Best model saved to %s", best_dir)
        return result.metrics

    def evaluate(self):
        if self._trainer is None:
            self.build()
        return self._trainer.evaluate()
