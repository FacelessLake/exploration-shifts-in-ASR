import torch
import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, \
    Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from transformers.trainer_pt_utils import get_parameter_names
from evaluate import load
from datasets import load_dataset, Audio
from dataclasses import dataclass
from typing import Dict, List, Union

DATA_PATH = "<path to your data>"
RANDOM_SEED = 147


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


def main():
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available!')
    device = torch.device('cuda')

    annot_len_dataset = load_dataset("parquet", data_dir=DATA_PATH)
    annot_len_dataset = annot_len_dataset.cast_column("audio", Audio(sampling_rate=16000))

    tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                                     pad_token="<pad>", word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, do_normalize=True, padding_value=0.0,
                                                 return_attention_mask=False, sampling_rate=16000)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def prepare_dataset(batch):
        audio = batch["audio"]

        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcription"]).input_ids
        return batch

    annot_len_dataset = annot_len_dataset.map(prepare_dataset, remove_columns=annot_len_dataset.column_names["train"])

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    cer_metric = load("cer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)

        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    model = Wav2Vec2ForCTC.from_pretrained("/userspace/bsm/models/wav2vec2-base-ru")
    model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir="/userspace/bsm/model_results/wav2vec2-pretrained-demo",
        overwrite_output_dir=True,
        gradient_checkpointing=True,
        group_by_length=True,

        metric_for_best_model="cer",
        greater_is_better=False,
        load_best_model_at_end=True,

        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,

        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=1,
        num_train_epochs=100,

        learning_rate=1e-4,
        weight_decay=1e-5,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )

    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    my_adam = torch.optim.Adam(optimizer_grouped_parameters, lr=training_args.learning_rate,
                               weight_decay=training_args.weight_decay)
    # my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_adam, gamma=0.99992996184159)
    my_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=my_adam, start_factor=0.01, total_iters=5260)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        optimizers=(my_adam, my_lr_scheduler),
        compute_metrics=compute_metrics,
        train_dataset=annot_len_dataset['train'],
        eval_dataset=annot_len_dataset['test'],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    # trainer.save_model("/userspace/bsm/model_results/model_checkpoint")


if __name__ == '__main__':
    main()
