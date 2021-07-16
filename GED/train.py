import argparse
import logging
import math
import random
import os

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed
)

from models.bert_for_ged import BertCrfForGED

logger = logging.getLogger(__name__)

def parser_args():
    parser = argparse.ArgumentParser(description='Make GED as a NER task')

    parser.add_argument('--train_file_path', type=str, default=None, required=True)
    parser.add_argument('--valid_file_path', type=str, default=None, required=True)
    parser.add_argument("--text_column_name", type=str, default=None,
        help="The column name of text to input in the file (a csv or JSON file).",)
    parser.add_argument("--label_column_name", type=str, default=None,
        help="The column name of label to input in the file (a csv or JSON file).",)
    parser.add_argument('--max_length', type=int, default=128,
                        help=(
                            "The maximum total input sequence length after tokenization. Sequence longer than this will be truncated,"
                            "sequences shorter will be padded if '--pad_to_max_length' is passed"
                        ))
    parser.add_argument('--pad_to_max_length', action="store_true",
                        help = "If passed, pad all samples to 'max_length', otherwise, dynamic padding is used")
    parser.add_argument('--model_name_or_path', type=str, required=True,
                        help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8,
                        help='Batch size (per device) for the training dataloader')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8,
                        help='Batch size (per device) for the evaluation dataloader')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay to use.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps', type=int, default=None,
                        help='Total number of training steps to perform. If provided, overrides num_train_epochs')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the final model.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    parser.add_argument("--label_all_tokens", action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",)
    parser.add_argument("--return_entity_level_metrics", action="store_true",
        help="Indication whether entity level metrics are to be returner.",)
    parser.add_argument("--debug", action="store_true", help="Activate debug mode and run training only with a subset of data.",)

    args = parser.parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args

def main():
    args = parser_args()

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    data_files = {}
    if args.train_file is not None:
        data_files['train'] = args.train_file
    if args.validation_file is not None:
        data_files['validation'] = args.validation_file
    extension = args.train_file.split('.')[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    if args.label_column_name is not None:
        label_column_name = args.label_column_name

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets['train'][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = BertCrfForGED.from_pretrained(args.model_name_or_path, config=config)

    model.resize_token_embeddings(len(tokenizer))

    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            is_split_into_words=False,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens \
                                     else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets['train'].column_names,
        desc='Running tokenizer on dataset'
    )

    train_dataset = processed_raw_datasets['train']
    eval_dataset = processed_raw_datasets['validation']

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')

    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    device = accelerator.device
    model.to(device)

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    metric = load_metric('seqeval')

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=1)
            labels = batch['labels']
            if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )

        eval_metric = compute_metrics()
        accelerator.print(f'epoch {epoch}:', eval_metric)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)

if __name__ == '__main__':
    main()