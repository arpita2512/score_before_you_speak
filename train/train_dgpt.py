from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_from_disk
from transformers import AutoTokenizer
import os
import wandb
import sys
import argparse

block_size = 512

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size + 1

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def main(exp_name, dataset_path, n_epochs, output_path):
    
    wandb.init(project=exp_name, mode="offline")
    
    model_id = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vocab = tokenizer.get_vocab()
    #print(len(vocab))
    special_tokens = {
        'bos_token': '<|startoftext|>',
        'additional_special_tokens': ['<|sp1|>', '<|sp2|>']
    }
    
    _ = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    
    train_data = load_from_disk(dataset_path)
    
    train_dataset = train_data.map(group_texts, batched=True)
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(vocab))

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_path,
        logging_strategy="epoch",
        learning_rate=6.25e-5,
        weight_decay=0.01,
        num_train_epochs=n_epochs,
        skip_memory_metrics=True,
        per_device_train_batch_size=16,
        save_strategy='no',
        group_by_length=True,
        report_to="wandb",
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
 
    parser.add_argument("--exp_name", default="", type=str)
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--n_epochs", default=None, type=int)
    parser.add_argument("--output_path", default="", type=str)

    args = parser.parse_args()
    exp_name = args.exp_name
    dataset_path = args.dataset_path
    n_epochs = args.n_epochs
    output_path = args.output_path

    main(exp_name, dataset_path, n_epochs, output_path)