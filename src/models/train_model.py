import os
from ..data import load_data
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import re

def get_next_version(output_dir):
    existing_versions = [int(re.search(r'v(\d+)', d).group(1)) for d in os.listdir(output_dir) if re.search(r'v(\d+)', d)]
    next_version = max(existing_versions, default=0) + 1
    return next_version

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/TrOCR/Fine_tune_TrOCR_on_IAM_Handwriting_Database_using_Seq2SeqTrainer.ipynb
def train_model(output_dir="output", checkpoint=None, from_pretrained_model="./Caroline", device='cpu', compute_cer_flag=False, version=None, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    
    if version is None:
        version = get_next_version(output_dir)
    versioned_output_dir = os.path.join(output_dir, f"v{version}")

    dataset = load_data.load_data(**kwargs)
    train_dataset, test_dataset = load_data.split_dataset(dataset)
    
    processor = TrOCRProcessor.from_pretrained(from_pretrained_model)
    train_dataloader, eval_dataloader = load_data.create_dataloaders(train_dataset, test_dataset, processor, kwargs.get('batch_size', 4))
    
    model = VisionEncoderDecoderModel.from_pretrained(from_pretrained_model)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # Print the epochs value to debug
    print(f"Training for {kwargs.get('epochs', 1)} epochs")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=versioned_output_dir,
        per_device_train_batch_size=kwargs.get('batch_size', 4),
        num_train_epochs=float(kwargs.get("epochs", 1)),  # Ensure it's a float
        logging_steps=kwargs.get("logging_steps", 100),
        save_steps=kwargs.get("save_steps", 100),
        # evaluation_strategy="steps",
        save_total_limit=kwargs.get("save_limit", 2),
        predict_with_generate=True,
        fp16=False,
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader.dataset,
        eval_dataset=eval_dataloader.dataset,
    )

    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    
    model.save_pretrained(versioned_output_dir)
    processor.save_pretrained(versioned_output_dir)
    
    return model, processor, eval_dataloader
