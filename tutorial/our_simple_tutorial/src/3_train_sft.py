"""
SFT (Supervised Fine-Tuning) Training Script
Uses TRL library to fine-tune Qwen2.5-3B
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_training_data(data_path: str) -> Dataset:
    """Load training data"""
    print(f"Loading training data: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Samples: {len(data)}")
    return Dataset.from_list(data)


def formatting_prompts_func(example):
    """Format single sample for training (supports two data formats)"""
    conversation = ""
    
    # Format 1: messages format (multi-turn dialogue)
    if 'messages' in example:
        messages = example['messages']
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == "system":
                conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                conversation += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    # Format 2: input/output format (single-turn dialogue)
    elif 'input' in example and 'output' in example:
        if example.get('instruction'):
            conversation += f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
        conversation += f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        conversation += f"<|im_start|>assistant\n{example['output']}<|im_end|>\n"
    
    return conversation


def setup_model_and_tokenizer(model_name: str = "../model/Qwen2.5-3B-Instruct",
                              use_4bit: bool = True,
                              num_gpus: int = None):
    """Setup model and tokenizer"""
    print(f"\nLoading model: {model_name}")
    print("-" * 60)
    
    # Configure GPU environment
    if num_gpus is None:
        num_gpus = min(2, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    else:
        num_gpus = min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    if torch.cuda.is_available() and num_gpus > 0:
        gpu_ids = ','.join(map(str, range(num_gpus)))
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
        print(f"GPU config: {num_gpus} GPUs (devices {gpu_ids})")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Set pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4bit quantization (saves memory)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    # Prepare model for training if using quantization
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    print("Model loaded successfully")
    return model, tokenizer


def setup_lora_config():
    """Setup LoRA configuration"""
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config


def train_sft(
    model_name: str = "../model/Qwen2.5-3B-Instruct",
    train_data_path: str = "../data/processed/medical_meadow/train.json",
    val_data_path: str = "../data/processed/medical_meadow/val.json",
    output_dir: str = "../output/trained_model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    use_4bit: bool = True,
    use_lora: bool = True,
    num_gpus: int = None
):
    """Execute SFT training"""
    from datetime import datetime
    
    print("\nStarting SFT Training")
    print("-" * 60)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = Path(output_dir) / timestamp
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path("../output/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training timestamp: {timestamp}")
    print(f"Model output directory: {model_output_dir}")
    print(f"Training log: {logs_dir / f'train_{timestamp}.json'}")
    
    # Load data
    train_dataset = load_training_data(train_data_path)
    
    # Load validation data if exists
    eval_dataset = None
    if Path(val_data_path).exists():
        eval_dataset = load_training_data(val_data_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name, use_4bit, num_gpus)
    
    # Apply LoRA
    if use_lora:
        print("\nApplying LoRA configuration")
        lora_config = setup_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(model_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100 if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_total_limit=3,
        fp16=False,
        bf16=True if torch.cuda.is_available() else False,
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="none",
        logging_dir=str(logs_dir),
        disable_tqdm=False,
    )
    
    # Create Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
    )
    
    # Start training
    print("\nStarting training...")
    print("-" * 60)
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    print("-" * 60)
    trainer.save_model(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))
    print(f"Model saved to: {model_output_dir}")
    
    # Save training log
    log_file_path = logs_dir / f"train_{timestamp}.json"
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"Training log saved to: {log_file_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument("--model_name", type=str, default="../model/Qwen2.5-3B-Instruct",
                       help="Model name or path")
    parser.add_argument("--train_data", type=str, default="../data/processed/medical_meadow/train.json",
                       help="Training data path")
    parser.add_argument("--val_data", type=str, default="../data/processed/medical_meadow/val.json",
                       help="Validation data path")
    parser.add_argument("--output_dir", type=str, default="../output/trained_model",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs to use (default: min(2, available))")
    parser.add_argument("--no_4bit", action="store_true",
                       help="Disable 4bit quantization")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA")
    
    args = parser.parse_args()
    
    # Execute training
    train_sft(
        model_name=args.model_name,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        use_4bit=not args.no_4bit,
        use_lora=not args.no_lora,
        num_gpus=args.num_gpus
    )
    
    print("\nSFT training complete!")
    print("-" * 60)
    print(f"Model saved in: {args.output_dir}")
    print("Next: Run 2_inference.py to test the fine-tuned model")


if __name__ == "__main__":
    main()
