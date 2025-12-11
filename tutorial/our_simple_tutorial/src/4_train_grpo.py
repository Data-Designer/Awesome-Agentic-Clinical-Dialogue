"""
GRPO (Group Relative Policy Optimization) Training Script
Rewritten following official TRL documentation for simplicity and stability
"""

import os
import sys


# Force use of GPU 0 and 1 only (please modify this line if you want to use different GPUs)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
print(f"GPU configuration: Using GPU 0,1 only", flush=True)

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


def load_training_data(data_path: str) -> Dataset:
    """
    Load training data in GRPO format
    
    GRPO data format requirements:
    - prompt: required field, input text (for generation)
    - reference: optional field, reference answer (for reward function)
    
    Args:
        data_path: Path to data file
    
    Returns:
        Dataset object with prompt field
    """
    print(f"Loading training data: {data_path}", flush=True)
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"  Original data: {len(data)} samples", flush=True)
    
    # Convert to GRPO format
    grpo_data = []
    for item in data:
        # Format 1: Already in prompt+reference format
        if 'prompt' in item:
            grpo_data.append({
                "prompt": item['prompt'].strip(),
                "reference": item.get('reference', '').strip()
            })
        
        # Format 2: messages format (multi-turn dialogue)
        elif 'messages' in item:
            messages = item['messages']
            prompt_msgs = [m for m in messages if m['role'] in ['system', 'user']]
            response_msgs = [m for m in messages if m['role'] == 'assistant']
            
            if prompt_msgs and response_msgs:
                prompt = '\n'.join([f"{m['role']}: {m['content']}" for m in prompt_msgs])
                reference = response_msgs[0]['content']
                grpo_data.append({"prompt": prompt, "reference": reference})
        
        # Format 3: input/output format (single-turn dialogue)
        elif 'input' in item and 'output' in item:
            prompt = f"User: {item['input']}"
            if item.get('instruction'):
                prompt = f"System: {item['instruction']}\n{prompt}"
            grpo_data.append({"prompt": prompt, "reference": item['output']})
    
    print(f"  Converted samples: {len(grpo_data)}", flush=True)
    
    return Dataset.from_list(grpo_data)


def create_reward_function():
    """
    Create reward function following official API format
    
    Official signature: reward_func(completions, **kwargs)
    
    Returns:
        Reward function
    """
    import re
    
    # Try to import evaluation libraries (optional)
    try:
        from rouge_score import rouge_scorer
        rouge_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        use_rouge = True
        print("  ROUGE evaluation available", flush=True)
    except ImportError:
        use_rouge = False
        rouge_obj = None
        print("  ROUGE not available, using simple evaluation", flush=True)
    
    def reward_func(completions, **kwargs):
        """
        Reward function - official API format
        
        Args:
            completions: List of generated completions
            **kwargs: May contain inputs (original data)
        
        Returns:
            List of rewards (float)
        """
        rewards = []
        inputs = kwargs.get('inputs', [])
        
        for i, completion in enumerate(completions):
            generated = completion.strip()
            
            # Get reference answer
            reference = ""
            if i < len(inputs):
                sample = inputs[i]
                if isinstance(sample, dict):
                    reference = sample.get('reference', '')
            
            # Base reward
            reward = 0.0
            
            # 1. Length reward (avoid too short or too long)
            gen_len = len(generated)
            if 20 <= gen_len <= 200:
                reward += 0.5
            elif gen_len < 10:
                reward -= 0.5
            elif gen_len > 500:
                reward -= 0.3
            
            # 2. If reference answer exists, calculate similarity
            if reference and use_rouge and rouge_obj:
                try:
                    scores = rouge_obj.score(reference, generated)
                    reward += scores['rougeL'].fmeasure * 2.0
                except:
                    pass
            
            # 3. Simple quality checks
            if generated:
                # Complete sentence ending
                if generated[-1] in '.!?':
                    reward += 0.2
                
                # Avoid excessive repetition
                words = generated.lower().split()
                if len(words) > 5:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio > 0.7:
                        reward += 0.3
                    elif unique_ratio < 0.5:
                        reward -= 0.5
            
            rewards.append(float(reward))
        
        return rewards
    
    return reward_func


def train_grpo(
    model_name: str = "../model/Qwen2.5-3B-Instruct",
    train_data_path: str = "test_data_tiny.json",
    output_dir: str = "../output/trained_model_grpo",
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    max_completion_length: int = 128,
    num_generation: int = 4,
    no_4bit: bool = True,  # Disable 4bit by default
    no_lora: bool = False,
):
    """
    Execute GRPO training following official documentation
    
    Args:
        model_name: Model name or path
        train_data_path: Training data path
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_completion_length: Maximum generation length
        num_generation: Number of candidates per prompt
        no_4bit: Disable 4bit quantization (recommended)
        no_lora: Disable LoRA
    """
    from datetime import datetime
    
    print("\nStarting GRPO training (official API version)", flush=True)
    print("-" * 60, flush=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_output_dir = Path(output_dir) / timestamp
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model output directory: {model_output_dir}", flush=True)
    print(f"Config: epochs={num_epochs}, batch_size={batch_size}, num_generation={num_generation}", flush=True)
    print(f"Quantization: {'Disabled' if no_4bit else 'Enabled (4bit)'} | LoRA: {'Disabled' if no_lora else 'Enabled'}", flush=True)
    
    # 1. Load data
    print("\n1. Loading data...", flush=True)
    train_dataset = load_training_data(train_data_path)
    
    # 2. Create reward function
    print("\n2. Creating reward function...", flush=True)
    reward_fn = create_reward_function()
    print("  Reward function created", flush=True)
    
    # 3. Configure training parameters
    print("\n3. Configuring training parameters...", flush=True)
    training_args = GRPOConfig(
        # Basic parameters
        output_dir=str(model_output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        
        # Optimizer parameters
        gradient_accumulation_steps=2,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.3,
        
        # Training control
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        bf16=True if torch.cuda.is_available() else False,
        
        # GRPO specific parameters
        num_generations=num_generation,
        generation_batch_size=num_generation,
        max_completion_length=max_completion_length,
        temperature=1.0,
        generation_kwargs={"top_p": 0.9, "top_k": 50},
        
        # Other
        report_to="none",
        disable_tqdm=False,
    )
    print("  GRPOConfig created successfully", flush=True)
    
    # 4. Initialize Trainer
    print("\n4. Initializing GRPOTrainer...", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Data: {len(train_dataset)} samples", flush=True)
    
    try:
        trainer = GRPOTrainer(
            model=model_name,
            args=training_args,
            train_dataset=train_dataset,
            reward_funcs=reward_fn,
        )
        print("  GRPOTrainer created successfully", flush=True)
        
    except Exception as e:
        print(f"  GRPOTrainer creation failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return
    
    # 5. Start training
    print("\n" + "-" * 60, flush=True)
    print("Starting training...", flush=True)
    print("Note: Each step requires generating multiple candidates, may be slow", flush=True)
    print("-" * 60 + "\n", flush=True)
    
    try:
        trainer.train()
        
        # Save model
        print("\n" + "-" * 60, flush=True)
        print("Saving model...", flush=True)
        print("-" * 60, flush=True)
        
        trainer.save_model(str(model_output_dir))
        print(f"Model saved to: {model_output_dir}", flush=True)
        
        # Save training log
        log_dir = Path("../output/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"train_grpo_{timestamp}.json"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"Training log saved to: {log_file}", flush=True)
        
        print("\n" + "-" * 60, flush=True)
        print("GRPO training completed successfully!", flush=True)
        print("-" * 60, flush=True)
        
    except Exception as e:
        print(f"\nTraining error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        print("\nIf encountering CUDA errors, try:", flush=True)
        print("1. Reduce num_generation (from 4 to 2)", flush=True)
        print("2. Reduce max_completion_length (from 128 to 64)", flush=True)
        print("3. Increase temperature (for more random generation)", flush=True)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO Training Script (Official API)")
    parser.add_argument("--model_name", type=str, default="../model/Qwen2.5-3B-Instruct",
                       help="Model name or path")
    parser.add_argument("--train_data", type=str, default="test_data_tiny.json",
                       help="Training data path")
    parser.add_argument("--output_dir", type=str, default="../output/trained_model_grpo",
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--max_completion_length", type=int, default=128,
                       help="Maximum generation length")
    parser.add_argument("--num_generation", type=int, default=4,
                       help="Number of candidates per prompt")
    parser.add_argument("--no_4bit", action="store_true", default=True,
                       help="Disable 4bit quantization (default)")
    parser.add_argument("--no_lora", action="store_true",
                       help="Disable LoRA")
    
    args = parser.parse_args()
    
    print("\nGRPO Training Configuration", flush=True)
    print("-" * 60, flush=True)
    print(f"Model: {args.model_name}", flush=True)
    print(f"Data: {args.train_data}", flush=True)
    print(f"GPU: 0,1 (restricted)", flush=True)
    print(f"Quantization: {'Disabled (recommended)' if args.no_4bit else 'Enabled (4bit)'}", flush=True)
    print("-" * 60, flush=True)
    
    # Execute training
    train_grpo(
        model_name=args.model_name,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_completion_length=args.max_completion_length,
        num_generation=args.num_generation,
        no_4bit=args.no_4bit,
        no_lora=args.no_lora,
    )


if __name__ == "__main__":
    main()
