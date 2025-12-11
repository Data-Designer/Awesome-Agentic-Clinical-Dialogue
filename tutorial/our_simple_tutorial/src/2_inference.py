"""
Model Inference Script
Uses Qwen2.5-3B model for inference testing
"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class MedicalChatbot:
    """Medical dialogue chatbot"""
    
    def __init__(self, model_name: str = "../model/Qwen2.5-3B-Instruct", 
                 device: str = "auto",
                 num_gpus: int = None,
                 gpu_memory_per_device: str = "10GiB",
                 cpu_memory: str = "30GiB",
                 max_memory: Dict = None):
        """Initialize model"""
        print(f"\nLoading model: {model_name}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Configure device and memory
        if max_memory is None and torch.cuda.is_available():
            if num_gpus is None:
                num_gpus = min(2, torch.cuda.device_count())
            else:
                num_gpus = min(num_gpus, torch.cuda.device_count())
            
            max_memory = {i: gpu_memory_per_device for i in range(num_gpus)}
            max_memory["cpu"] = cpu_memory
            self.num_gpus = num_gpus
            print(f"GPU config: {num_gpus} GPUs, {gpu_memory_per_device} each")
        else:
            self.num_gpus = 0 if not torch.cuda.is_available() else torch.cuda.device_count()
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            max_memory=max_memory,
            trust_remote_code=True
        )
        
        print("Model loaded successfully")
        if torch.cuda.is_available():
            print(f"  GPUs: {self.num_gpus}")
            print(f"  GPU model: {torch.cuda.get_device_name(0)}")
        else:
            print("  Using CPU")
    
    def chat(self, messages: List[Dict[str, str]], 
             max_new_tokens: int = 512,
             temperature: float = 0.7,
             top_p: float = 0.9) -> str:
        """Generate conversation response"""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def batch_inference(self, data_list: List[Dict], 
                       batch_size: int = 4,
                       save_path: str = None) -> List[Dict]:
        """Batch inference"""
        from tqdm import tqdm
        
        results = []
        
        with tqdm(total=len(data_list), desc="Inference", unit="samples") as pbar:
            for item in data_list:
                # Extract messages
                if "messages" in item:
                    messages = item["messages"][:-1] if item["messages"][-1]["role"] == "assistant" else item["messages"]
                    ground_truth = item["messages"][-1]["content"] if item["messages"][-1]["role"] == "assistant" else ""
                else:
                    messages = [{"role": "user", "content": item.get("input", "")}]
                    ground_truth = item.get("output", "")
                
                # Add system prompt to limit length
                if not any(msg.get("role") == "system" for msg in messages):
                    messages.insert(0, {
                        "role": "system",
                        "content": "You are a medical AI assistant. Provide concise and accurate responses."
                    })
                
                # Generate response
                try:
                    response = self.chat(messages, max_new_tokens=256)
                    results.append({
                        "messages": messages[1:] if messages[0]["role"] == "system" else messages,
                        "prediction": response,
                        "ground_truth": ground_truth
                    })
                except Exception as e:
                    results.append({
                        "messages": messages[1:] if messages[0]["role"] == "system" else messages,
                        "prediction": "[Inference failed]",
                        "ground_truth": ground_truth,
                        "error": str(e)
                    })
                
                pbar.update(1)
        
        # Save results
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {save_path}")
        
        return results


def evaluate_responses(results: List[Dict]) -> Dict:
    """Evaluate generated response quality"""
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import numpy as np
    
    print("\nEvaluating response quality")
    
    metrics = {
        "bleu_scores": [],
        "rouge1_scores": [],
        "rouge2_scores": [],
        "rougeL_scores": [],
        "response_lengths": [],
        "success_rate": 0
    }
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction()
    success_count = 0
    
    for result in results:
        prediction = result.get("prediction", "")
        ground_truth = result.get("ground_truth", "")
        
        if not prediction or prediction == "[Inference failed]":
            continue
        
        success_count += 1
        
        # BLEU score
        try:
            reference = [ground_truth.split()]
            hypothesis = prediction.split()
            bleu = sentence_bleu(reference, hypothesis, smoothing_function=smooth.method1)
            metrics["bleu_scores"].append(bleu)
        except:
            pass
        
        # ROUGE scores
        try:
            rouge_scores = scorer.score(ground_truth, prediction)
            metrics["rouge1_scores"].append(rouge_scores['rouge1'].fmeasure)
            metrics["rouge2_scores"].append(rouge_scores['rouge2'].fmeasure)
            metrics["rougeL_scores"].append(rouge_scores['rougeL'].fmeasure)
        except:
            pass
        
        metrics["response_lengths"].append(len(prediction))
    
    # Calculate averages
    metrics["success_rate"] = success_count / len(results) if results else 0
    metrics["avg_bleu"] = np.mean(metrics["bleu_scores"]) if metrics["bleu_scores"] else 0
    metrics["avg_rouge1"] = np.mean(metrics["rouge1_scores"]) if metrics["rouge1_scores"] else 0
    metrics["avg_rouge2"] = np.mean(metrics["rouge2_scores"]) if metrics["rouge2_scores"] else 0
    metrics["avg_rougeL"] = np.mean(metrics["rougeL_scores"]) if metrics["rougeL_scores"] else 0
    metrics["avg_response_length"] = np.mean(metrics["response_lengths"]) if metrics["response_lengths"] else 0
    
    # Print evaluation results
    print(f"\nSuccess rate: {metrics['success_rate']*100:.2f}%")
    print(f"Average BLEU: {metrics['avg_bleu']:.4f}")
    print(f"Average ROUGE-1: {metrics['avg_rouge1']:.4f}")
    print(f"Average ROUGE-2: {metrics['avg_rouge2']:.4f}")
    print(f"Average ROUGE-L: {metrics['avg_rougeL']:.4f}")
    print(f"Average response length: {metrics['avg_response_length']:.1f} chars")
    
    return metrics


def run_inference_and_evaluation(model_path: str = "../model/Qwen2.5-3B-Instruct",
                                 test_data_path: str = "../data/processed/medical_meadow/test.json",
                                 output_dir: str = "../output",
                                 batch_size: int = 4,
                                 num_samples: int = None,
                                 num_gpus: int = None,
                                 gpu_memory: str = "10GiB",
                                 cpu_memory: str = "30GiB"):
    """Run inference and evaluation"""
    import time
    from datetime import datetime
    
    print("\nMedical Dialogue Model Inference & Evaluation")
    print("-" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(output_dir) / "logs" / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Log directory: {log_dir}")
    
    # Initialize model
    print("\n1. Loading model...")
    start_time = time.time()
    chatbot = MedicalChatbot(
        model_name=model_path,
        num_gpus=num_gpus,
        gpu_memory_per_device=gpu_memory,
        cpu_memory=cpu_memory
    )
    model_load_time = time.time() - start_time
    print(f"   Model load time: {model_load_time:.2f}s")
    
    # Load test data
    print("\n2. Loading test data...")
    test_path = Path(test_data_path)
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        print("Please run 1_process_dataset.py first")
        return
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if num_samples:
        test_data = test_data[:num_samples]
    
    print(f"   Test samples: {len(test_data)}")
    
    # Batch inference
    print("\n3. Running inference...")
    start_time = time.time()
    results_path = log_dir / "inference_results.json"
    results = chatbot.batch_inference(
        test_data,
        batch_size=batch_size,
        save_path=str(results_path)
    )
    inference_time = time.time() - start_time
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Average per sample: {inference_time/len(test_data):.2f}s")
    
    # Evaluation
    print("\n4. Evaluating results...")
    metrics = evaluate_responses(results)
    
    # Save evaluation results
    eval_results = {
        "timestamp": timestamp,
        "model_path": model_path,
        "test_data_path": test_data_path,
        "num_samples": len(test_data),
        "batch_size": batch_size,
        "model_load_time": model_load_time,
        "inference_time": inference_time,
        "avg_time_per_sample": inference_time / len(test_data),
        "metrics": {
            "success_rate": metrics["success_rate"],
            "avg_bleu": metrics["avg_bleu"],
            "avg_rouge1": metrics["avg_rouge1"],
            "avg_rouge2": metrics["avg_rouge2"],
            "avg_rougeL": metrics["avg_rougeL"],
            "avg_response_length": metrics["avg_response_length"]
        }
    }
    
    eval_path = log_dir / "evaluation.json"
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to: {eval_path}")
    
    # Show sample results
    print("\nSample Results (first 3)")
    print("-" * 60)
    for i, result in enumerate(results[:3], 1):
        print(f"\nSample {i}:")
        print(f"Question: {result['messages'][-1]['content'][:80]}...")
        print(f"Prediction: {result['prediction'][:150]}...")
        if result.get('ground_truth'):
            print(f"Ground truth: {result['ground_truth'][:150]}...")
        print("-" * 60)
    
    print("\nInference and evaluation complete!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Dialogue Model Inference & Evaluation")
    parser.add_argument("--model_path", type=str, default="../model/Qwen2.5-3B-Instruct",
                       help="Model path")
    parser.add_argument("--test_data", type=str, default="../data/processed/medical_meadow/test.json",
                       help="Test data path")
    parser.add_argument("--output_dir", type=str, default="../output",
                       help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of test samples (default: all)")
    parser.add_argument("--num_gpus", type=int, default=None,
                       help="Number of GPUs to use (default: min(2, available))")
    parser.add_argument("--gpu_memory", type=str, default="10GiB",
                       help="Max memory per GPU (default: 10GiB)")
    parser.add_argument("--cpu_memory", type=str, default="30GiB",
                       help="Max CPU memory (default: 30GiB)")
    
    args = parser.parse_args()
    
    run_inference_and_evaluation(
        model_path=args.model_path,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        num_gpus=args.num_gpus,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory
    )


if __name__ == "__main__":
    main()
