"""
Dataset Processor Class
Processes medical dialogue datasets
Supports unified processing of single-turn and multi-turn dialogues
"""

import os
import json
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from pathlib import Path


class ClinicalDialogueProcessor:
    """Medical dialogue dataset processor"""
    
    def __init__(self, data_dir: str = "../data", output_dir: str = None):
        """
        Initialize processor
        
        Args:
            data_dir: Data storage directory
            output_dir: Output directory (default: data_dir/processed)
        """
        self.data_dir = Path(data_dir)
        if output_dir is None:
            self.output_dir = self.data_dir / "processed"
        else:
            self.output_dir = Path(output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_turn_qa(self, dataset: Dataset, 
                               question_col: str = "input",
                               answer_col: str = "output",
                               instruction_col: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Process single-turn QA dataset
        
        Args:
            dataset: Original dataset
            question_col: Question column name
            answer_col: Answer column name
            instruction_col: Instruction column name (optional)
        
        Returns:
            Standardized data list
        """
        processed_data = []
        
        for item in dataset:
            processed_item = {
                "instruction": item.get(instruction_col, "") if instruction_col else "",
                "input": str(item.get(question_col, "")),
                "output": str(item.get(answer_col, "")),
                "type": "single_turn"
            }
            processed_data.append(processed_item)
        
        return processed_data
    
    def process_multi_turn_dialogue(self, dataset: Dataset,
                                   dialogue_col: str = "messages") -> List[Dict[str, Any]]:
        """
        Process multi-turn dialogue dataset
        
        Args:
            dataset: Original dataset
            dialogue_col: Dialogue column name
        
        Returns:
            Standardized data list
        """
        processed_data = []
        
        for item in dataset:
            messages = item.get(dialogue_col, [])
            
            # If messages is a string, try to parse
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except:
                    continue
            
            # Standardize message format
            standardized_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("from", ""))
                    content = msg.get("content", msg.get("value", ""))
                    
                    # Unify role names
                    if role in ["human", "user", "patient"]:
                        role = "user"
                    elif role in ["assistant", "gpt", "doctor"]:
                        role = "assistant"
                    
                    if role and content:
                        standardized_messages.append({
                            "role": role,
                            "content": str(content)
                        })
            
            if standardized_messages:
                processed_item = {
                    "messages": standardized_messages,
                    "type": "multi_turn"
                }
                processed_data.append(processed_item)
        
        return processed_data
    
    def convert_to_chat_format(self, data: List[Dict[str, Any]], 
                               model_name: str = "qwen") -> List[Dict[str, Any]]:
        """
        Convert to chat format (for training)
        
        Args:
            data: Processed data
            model_name: Model name (for format determination)
        
        Returns:
            Chat format data
        """
        chat_data = []
        
        for item in data:
            if item["type"] == "single_turn":
                # Convert single-turn to message format
                messages = []
                if item.get("instruction"):
                    messages.append({
                        "role": "system",
                        "content": item["instruction"]
                    })
                messages.append({
                    "role": "user",
                    "content": item["input"]
                })
                messages.append({
                    "role": "assistant",
                    "content": item["output"]
                })
                chat_data.append({"messages": messages})
            
            elif item["type"] == "multi_turn":
                # Use multi-turn directly
                chat_data.append({"messages": item["messages"]})
        
        return chat_data
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str):
        """
        Save data as JSON file
        
        Args:
            data: Data to save
            filename: File name
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved to: {output_path}")
        print(f"  Samples: {len(data)}")
    
    def load_from_json(self, filename: str) -> List[Dict[str, Any]]:
        """
        Load data from JSON file
        
        Args:
            filename: File name
        
        Returns:
            Loaded data
        """
        input_path = self.output_dir / filename
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from {input_path}")
        return data
    
    def split_dataset(self, data: List[Dict[str, Any]], 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split dataset
        
        Args:
            data: Original data
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        
        Returns:
            Split data dictionary
        """
        import random
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        return {
            "train": data[:train_end],
            "validation": data[train_end:val_end],
            "test": data[val_end:]
        }
    
    def get_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get dataset statistics
        
        Args:
            data: Data list
        
        Returns:
            Statistics dictionary
        """
        stats = {
            "total_samples": len(data),
            "single_turn": 0,
            "multi_turn": 0,
            "avg_turns": 0,
            "max_turns": 0,
            "min_turns": float('inf')
        }
        
        turn_counts = []
        for item in data:
            if item.get("type") == "single_turn":
                stats["single_turn"] += 1
                turn_counts.append(1)
            elif item.get("type") == "multi_turn":
                stats["multi_turn"] += 1
                num_turns = len(item.get("messages", [])) // 2
                turn_counts.append(num_turns)
        
        if turn_counts:
            stats["avg_turns"] = sum(turn_counts) / len(turn_counts)
            stats["max_turns"] = max(turn_counts)
            stats["min_turns"] = min(turn_counts)
        
        return stats
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Print statistics"""
        print("\nDataset Statistics")
        print("-" * 50)
        print(f"Total samples: {stats['total_samples']}")
        print(f"Single-turn: {stats['single_turn']}")
        print(f"Multi-turn: {stats['multi_turn']}")
        print(f"Average turns: {stats['avg_turns']:.2f}")
        print(f"Max turns: {stats['max_turns']}")
        print(f"Min turns: {stats['min_turns']}")
        print("-" * 50)
    
    def preview_samples(self, data: List[Dict[str, Any]], n: int = 3):
        """
        Preview samples
        
        Args:
            data: Data list
            n: Number of samples to preview
        """
        print(f"\nSample Preview (first {n})")
        print("-" * 50)
        
        for i, item in enumerate(data[:n], 1):
            print(f"\nSample {i}:")
            print(json.dumps(item, ensure_ascii=False, indent=2))
            print("-" * 50)
