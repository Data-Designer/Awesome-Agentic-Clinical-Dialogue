"""
Dataset Download Script
Downloads medical dialogue datasets including single-turn QA and multi-turn conversations
"""

import os
from datasets import load_dataset
from pathlib import Path


def download_single_turn_dataset():
    """Download single-turn QA dataset"""
    print("\nDownloading single-turn QA dataset: medical_meadow_medical_flashcards")
    
    try:
        # Download first 1000 samples as example
        dataset = load_dataset(
            "medalpaca/medical_meadow_medical_flashcards",
            split="train[:1000]"
        )
        
        print(f"Successfully downloaded: {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")
        print(f"Example: {dataset[0]}")
        
        # Save to local
        save_dir = Path("../data/raw/medical_meadow_medical_flashcards")
        save_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_dir))
        print(f"Saved to: {save_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"Failed to download single-turn dataset: {e}")
        print("Please check network connection or HuggingFace mirror settings")
        return None


def download_multi_turn_dataset():
    """Download multi-turn dialogue dataset"""
    print("\nDownloading multi-turn dataset: ChatDoctor-HealthCareMagic-100k")
    
    try:
        # Download first 500 samples
        dataset = load_dataset(
            "lavita/ChatDoctor-HealthCareMagic-100k",
            split="train[:500]"
        )
        
        print(f"Successfully downloaded: {len(dataset)} samples")
        print(f"Columns: {dataset.column_names}")
        print(f"Example: {dataset[0]}")
        
        # Save to local
        save_dir = Path("../data/raw/ChatDoctor-HealthCareMagic")
        save_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_dir))
        print(f"Saved to: {save_dir}")
        
        return dataset
        
    except Exception as e:
        print(f"Failed to download multi-turn dataset: {e}")
        print("Trying backup dataset...")
        
        # Backup: smaller dataset
        try:
            dataset = load_dataset(
                "ruslanmv/ai-medical-chatbot",
                split="train[:500]"
            )
            
            print(f"Successfully downloaded backup: {len(dataset)} samples")
            print(f"Columns: {dataset.column_names}")
            
            # Save to local
            save_dir = Path("../data/raw/ai-medical-chatbot")
            save_dir.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(save_dir))
            print(f"Saved to: {save_dir}")
            
            return dataset
            
        except Exception as e2:
            print(f"Backup also failed: {e2}")
            return None


def main():
    """Main function"""
    print("\nMedical Dialogue Dataset Download Tool")
    print("-" * 50)
    
    # Download single-turn QA dataset
    single_turn_data = download_single_turn_dataset()
    
    # Download multi-turn dialogue dataset
    multi_turn_data = download_multi_turn_dataset()
    
    if single_turn_data is None and multi_turn_data is None:
        print("\nAll downloads failed.")
    else:
        print("\nDataset download complete!")
        print("Next: Run 1_process_dataset.py to process the data")


if __name__ == "__main__":
    main()
