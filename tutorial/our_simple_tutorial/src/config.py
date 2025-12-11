"""
Configuration File
Centralized management of all project configuration parameters
"""

import os
from pathlib import Path

# Path Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = OUTPUT_DIR / "logs"

# Create necessary directories
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# GPU Configuration
MAX_GPUS = 2
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(MAX_GPUS)))

# Model Configuration
BASE_MODEL_NAME = str(MODEL_DIR / "Qwen2.5-3B-Instruct")
SFT_MODEL_DIR = OUTPUT_DIR / "sft_model"
GRPO_MODEL_DIR = OUTPUT_DIR / "grpo_model"

# Dataset Configuration
SINGLE_TURN_DATASET = "medalpaca/medical_meadow_medical_flashcards"
MULTI_TURN_DATASET = "lavita/ChatDoctor-HealthCareMagic-100k"
MULTI_TURN_DATASET_BACKUP = "ruslanmv/ai-medical-chatbot"
DATASET_SIZE_LIMIT = 1000
DATA_SAMPLE_RATIO = 0.1

# Training Configuration
SFT_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 0.3,
    "use_4bit": True,
    "use_lora": True,
}

GRPO_CONFIG = {
    "num_epochs": 2,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "max_seq_length": 512,
    "gradient_accumulation_steps": 4,
    "warmup_ratio": 0.1,
    "max_grad_norm": 0.3,
    "num_generations": 4,
    "temperature": 0.7,
    "top_p": 0.9,
    "use_4bit": True,
    "use_lora": True,
}

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
}

# Inference Configuration
INFERENCE_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

# Data Processing Configuration
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Other Configuration
RANDOM_SEED = 42
NUM_PREVIEW_SAMPLES = 3
USE_FP16 = True
USE_GRADIENT_CHECKPOINTING = True


if __name__ == "__main__":
    """Test configuration"""
    print("Project Configuration")
    print("-" * 50)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Model directory: {MODEL_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Max GPUs: {MAX_GPUS}")
