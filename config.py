from pathlib import Path
import yaml
import torch


"""
Load Configuration and Set Global Parameters
"""


def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_device():
    """
    Detect and print the available computation device (GPU or CPU).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    return device


# Load configuration from file
config = load_config()

# Define paths from configuration
SAVE_MODEL_PATH = Path(config['paths']['save_model'])
LOAD_MODEL_PATH = Path(config['paths']['load_model'])
TRAIN_DATA_PATH = Path(config['paths']['train_data'])
EVAL_DATA_PATH = Path(config['paths']['eval_data'])
PRED_DATA_PATH = Path(config['paths']['pred_data'])


# Image settings
IMAGE_WIDTH = config['image']['width']
IMAGE_HEIGHT = config['image']['height']


# Character set and mappings
CHARACTER_LENGTH = config['characters']['length']
CHARACTER_SET = config['characters']['set']

# Training settings
BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']
DATESET_SIZE = config['training']['dataset_size']

# Evaluation settings
EVAL_ACC_THRESHOLD = config['evaluation']['acc_threshold']

# Captcha style
BG_COLOR = config['captcha']['style']['bg_color']
FG_COLOR = config['captcha']['style']['fg_color']
CHARACTER_OFFSET_DX = config['captcha']['style']['character_offset_dx']
CHARACTER_OFFSET_DY = config['captcha']['style']['character_offset_dy']
CHARACTER_ROTATE = config['captcha']['style']['character_rotate']
CHARACTER_WARP_DX = config['captcha']['style']['character_warp_dx']
CHARACTER_WARP_DY = config['captcha']['style']['character_warp_dy']
WORD_SPACE_PROBABILITY = config['captcha']['style']['word_space_probability']
WORD_OFFSET_DX = config['captcha']['style']['word_offset_dx']


