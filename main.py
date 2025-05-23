import json
import os
from pathlib import Path
from types import SimpleNamespace

import click
from config import (
    SAVE_MODEL_PATH, LOAD_MODEL_PATH, TRAIN_DATA_PATH, EVAL_DATA_PATH,
    IMAGE_WIDTH, IMAGE_HEIGHT, CHARACTER_LENGTH, CHARACTER_SET, EPOCHS,
    BATCH_SIZE, LEARNING_RATE, DATESET_SIZE, EVAL_ACC_THRESHOLD, PRED_DATA_PATH
)
from train import training as train_main
from predict import main as predict_main
from evaluate import main as evaluate_main
from generate import main as generate_main


@click.group()
def cli():
    """ CLI tool for OCR captcha."""
    pass


@cli.command(
    help="🚀 Train the CAPTCHA model using given training and evaluation datasets.\n\n"
         "Options allow controlling training behavior such as cache usage, number of epochs, batch size, learning rate, etc."
)
@click.option('--cache/--no-cache', default=False, help="Cache dataset in memory (default: disabled). If --no-cache is used, the dataset will be stored on disk instead of in memory.")
@click.option('--load-only', is_flag=True, default=False, help="Only load existing dataset from disk without generating new samples")
@click.option('--epochs', default=EPOCHS, show_default=True, type=int, help="Number of training epochs")
@click.option('--batch-size', default=BATCH_SIZE, show_default=True, type=int, help="Batch size for training")
@click.option('--learning-rate', default=LEARNING_RATE, show_default=True, type=float, help="Learning rate for model optimization")
@click.option('--dataset-size', default=DATESET_SIZE, show_default=True, type=int, help="Number of training samples in the dataset")
@click.option('--save-model-path', default=SAVE_MODEL_PATH, show_default=True, type=click.Path(), help="Path to save the trained model")
@click.option('--character-set', default=CHARACTER_SET, show_default=True, type=str, help="Character set used for captcha training")
@click.option('--character-length', default=CHARACTER_LENGTH, show_default=True, type=int, help="Length of captcha text to recognize")
@click.option('--image-width', default=IMAGE_WIDTH, show_default=True, type=int, help="Width of input image")
@click.option('--image-height', default=IMAGE_HEIGHT, show_default=True, type=int, help="Height of input image")
@click.option('--train-data-path', default=TRAIN_DATA_PATH, show_default=True, type=click.Path(), help="Path to the training data")
@click.option('--eval-data-path', default=EVAL_DATA_PATH, show_default=True, type=click.Path(), help="Path to the evaluation data")
@click.option('--eval-save-image', is_flag=False, help="Save evaluation result images after training (default: False)")
@click.option('--eval-acc-threshold', default=EVAL_ACC_THRESHOLD, show_default=True, type=int, help="Accuracy threshold for saving the best model")
def train(
        cache, load_only, epochs, batch_size, learning_rate, dataset_size,
        save_model_path, character_set, character_length,
        image_width, image_height, train_data_path, eval_data_path,
        eval_save_image, eval_acc_threshold
):
    """🚀 Train the model"""
    args = SimpleNamespace(
        cache=cache,
        load_only=load_only,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dataset_size=dataset_size,
        save_model_path=save_model_path,
        character_set=character_set,
        character_length=character_length,
        image_width=image_width,
        image_height=image_height,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        eval_save_image=eval_save_image,
        eval_acc_threshold=eval_acc_threshold
    )
    print(f"🚀 Training started with: \n{json.dumps(vars(args), indent=4, cls=PathEncoder)}")
    train_main(args)


@cli.command(
    help="🔮 Predict CAPTCHA text from a single image file or a folder containing multiple images.\n\n"
         "Provide the path as the first argument. It accepts either a file path or a directory path."
)
@click.argument('path', type=click.Path(exists=True))
@click.option('--model-path', default=LOAD_MODEL_PATH, show_default=True, type=click.Path(), help="Path to load model")
@click.option('--character-set', default=CHARACTER_SET, show_default=True, type=str, help="Character set for CAPTCHA")
@click.option('--character-length', default=CHARACTER_LENGTH, show_default=True, type=int, help="Character length of CAPTCHA")
@click.option('--image-width', default=IMAGE_WIDTH, show_default=True, type=int, help="Width of input images (should match training image width)")
@click.option('--image-height', default=IMAGE_HEIGHT, show_default=True, type=int, help="Height of input images (should match training image height)")
def predict(path, model_path, character_set, character_length, image_width, image_height):
    """🔮 Predict on a specific image"""
    print(f"🔮 Prediction started for: \n\t{path}")
    args = SimpleNamespace(
        model_path=model_path,
        character_set=character_set,
        character_length=character_length,
        image_width=image_width,
        image_height=image_height,
    )

    if os.path.isdir(path):
        args.folder_path = path
        args.image_path = None
    else:
        args.image_path = path
        args.folder_path = None
    predict_main(args)


@cli.command(
    help="📊 Evaluate a trained model using a test dataset.\n\n"
         "Options allow adjusting batch size, image dimensions, and accuracy threshold, etc."
)
@click.option('--cache/--no-cache', default=False, help="Cache dataset in memory (default: disabled). If --no-cache is used, the dataset will be stored on disk instead of in memory.")
@click.option('--load-only', is_flag=True, default=False, help="Only load existing dataset from disk without generating new samples")
@click.option('--dataset-size', default=DATESET_SIZE, show_default=True, type=int, help="Number of samples for evaluation")
@click.option('--batch-size', default=BATCH_SIZE, show_default=True, type=int, help="Batch size for evaluation")
@click.option('--character-set', default=CHARACTER_SET, show_default=True, type=str, help="Character set for CAPTCHA")
@click.option('--character-length', default=CHARACTER_LENGTH, show_default=True, type=int, help="Character length of CAPTCHA")
@click.option('--image-width', default=IMAGE_WIDTH, show_default=True, type=int, help="Width of input images (should match training image width)")
@click.option('--image-height', default=IMAGE_HEIGHT, show_default=True, type=int, help="Height of input images (should match training image height)")
@click.option('--save-image', is_flag=False, help="Save prediction result images to disk")
@click.option('--eval-acc-threshold', default=EVAL_ACC_THRESHOLD, show_default=True, type=int, help="Override evaluation accuracy threshold")
@click.option('--model-path', default=LOAD_MODEL_PATH, show_default=True, type=click.Path(), help="Path to load model")
@click.option('--train-data-path', default=TRAIN_DATA_PATH, show_default=True, type=click.Path(), help="Path to the training data")
@click.option('--eval-data-path', default=EVAL_DATA_PATH, show_default=True, type=click.Path(), help="Path to the evaluation data")
def evaluate(
        cache, load_only, dataset_size, batch_size, character_set, character_length,
        image_width, image_height, save_image, eval_acc_threshold,
        model_path, train_data_path, eval_data_path
):
    """📊 Evaluate the model"""
    args = SimpleNamespace(
        cache=cache,
        load_only=load_only,
        dataset_size=dataset_size,
        batch_size=batch_size,
        character_set=character_set,
        character_length=character_length,
        image_width=image_width,
        image_height=image_height,
        save_image=save_image,
        eval_acc_threshold=eval_acc_threshold,
        load_model_path=model_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path
    )

    print(f"📊 Evaluation started with: \n{json.dumps(vars(args), indent=4, cls=PathEncoder)}")
    evaluate_main(args)


@cli.command(
    help="🧪 Generate CAPTCHA image(s) for testing or training.\n\n"
         "You can customize the number of images, character set, image size, and output path."
)
@click.option('--count', default=DATESET_SIZE, show_default=True, type=int, help="Number of CAPTCHA images to generate")
@click.option('--save-path', default=PRED_DATA_PATH, show_default=True, type=click.Path(), help="Directory path to save generated images")
@click.option('--character-set', default=CHARACTER_SET, show_default=True, type=str, help="Character set used in CAPTCHA")
@click.option('--character-length', default=CHARACTER_LENGTH, show_default=True, type=int, help="Number of characters in each CAPTCHA")
@click.option('--width', default=IMAGE_WIDTH, show_default=True, type=int, help="Width of the generated images")
@click.option('--height', default=IMAGE_HEIGHT, show_default=True, type=int, help="Height of the generated images")
@click.option('--custom/--no-custom', default=False, show_default=True, help="Enable or disable custom CAPTCHA rendering settings (default: disabled)")
@click.option('--fonts', callback=lambda ctx, param, value: value.split() if value else [], help='Font file paths (randomly chosen per character in each image), separated by space. e.g.: --fonts ./fonts/1.ttf ./fonts/2.ttf')
@click.option('--font-sizes', callback=lambda ctx, param, value: list(map(int, value.split())) if value else [], help='Font sizes (randomly chosen per character in each image), separated by space. e.g. --font-sizes 42, 50, 56')
@click.option('--bg-color', type=(int, int, int), help="Background color in RGB, e.g., 255 255 0")
@click.option('--fg-color', type=(int, int, int), help="Foreground (text) color in RGB, e.g., 255 0 0")
@click.option('--character-offset-dx', type=(int, int), help="Character X offset range (e.g., 0 4)")
@click.option('--character-offset-dy', type=(int, int), help="Character Y offset range (e.g., 0 6)")
@click.option('--character-rotate', type=(int, int), help="Rotation angle range (e.g., -30 30)")
@click.option('--character-warp-dx', type=(float, float), help="Horizontal warp range (e.g., 0.1 0.3)")
@click.option('--character-warp-dy', type=(float, float), help="Vertical warp range (e.g., 0.2 0.3)")
@click.option('--word-space-probability', type=float, help="Probability to add space between characters (e.g., 0.5)")
@click.option('--word-offset-dx', type=float, help="Word horizontal offset factor (e.g., 0.25)")
def generate(count, save_path, character_set, character_length, width, height,
             custom, fonts, font_sizes, bg_color, fg_color,
             character_offset_dx, character_offset_dy, character_rotate,
             character_warp_dx, character_warp_dy, word_space_probability, word_offset_dx):
    """🧪 Generate CAPTCHA image(s)"""
    args = SimpleNamespace(
        count=count,
        character_set=character_set,
        character_length=character_length,
        image_width=width,
        image_height=height,
        pred_data_path=save_path,
        custom=custom,
        fonts=fonts,
        font_sizes=font_sizes,
        bg_color=bg_color,
        fg_color=fg_color,
        character_offset_dx=character_offset_dx,
        character_offset_dy=character_offset_dy,
        character_rotate=character_rotate,
        character_warp_dx=character_warp_dx,
        character_warp_dy=character_warp_dy,
        word_space_probability=word_space_probability,
        word_offset_dx=word_offset_dx
    )

    generate_main(args)


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


if __name__ == "__main__":
    cli()
