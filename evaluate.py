import os
from types import SimpleNamespace

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from codec import decode_prediction, decode
from config import (
    BATCH_SIZE, DATESET_SIZE, CHARACTER_LENGTH, CHARACTER_SET, EVAL_DATA_PATH, TRAIN_DATA_PATH,
    EVAL_ACC_THRESHOLD, LOAD_MODEL_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, load_device
)
from dataset import CaptchaDataset
from model import load_model, CNNWithOneHot


def save_images(images, true_str, pred_str, eval_data_path):
    img = images.cpu()
    save_path = os.path.join(eval_data_path, f"{true_str}_{pred_str}.png")
    vutils.save_image(img, save_path)


def evaluate(model, dataloader, device, config):
    model.eval()
    print("Starting model evaluation...")
    correct = 0
    total = 0

    os.makedirs(config.eval_data_path, exist_ok=True)

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            batch_size = outputs.size(0)

            for b in range(batch_size):
                pred = outputs[b]
                pred_str = decode_prediction(pred, config.character_set, config.character_length)

                label_vec = labels[b].cpu().numpy()
                true_str = decode(label_vec, config.character_set)

                mark = "✅" if pred_str == true_str else "❌"
                print(f'{mark} true: {true_str} | pred: {pred_str}')

                if config.save_image:
                    save_images(images[b], true_str, pred_str, config.eval_data_path)

                total += 1
                if pred_str == true_str:
                    correct += 1

                if total % 200 == 0:
                    acc = 100 * (correct / total)
                    print("\n" + "-" * 60)
                    print(f'📈 Evaluated {total} samples — Accuracy: {acc:.2f}%')
                    print("-" * 60)

                    tolerance_threshold = config.eval_acc_threshold - 3
                    if acc < tolerance_threshold:
                        print(f"⚠️ Accuracy is more than 3% below the threshold of 75%,"
                              f" stopping evaluation early.")
                        return acc  # exit evaluate

    final_acc = 100 * (correct / total)
    print(f'\n🎯 Final Accuracy: {final_acc:.2f}% on {total} samples')
    return final_acc


def main(args=None):
    args = args or {}

    load_model_path = getattr(args, 'load_model_path', LOAD_MODEL_PATH)
    train_data_path = getattr(args, 'train_data_path', TRAIN_DATA_PATH)
    eval_data_path = getattr(args, 'eval_data_path', EVAL_DATA_PATH)

    cache_mode = getattr(args, 'cache', True)
    dataset_size = getattr(args, 'dataset_size', DATESET_SIZE)
    batch_size = getattr(args, 'batch_size', BATCH_SIZE)
    character_set = getattr(args, 'character_set', CHARACTER_SET)
    character_length = getattr(args, 'character_length', CHARACTER_LENGTH)
    image_width = getattr(args, 'image_width', IMAGE_WIDTH)
    image_height = getattr(args, 'image_height', IMAGE_HEIGHT)

    eval_acc_threshold = getattr(args, 'eval_acc_threshold', EVAL_ACC_THRESHOLD)
    save_image = getattr(args, 'save_image', False)

    eval_config = SimpleNamespace(
        character_set=character_set,
        character_length=character_length,
        eval_data_path=eval_data_path,
        eval_acc_threshold=eval_acc_threshold,
        save_image=save_image
    )

    captcha_config = SimpleNamespace(
        character_set=character_set,
        character_length=character_length,
        image_width=image_width,
        image_height=image_height
    )

    device = load_device()

    # Load the trained model
    model = load_model(CNNWithOneHot(image_width, image_height, character_length, len(character_set)), load_model_path, device)

    # Load evaluation dataset in memory
    dataset = CaptchaDataset(size=dataset_size, cache=cache_mode, train_data_path=train_data_path, config=captcha_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Run model evaluation
    evaluate(model, dataloader, device, eval_config)


if __name__ == "__main__":
    from main import evaluate
    evaluate()
