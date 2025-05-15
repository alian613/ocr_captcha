from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, EVAL_ACC_THRESHOLD, CHARACTER_SET, CHARACTER_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT,
    DATESET_SIZE, SAVE_MODEL_PATH, EVAL_DATA_PATH, TRAIN_DATA_PATH, load_device
)
from dataset import CaptchaDataset
from evaluate import evaluate
from model import save_model, CNNWithOneHot


def training(args=None):
    args = args or {}

    epochs = getattr(args, 'epochs', EPOCHS)
    dataset_size = getattr(args, 'dataset_size', DATESET_SIZE)
    learning_rate = getattr(args, 'learning_rate', LEARNING_RATE)
    batch_size = getattr(args, 'batch_size', BATCH_SIZE)
    cache_mode = getattr(args, 'cache', True)
    load_only = getattr(args, 'load_only', False)

    character_set = getattr(args, 'character_set', CHARACTER_SET)
    character_length = getattr(args, 'character_length', CHARACTER_LENGTH)
    image_width = getattr(args, 'image_width', IMAGE_WIDTH)
    image_height = getattr(args, 'image_height', IMAGE_HEIGHT)

    save_model_path = getattr(args, 'save_model_path', SAVE_MODEL_PATH)
    train_data_path = getattr(args, 'train_data_path', TRAIN_DATA_PATH)
    eval_data_path = getattr(args, 'eval_data_path', EVAL_DATA_PATH)
    eval_save_image = getattr(args, 'eval_save_image', SAVE_MODEL_PATH)
    eval_acc_threshold = getattr(args, 'eval_acc_threshold', EVAL_ACC_THRESHOLD)

    char2idx = {c: i + 1 for i, c in enumerate(character_set)}
    num_classes = len(char2idx)

    captcha_config = SimpleNamespace(
        character_set=character_set,
        character_length=character_length,
        image_width=image_width,
        image_height=image_height
    )

    eval_config = SimpleNamespace(
        character_set=character_set,
        character_length=character_length,
        eval_data_path=eval_data_path,
        eval_acc_threshold=eval_acc_threshold,
        save_image=eval_save_image
    )

    # Detect and use GPU if available
    device = load_device()

    # Initialize model
    model = CNNWithOneHot(image_width, image_height, character_length, num_classes).to(device)
    print("Model is on:", next(model.parameters()).device)
    model.train()

    # Training components
    criterion = nn.MultiLabelSoftMarginLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load training dataset
    dataset = CaptchaDataset(size=dataset_size, cache=cache_mode, load_only=load_only, train_data_path=train_data_path, config=captcha_config)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())

    max_eval_acc = -1

    # Training loop
    for epoch in range(epochs):
        total_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Training Epoch [{epoch + 1}/{epochs}]", leave=False):
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"⏳ Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss:.4f}")

        eval_acc = evaluate(model, dataloader, device, eval_config)
        if eval_acc > max_eval_acc:
            max_eval_acc = eval_acc
            save_model(model, save_model_path)
            print("🏆 Saved best model!")

        if max_eval_acc == 100:
            break

    print(f"🏁 Training finished! Model Accuracy: {max_eval_acc:.2f}%")
    print(f"📂 Check model at: {save_model_path}")


if __name__ == '__main__':
    from main import train
    train()
