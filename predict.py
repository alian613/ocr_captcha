import os
import torch
from codec import decode_prediction
from config import (
    CHARACTER_LENGTH, CHARACTER_SET, LOAD_MODEL_PATH, load_device, IMAGE_WIDTH, IMAGE_HEIGHT
)
from dataset import transform
from model import load_model, CNNWithOneHot
from PIL import Image


def _predict_from_image(image: Image.Image, model, device, character_set, character_length) -> str:
    image = image.convert("L")
    width, height = image.size
    compose = transform(image_height=height, image_width=width)
    image = compose(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)[0]
        pred_str = decode_prediction(output, character_set, character_length)
    return pred_str


def predict_image(image_path, model, device, character_set, character_length) -> str:
    image = Image.open(image_path)
    return _predict_from_image(image, model, device, character_set, character_length)


def recognize(image: Image.Image) -> str:
    device = load_device()
    model = load_model(
        CNNWithOneHot(IMAGE_WIDTH, IMAGE_HEIGHT, CHARACTER_LENGTH, len(CHARACTER_SET)),
        LOAD_MODEL_PATH,
        device
    )
    return _predict_from_image(image, model, device, CHARACTER_SET, CHARACTER_LENGTH)


def main(args=None):
    character_set = getattr(args, 'character_set', CHARACTER_SET)
    character_length = getattr(args, 'character_length', CHARACTER_LENGTH)
    image_width = getattr(args, 'image_width', IMAGE_WIDTH)
    image_height = getattr(args, 'image_height', IMAGE_HEIGHT)
    model_path = getattr(args, 'model_path', LOAD_MODEL_PATH)
    image_path = getattr(args, 'image_path', None)
    folder_path = getattr(args, 'folder_path', None)

    if not image_path and not folder_path:
        print("❗ Please provide either an image path or a folder path.")
        return

    device = load_device()
    model = load_model(CNNWithOneHot(image_width, image_height, character_length, len(character_set)), model_path, device)

    if image_path:
        pred = predict_image(image_path, model, device, character_set, character_length)
        print(f"🖼️  {os.path.basename(image_path)} → Prediction: {pred}")

    elif folder_path:
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        for filename in files:
            fpath = os.path.join(folder_path, filename)
            pred = predict_image(fpath, model, device, character_set, character_length)
            print(f"🖼️  {filename} → Prediction: {pred}")


if __name__ == "__main__":
    from main import predict
    predict()
