import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import TRAIN_DATA_PATH
from generate import CaptchaGenerator
from codec import encode
from tqdm import tqdm


def transform(image_height, image_width):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert RGB to grayscale
        transforms.Resize((image_height, image_width)),
        transforms.ToTensor()
    ])


class CaptchaDataset(Dataset):
    """
    Training dataset for CAPTCHA images.

    Args:
        size (int): Number of samples to generate.
        cache (bool):
            If True, generate and store images/labels in memory (RAM).
            If False, generate images on disk (disk-based).
        load_only (bool):
            If True, skip generation and load existing images in the folder.
    """
    def __init__(self, size=10000, cache=False, load_only=False, train_data_path=TRAIN_DATA_PATH, config=None):
        self.size = size
        self.cache = cache

        self.character_set = config.character_set
        self.character_length = config.character_length
        self.image_width = config.image_width
        self.image_height = config.image_height
        self.train_data_path = train_data_path

        self.data = []
        self.labels = []

        generator = CaptchaGenerator(character_set=self.character_set, character_length=self.character_length,
                                     width=self.image_width, height=self.image_height)

        print("Starting dataset generation...")

        if load_only:
            print(f"Loading existing CAPTCHA images from folder {self.train_data_path}...")
            self.image_files = sorted(self.train_data_path.glob("*.png"))
            self.size = len(self.image_files)
            return  # skip generation
        else:
            if cache:
                print(f"Generating {size} samples in memory...")
                for _ in tqdm(range(size), desc="🧠 Generating (cache mode)"):
                    img, text = generator.generate_image_and_text()
                    self.data.append(img)
                    self.labels.append(text)
            else:
                self.train_data_path.mkdir(parents=True, exist_ok=True)
                existing = set(os.listdir(self.train_data_path))

                count = 0
                print(f"Generating {size} samples on disk...")
                with tqdm(total=size, desc="💽 Generating (disk mode)") as pbar:
                    while count < size:
                        img, text = generator.generate_image_and_text()
                        filename = f"{text}.png"
                        if filename in existing:
                            continue
                        img.save(self.train_data_path / filename)
                        existing.add(filename)
                        count += 1
                        pbar.update(1)

                self.image_files = sorted(self.train_data_path.glob("*.png"))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # image transform
        compose = transform(image_height=self.image_height, image_width=self.image_width)

        if self.cache:
            img = compose(self.data[idx])
            label = encode(self.labels[idx], self.character_set, self.character_length)
        else:
            img_path = self.image_files[idx]
            path = os.path.join(img_path)
            img = Image.open(path).convert('L')  # Convert to grayscale
            img = compose(img)

            label_text = img_path.stem  # stem Extract filename without extension
            print()
            print(f"🖼️ Loaded image: {img_path.name} / Label: '{label_text}'")
            try:
                label = encode(label_text, self.character_set, self.character_length)
            except Exception as e:
                print(f"❌ Failed to encode label: '{label_text}'")
            label = encode(label_text, self.character_set, self.character_length)
        return img, label
