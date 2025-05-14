import os
import random
from tqdm import tqdm

from captcha.image import ImageCaptcha

from config import CHARACTER_SET, CHARACTER_LENGTH, IMAGE_WIDTH, IMAGE_HEIGHT, PRED_DATA_PATH, DATESET_SIZE


class CaptchaGenerator:
    def __init__(self, character_set, character_length, width=160, height=60):
        self.character_set = character_set
        self.character_length = character_length
        self.width = width
        self.height = height
        self.image_captcha = ImageCaptcha(width=self.width, height=self.height)

    def generate_text(self):
        """Generate a random CAPTCHA text"""
        return ''.join(random.choices(self.character_set, k=self.character_length))

    def generate_image(self, text):
        """Generate a CAPTCHA image for the given text"""
        return self.image_captcha.generate_image(text)

    def generate_image_and_text(self):
        """Generate a CAPTCHA image and its corresponding text (not saved)"""
        text = self.generate_text()
        image = self.generate_image(text)
        return image, text

    def generate_dataset(self, count, save_path):
        """
        Generate a dataset of CAPTCHA images and save them (avoid duplicates).
        :param count: Number of images to generate
        :param save_path: Directory to save generated images
        """
        os.makedirs(save_path, exist_ok=True)
        generated = set()

        with tqdm(total=count, desc="Generating CAPTCHA...", unit="img") as pbar:
            while len(generated) < count:
                text = self.generate_text()
                if text in generated:
                    continue  # Avoid duplicate CAPTCHA
                generated.add(text)

                image = self.generate_image(text)
                image.save(os.path.join(save_path, f"{text}.png"))

                pbar.update(1)


def main(args=None):
    args = args or {}

    count = getattr(args, 'count', DATESET_SIZE)
    character_set = getattr(args, 'character_set', CHARACTER_SET)
    character_length = getattr(args, 'character_length', CHARACTER_LENGTH)
    image_width = getattr(args, 'width', IMAGE_WIDTH)
    image_height = getattr(args, 'height', IMAGE_HEIGHT)
    pred_data_path = getattr(args, 'save_path', PRED_DATA_PATH)

    generator = CaptchaGenerator(
        character_set=character_set,
        character_length=character_length,
        width=image_width,
        height=image_height
    )

    if count == 1:
        # Single image generation
        image, text = generator.generate_image_and_text()
        os.makedirs(pred_data_path, exist_ok=True)
        image.save(os.path.join(pred_data_path, f"{text}.png"))
        print(f"🖼️ Successfully generated one CAPTCHA: {text}")
        print(f"📂 Check captcha at: {pred_data_path}")
    else:
        # Batch image generation
        print(f"Generating {count} CAPTCHA images...")
        generator.generate_dataset(count=count, save_path=pred_data_path)
        print(f"🖼️ Successfully generated {count} CAPTCHA")
        print(f"📂 Check captcha at: {pred_data_path}")


if __name__ == '__main__':
    from main import generate
    generate()
