import os
import random
from tqdm import tqdm

from captcha.image import ImageCaptcha

from config import (
    CHARACTER_SET, CHARACTER_LENGTH,
    IMAGE_WIDTH, IMAGE_HEIGHT, PRED_DATA_PATH, DATESET_SIZE
)


class CaptchaGenerator:
    def __init__(self, character_set, character_length, width=160, height=60, captcha_kwargs=None):
        self.character_set = character_set
        self.character_length = character_length
        self.width = width
        self.height = height

        captcha_kwargs = captcha_kwargs or {}
        fonts = captcha_kwargs.get('fonts')
        font_sizes = captcha_kwargs.get('font_sizes')
        self.bg_color = captcha_kwargs.get('bg_color')
        self.fg_color = captcha_kwargs.get('fg_color')

        self.image_captcha = ImageCaptcha(
            width=self.width,
            height=self.height,
            fonts=list(fonts) if fonts else None,
            font_sizes=tuple(font_sizes) if font_sizes else None
        )

        for attr in [
            'character_offset_dx', 'character_offset_dy',
            'character_rotate', 'character_warp_dx', 'character_warp_dy',
            'word_space_probability', 'word_offset_dx'
        ]:
            value = captcha_kwargs.get(attr)
            if value is not None:
                setattr(self.image_captcha, attr, value)

    def generate_text(self):
        """Generate a random CAPTCHA text"""
        return ''.join(random.choices(self.character_set, k=self.character_length))

    def generate_image(self, text):
        """Generate a CAPTCHA image for the given text"""
        if self.bg_color is not None and self.fg_color is not None:
            return self.image_captcha.generate_image(text, self.bg_color, self.fg_color)

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


def load_captcha_style_config(args):
    from config import (
        BG_COLOR, FG_COLOR,
        CHARACTER_OFFSET_DX, CHARACTER_OFFSET_DY,
        CHARACTER_ROTATE,
        CHARACTER_WARP_DX, CHARACTER_WARP_DY,
        WORD_SPACE_PROBABILITY, WORD_OFFSET_DX
    )

    def get_or_none(name, default):
        return getattr(args, name, default)

    return {
        'fonts': get_or_none('fonts', None),
        'font_sizes': get_or_none('font_sizes', None),
        'bg_color': get_or_none('bg_color', BG_COLOR),
        'fg_color': get_or_none('fg_color', FG_COLOR),
        'character_offset_dx': get_or_none('character_offset_dx', CHARACTER_OFFSET_DX),
        'character_offset_dy': get_or_none('character_offset_dy', CHARACTER_OFFSET_DY),
        'character_rotate': get_or_none('character_rotate', CHARACTER_ROTATE),
        'character_warp_dx': get_or_none('character_warp_dx', CHARACTER_WARP_DX),
        'character_warp_dy': get_or_none('character_warp_dy', CHARACTER_WARP_DY),
        'word_space_probability': get_or_none('word_space_probability', WORD_SPACE_PROBABILITY),
        'word_offset_dx': get_or_none('word_offset_dx', WORD_OFFSET_DX),
    }


def main(args=None):
    args = args or {}

    count = getattr(args, 'count', DATESET_SIZE)
    character_set = getattr(args, 'character_set', CHARACTER_SET)
    character_length = getattr(args, 'character_length', CHARACTER_LENGTH)
    image_width = getattr(args, 'width', IMAGE_WIDTH)
    image_height = getattr(args, 'height', IMAGE_HEIGHT)
    pred_data_path = getattr(args, 'save_path', PRED_DATA_PATH)

    custom_captcha = getattr(args, 'custom', False)
    captcha_style_config = load_captcha_style_config(args) if custom_captcha else {}

    generator = CaptchaGenerator(
        character_set=character_set,
        character_length=character_length,
        width=image_width,
        height=image_height,
        captcha_kwargs=captcha_style_config
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
