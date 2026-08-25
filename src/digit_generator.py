from PIL import Image, ImageDraw, ImageFont
import random, os

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf", 
    "/System/Library/Fonts/Helvetica.ttc", 
]

NUM_TO_GENERATE = 10

def random_bg_color():
    # Light background
    r = random.randint(220, 255)
    g = random.randint(220, 255)
    b = random.randint(220, 255)
    return (r, g, b)

def random_text_color():
    # Dark text
    r = random.randint(0, 60)
    g = random.randint(0, 60)
    b = random.randint(0, 60)
    return (r, g, b)

def render_digit(digit: str, font_path: str, size=64):
    img = Image.new("RGB", (size, size), color=random_bg_color())
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=random.randint(28, 44))
    center = (size // 2, size // 2)
    draw.text(
        center,
        digit,
        font=font,
        fill=random_text_color(),
        anchor="mm",
    )
    return img

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

for digit in "0123456789":
    for font in FONTS:
        for id in range(NUM_TO_GENERATE):
            img = render_digit(digit, font)
            save_image(img, f"data/cells/train/digit_{digit}/img_{id}.png")