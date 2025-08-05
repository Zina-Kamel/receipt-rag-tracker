import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import pdfplumber
import requests
import tempfile
import json
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(pil_img)
    img = img.convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    img_np = np.array(img)
    img_np = cv2.fastNlMeansDenoising(img_np, h=10)
    _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(img_np)

def ocr_space_file(pil_img: Image.Image, overlay=False, api_key='helloworld', language='eng') -> str:
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
        pil_img.save(tmp.name)
        payload = {
            'isOverlayRequired': overlay,
            'apikey': api_key,
            'language': language,
        }
        with open(tmp.name, 'rb') as f:
            r = requests.post(
                'https://api.ocr.space/parse/image',
                files={tmp.name: f},
                data=payload,
            )
    result = json.loads(r.content.decode())
    parsed = result.get("ParsedResults", [{}])[0].get("ParsedText", "")
    return parsed

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')

def run_paddleocr(pil_image: Image.Image) -> str:
    image_np = np.array(pil_image.convert("RGB"))
    results = ocr_engine.ocr(image_np)
    text_lines = [line[1][0] for result in results for line in result]
    return "\n".join(text_lines)


def pdf_to_images(pdf_path: str) -> list[Image.Image]:
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pil_image = page.to_image(resolution=300).original
            images.append(pil_image)
    return images
