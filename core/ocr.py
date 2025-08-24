import json
from typing import List, Union
import logging

import numpy as np
from PIL import Image
import pdfplumber
import paddle
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_ocr_engine(device: str = None) -> PaddleOCR:
    """
    Initialize a PaddleOCR engine.
    
    Args:
        device: 'cpu' or 'gpu'. If None, auto-detects GPU availability.
        
    Returns:
        PaddleOCR instance.
    """
    if device is None:
        device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    logger.info(f"[OCR] Initializing PaddleOCR on {device.upper()}")

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang='en',
        device=device,
    )


def run_paddleocr(pil_image: Image.Image, ocr_engine: PaddleOCR) -> str:
    """
    Run OCR on a PIL image and return recognized words as JSON.
    
    Args:
        pil_image: PIL.Image.Image to process.
        ocr_engine: Pre-initialized PaddleOCR engine.
        
    Returns:
        JSON string containing OCR words.
    """
    image_np = np.array(pil_image.convert("RGB"), dtype=np.uint8)

    try:
        results = ocr_engine.predict(image_np)
    except Exception as e:
        logger.warning(f"[OCR WARNING] OCR failed on GPU: {e}")
        logger.info("[OCR] Retrying on CPU...")
        cpu_engine = create_ocr_engine(device="cpu")
        results = cpu_engine.predict(image_np)

    rec_texts: List[str] = []
    for res in results:
        rec_texts.extend(res.get("rec_texts", []))

    json_dict = {"ocr_words": rec_texts}
    return json.dumps(json_dict, ensure_ascii=False)


def pdf_to_images(pdf_file: Union[str, bytes]) -> List[Image.Image]:
    """
    Convert a PDF file (path or bytes) to a list of PIL images.
    
    Args:
        pdf_file: File path or bytes-like object representing the PDF.
        
    Returns:
        List of PIL.Image.Image objects, one per page.
    """
    images: List[Image.Image] = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            pil_image = page.to_image(resolution=300).original
            images.append(pil_image)

    logger.info(f"[PDF] Converted {len(images)} pages to images.")
    return images