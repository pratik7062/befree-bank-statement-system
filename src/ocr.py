import pdfplumber
import pytesseract
from PIL import Image

def ocr_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            image = page.to_image(resolution=250).original
            texts.append(pytesseract.image_to_string(image, config="--psm 6"))
    return "\n\n".join(texts)
