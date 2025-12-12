"""
File processing tools for various formats.
Supports PDF, Excel, CSV, images with OCR, and more.
"""

import PyPDF2
import pdfplumber
import openpyxl
import pandas as pd
from PIL import Image
import pytesseract
from pathlib import Path
from typing import Dict, Any, Optional
from utils import setup_logger

logger = setup_logger("file_tools")


def read_pdf(file_path: str, method: str = "pdfplumber") -> str:
    """
    Extract text from PDF.

    Args:
        file_path: Path to PDF file
        method: Extraction method ('pdfplumber' or 'pypdf2')

    Returns:
        Extracted text from PDF
    """
    logger.info(f"Reading PDF: {file_path}")

    try:
        if method == "pdfplumber":
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text

        else:  # PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""


def read_excel(file_path: str) -> Dict[str, Any]:
    """
    Read Excel file and return structured data.

    Args:
        file_path: Path to Excel file

    Returns:
        Dictionary with sheet data including columns, rows, and summary stats
    """
    logger.info(f"Reading Excel: {file_path}")

    try:
        df = pd.read_excel(file_path, sheet_name=None)

        result = {}
        for sheet_name, sheet_df in df.items():
            result[sheet_name] = {
                'columns': sheet_df.columns.tolist(),
                'rows': sheet_df.values.tolist(),
                'summary': sheet_df.describe().to_dict() if not sheet_df.empty else {}
            }

        return result

    except Exception as e:
        logger.error(f"Error reading Excel: {e}")
        return {}


def read_csv(file_path: str) -> Dict[str, Any]:
    """
    Read CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Dictionary with columns, rows, shape, and summary stats
    """
    logger.info(f"Reading CSV: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return {
            'columns': df.columns.tolist(),
            'rows': df.values.tolist(),
            'shape': df.shape,
            'summary': df.describe().to_dict() if not df.empty else {}
        }
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return {}


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from image using OCR.

    Args:
        image_path: Path to image file

    Returns:
        Text extracted from image via OCR
    """
    logger.info(f"OCR on image: {image_path}")

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logger.error(f"Error with OCR: {e}")
        return ""


def get_image_info(image_path: str) -> Dict[str, Any]:
    """
    Get image metadata.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image format, size, mode, and OCR detection
    """
    try:
        image = Image.open(image_path)
        return {
            'format': image.format,
            'size': image.size,
            'mode': image.mode,
            'has_text': bool(extract_text_from_image(image_path).strip())
        }
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        return {}


def process_file(file_path: str) -> Dict[str, Any]:
    """
    Process file based on extension.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file type and extracted content/data
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    handlers = {
        '.pdf': lambda: {'type': 'pdf', 'content': read_pdf(file_path)},
        '.xlsx': lambda: {'type': 'excel', 'content': read_excel(file_path)},
        '.xls': lambda: {'type': 'excel', 'content': read_excel(file_path)},
        '.csv': lambda: {'type': 'csv', 'content': read_csv(file_path)},
        '.png': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
        '.jpg': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
        '.jpeg': lambda: {'type': 'image', 'info': get_image_info(file_path), 'text': extract_text_from_image(file_path)},
    }

    handler = handlers.get(ext)
    if handler:
        return handler()
    else:
        logger.warning(f"Unsupported file type: {ext}")
        return {'type': 'unknown', 'error': f'Unsupported file type: {ext}'}
