import cv2
import numpy as np
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

def preprocess_image(image):
    """
    Preprocesses the input image to improve the accuracy of OCR (Optical Character Recognition).

    Args:
        image (PIL.Image.Image): Image to be processed.
    
    Returns:
        PIL.Image.Image: Processed image ready for OCR.
    """
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    images_folder = 'images'
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    processed_image_path = os.path.join(images_folder, 'processed_image.png')
    
    cv2.imwrite(processed_image_path, binary_image)

    processed_image_pil = Image.open(processed_image_path)

    return processed_image_pil

def extract_text_from_image(image):
    """
    Extracts text from the given image using Tesseract OCR.

    Args:
        image (PIL.Image.Image): Image from which text needs to be extracted.
    
    Returns:
        str: Extracted text from the image.
    """
    custom_config = r'--oem 3 --psm 4'
    text = pytesseract.image_to_string(image, config=custom_config)
    return text

def extract_images_from_file(file_path):
    """
    Extracts images from a PDF file.

    Args:
        file_path (str): Path to the file from which images are to be extracted.
    
    Returns:
        list[PIL.Image.Image]: List of images extracted from the PDF.
    """
    images = convert_from_path(file_path)
    return images

def extract_text_from_pdf(pdf_path, output_path):
    """
    Extracts text from each page of the PDF and returns the concatenated text.

    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the processed images.
    
    Returns:
        str: Concatenated text extracted from the entire PDF.
    """
    images = extract_images_from_file(pdf_path)
    full_text = ""
    
    for i, image in enumerate(images):
        processed_image = preprocess_image(image)
        
        # Save the processed image temporarily
        temp_image_path = os.path.join(output_path, f'processed_image_{i+1}.png')
        processed_image.save(temp_image_path)
        
        # Extract text from the processed image
        text = extract_text_from_image(processed_image)
        full_text += text + "\n"
    
    return full_text
