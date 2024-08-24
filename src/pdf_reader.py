import cv2
import numpy as np
import re
from PIL import Image
import pytesseract
import os
from pdf2image import convert_from_path

def find_and_crop(image, template_dir, output_path):
    """
    Finds the location of the templates within the main image,
    compares their similarity, and crops all instances of the main 
    image based on the desired template.

    Args:
        image (PIL.Image.Image): The main image in PIL format.
        template_dir (str): Directory containing the template images.
        output_path (str): Path to save the cropped images.
    
    Returns:
        None: Saves all cropped images to the specified output path if the desired template has higher similarity.
    """

    # Save the processed PIL image to a temporary file
    temp_image_path = os.path.join(output_path, 'temp_image.png')
    image.save(temp_image_path)

    main_image = cv2.imread(temp_image_path)

    deleted_img_template = cv2.imread(os.path.join(template_dir, 'deleted_img_template.png'))
    desired_img_template = cv2.imread(os.path.join(template_dir, 'desired_img_template.png'))

    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    
    # Convert both templates to grayscale
    deleted_gray = cv2.cvtColor(deleted_img_template, cv2.COLOR_BGR2GRAY)
    desired_gray = cv2.cvtColor(desired_img_template, cv2.COLOR_BGR2GRAY)

    # Match the main image with both templates
    result_deleted = cv2.matchTemplate(main_gray, deleted_gray, cv2.TM_CCOEFF_NORMED)
    result_desired = cv2.matchTemplate(main_gray, desired_gray, cv2.TM_CCOEFF_NORMED)

    # Get the maximum similarity values
    _, max_val_deleted, _, _ = cv2.minMaxLoc(result_deleted)
    _, max_val_desired, _, _ = cv2.minMaxLoc(result_desired)

    # Compare the maximum values to determine which template is more similar
    if max_val_deleted > max_val_desired:
        print("The main image is more similar to the deleted image template. No action taken.")
    else:
        print("The main image is more similar to the desired image template. Cropping all instances of the template in the image.")
        template_height, template_width = desired_gray.shape

        # Find all locations of the desired template
        threshold = 0.8  # Adjust this threshold as needed
        loc = np.where(result_desired >= threshold)

        # Iterate through all found locations and crop the corresponding part of the image
        for i, pt in enumerate(zip(*loc[::-1])):
            bottom_right = (pt[0] + template_width, pt[1] + template_height)
            cropped_image = main_image[pt[1]:pt[1] + template_height, pt[0]:pt[0] + template_width]
            cropped_image_path = os.path.join(output_path, f'cropped_image_{i + 1}.png')
            cv2.imwrite(cropped_image_path, cropped_image)
            print(f"Cropped image saved to {cropped_image_path}")

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

    kernel = np.ones((1, 1), np.uint8)
    processed_image = cv2.dilate(binary_image, kernel, iterations=1)
    processed_image = cv2.erode(processed_image, kernel, iterations=1)

    images_folder = 'images'
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    processed_image_path = os.path.join(images_folder, 'processed_image.png')
    
    cv2.imwrite(processed_image_path, processed_image)

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
    Extracts images from a file. Supports both PDFs and image files.

    Args:
        file_path (str): Path to the file from which images are to be extracted.
    
    Returns:
        list[PIL.Image.Image]: List of images extracted from the file.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.pdf':
        images = convert_from_path(file_path)
    else:
        image = Image.open(file_path)
        images = [image]
    return images

def extract_name(text, id_type):
    """
    Extracts the name from the OCR-processed text based on the ID type.

    Args:
        text (str): OCR-processed text from which the name needs to be extracted.
        id_type (str): Type of ID ('aadhar' or 'pan').
    
    Returns:
        str: Extracted name or 'Couldnt extract name' if extraction fails.
    """
    lines = text.split('\n')
    dob_index = None
    name_index = None
    
    for i, line in enumerate(lines):
        if id_type == 'aadhar':
            if 'DOB' in line or 'D0B' in line:
                dob_index = i
                break
            elif 'MALE' in line or 'FEMALE' in line or 'TRANSGENDER' in line:
                dob_index = i - 1
        
        elif id_type == 'pan':
            if 'FATHER\'S NAME' in line.upper():
                dob_index = i - 1
                break
            elif 'DOB' in line.upper():
                dob_index = i - 1
                break
    
    if dob_index is not None:
        name_index = dob_index - 1
        while name_index >= 0 and lines[name_index].strip() == '':
            name_index -= 1
        if name_index >= 0:
            alphabets_only = re.sub(r'[^a-zA-Z\s]', '', lines[name_index])
            words = alphabets_only.split()
            capitalized_words = [word for word in words if word[0].isupper()]
            result = ' '.join(capitalized_words)
            return result.strip()
        else:
            print('No valid name line found before DOB')
    else:
        print('DOB or Father\'s Name not found in the text')

    return 'Couldnt extract name'

def extract_user_details_from_text(text, id_type):
    """
    Extracts user details such as name and ID number from OCR-processed text.

    Args:
        text (str): OCR-processed text from which details are to be extracted.
        id_type (str): Type of ID ('aadhar' or 'pan').
    
    Returns:
        dict: Dictionary containing the extracted user details (name and ID).
    """
    name = None
    user_details = {}
    
    if id_type == 'aadhar':
        id_pattern = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
        name = extract_name(text, id_type)
    elif id_type == 'pan':
        id_pattern = re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b')
        name = extract_name(text, id_type)
    else:
        raise ValueError('Unsupported ID type')
    
    id_match = id_pattern.search(text)
    
    if id_match:
        user_details['id'] = id_match.group(0)
    
    if name:
        user_details['name'] = name
    
    return user_details

def extract_details_from_file(image_path, template_path, output_path):
    """
    Extracts user details from a given file (image or PDF). If the OCR text is too long, it crops the image and retries extraction.

    Args:
        image_path (str): Path to the input file.
        template_path (str): Path to the template image for cropping.
        output_path (str): Path to save the cropped image.
    
    Returns:
        dict: Dictionary containing the extracted user details (name and ID).
    """
    file_extension = os.path.splitext(image_path)[1].lower()
    
    if file_extension == '.pdf':
        images = extract_images_from_file(image_path)
    else:
        raise ValueError('Unsupported file format')
    
    user_details = {}
    for image in images:
        processed_image = preprocess_image(image)
        find_and_crop(processed_image, template_path, output_path)

        text = ''
        # Loop through each file in the folder
        for filename in os.listdir(output_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(output_path, filename)
                with Image.open(file_path) as img:
                    text += extract_text_from_image(img)
                    text += '\n\n\n'
        
    print('text : \n', text)
    
    return user_details