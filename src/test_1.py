import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_table_structure(binary_image):
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    table_structure = np.zeros_like(binary_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(table_structure, (x1, y1), (x2, y2), 255, 2)
    
    return table_structure

def extract_table_cells(table_structure):
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cells.append((x, y, w, h))
    return cells

def extract_text_from_cells(image, cells):
    extracted_data = []
    for cell in cells:
        x, y, w, h = cell
        cropped_image = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(cropped_image, config='--psm 6')
        extracted_data.append(text.strip())
    return extracted_data

image_path = r'C:\Users\Owner\Desktop\kshetra\extracted_images\temp_image_1.png'
binary_image = preprocess_image(image_path)
table_structure = detect_table_structure(binary_image)
cells = extract_table_cells(table_structure)
extracted_data = extract_text_from_cells(cv2.imread(image_path), cells)

# Output the extracted data
for data in extracted_data:
    print(data)
