import cv2
import numpy as np
import os

def crop_entries_without_deleted(image_path, output_folder, deleted_template_path):
    # Load the main image
    main_image = cv2.imread(image_path)
    main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    
    # Load the 'DELETED' template
    deleted_template = cv2.imread(deleted_template_path, 0)
    
    # Match template to find 'DELETED' areas
    result = cv2.matchTemplate(main_gray, deleted_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8  # Adjust as needed
    loc = np.where(result >= threshold)

    # Find all rectangles where 'DELETED' is located
    deleted_rectangles = []
    template_height, template_width = deleted_template.shape
    for pt in zip(*loc[::-1]):
        rect = (pt[0], pt[1], pt[0] + template_width, pt[1] + template_height)
        deleted_rectangles.append(rect)
    
    # Iterate through each entry in the main image and crop if it doesn't overlap with any 'DELETED' rectangle
    entry_height = 250  # Adjust based on the layout of your entries
    entry_width = 600# Adjust based on the layout of your entries
    
    entry_count = 0
    
    # Iterate over the entries by the estimated height and width
    for y in range(0, main_image.shape[0], entry_height):
        for x in range(0, main_image.shape[1], entry_width):
            # Define the entry rectangle
            entry_rect = (x, y, x + entry_width, y + entry_height)
            
            # Check for overlap with any 'DELETED' rectangles
            overlap = False
            for dr in deleted_rectangles:
                # Check if rectangles overlap
                if not (entry_rect[2] <= dr[0] or entry_rect[0] >= dr[2] or 
                        entry_rect[3] <= dr[1] or entry_rect[1] >= dr[3]):
                    overlap = True
                    break
            
            # Crop the entry if no overlap with 'DELETED' rectangles
            if not overlap:
                entry_count += 1
                cropped_entry = main_image[y:y + entry_height, x:x + entry_width]
                cropped_entry_path = os.path.join(output_folder, f'cropped_entry_{entry_count}.png')
                cv2.imwrite(cropped_entry_path, cropped_entry)
                print(f'Cropped entry saved: {cropped_entry_path}')

# Paths to the images and template
image_path = r'C:\Users\Owner\Desktop\kshetra\images\processed_image.png'  # Your uploaded image
output_folder = r'C:\Users\Owner\Desktop\kshetra\extracted_images'  # Folder where cropped images will be saved
deleted_template_path = r'C:\Users\Owner\Desktop\kshetra\images\deleted_img_template.png'  # Path to your 'DELETED' template image

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Execute the function
crop_entries_without_deleted(image_path, output_folder, deleted_template_path)
