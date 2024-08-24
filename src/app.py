import os
from pdf_2 import extract_text_from_pdf
from extract_entity import extract_entities
import pandas as pd

def write_entities_to_csv(entities, output_csv_path):
    # Convert dictionary to DataFrame
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in entities.items()]))
    
    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)



# Define the paths
pdf_path = r'C:\Users\Owner\Desktop\kshetra\Sample Problem.pdf'
output_path = r'C:\Users\Owner\Desktop\kshetra\extracted_images'
output_csv_path = r'C:\Users\Owner\Desktop\kshetra\entities.csv'

# Ensure the output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Extract text from the PDF
extracted_text = extract_text_from_pdf(pdf_path, output_path)
entities = extract_entities(extracted_text)

# Print the extracted text
print('Extracted Text:', extracted_text)
print('Entities:', entities)



write_entities_to_csv(entities, output_csv_path)
