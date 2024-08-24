import re

def extract_entities(text):
    # Refined patterns for each entity
    name_pattern = r"Name[ :=]+([A-Z][a-zA-Z\s]*)(?=\s+(?:Husband's Name|Father's Name|Others|Age|Gender|House Number|Unique Number)|\n)"
    relative_name_pattern = r"(?:Husband's Name|Father's Name|Others)[ :=]+([A-Z][a-zA-Z\s]*)"
    relationship_type_pattern = r"(Husband's Name|Father's Name|Others)"
    age_pattern = r"Age[ :=]+(\d{1,3})"
    gender_pattern = r"Gender[ :=]+(Male|Female)"
    house_number_pattern = r"House Number[ :=]+([\d\-a-zA-Z]*)"
    unique_number_pattern = r"\b[A-Z]{4}\d{6}\b"

    # Extract entities
    entities = {
        "Names": re.findall(name_pattern, text),
        "Relative Names": re.findall(relative_name_pattern, text),
        "Relationship Types": re.findall(relationship_type_pattern, text),
        "Ages": re.findall(age_pattern, text),
        "Genders": re.findall(gender_pattern, text),
        "House Numbers": re.findall(house_number_pattern, text),
        "Unique Numbers": re.findall(unique_number_pattern, text)
    }

    # Post-processing to clean up extracted names
    entities['Names'] = [name.strip() for name in entities['Names']]
    entities['Relative Names'] = [name.strip() for name in entities['Relative Names']]
    entities['Relationship Types'] = [rt.replace("Husband's Name", "HSBN").replace("Father's Name", "FTHR").replace("Others", "OTHR") for rt in entities['Relationship Types']]
    
    # Refine house numbers
    refined_house_numbers = []
    for number in entities['House Numbers']:
        if re.match(r'^\d{2}[a-zA-Z]*$', number):
            refined_house_numbers.append(number)
        else:
            refined_house_numbers.append('-')
    
    entities['House Numbers'] = refined_house_numbers

    return entities
