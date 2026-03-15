import pandas as pd
import random
from collections import Counter
import os

# Set random seed for reproducibility
random.seed(42)

# File paths
input_file = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
output_file = 'symptom_sentence_dataset.csv'

# Load the dataset
print(f"Loading dataset from {input_file}...")
df = pd.read_csv(input_file)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns[:10])}... (showing first 10)")

# Extract symptoms (all columns except the first one which is 'diseases')
disease_col = df.columns[0]
symptom_columns = df.columns[1:]

print(f"\nTotal symptom columns: {len(symptom_columns)}")
print(f"Disease column: '{disease_col}'")

# Templates for varied sentence structures
sentence_templates = [
    "The patient reports {symptoms}.",
    "Symptoms include {symptoms}.",
    "The patient is experiencing {symptoms}.",
    "Reported symptoms are {symptoms}.",
    "The patient presents with {symptoms}.",
    "Clinical presentation includes {symptoms}.",
    "The patient shows {symptoms}.",
    "Notable symptoms are {symptoms}."
]

def format_symptoms(symptoms_list):
    """
    Join symptoms naturally with commas and 'and'.
    Example: ['fever', 'cough', 'fatigue'] -> 'fever, cough, and fatigue'
    """
    if len(symptoms_list) == 0:
        return ""
    elif len(symptoms_list) == 1:
        return symptoms_list[0]
    elif len(symptoms_list) == 2:
        return f"{symptoms_list[0]} and {symptoms_list[1]}"
    else:
        return ", ".join(symptoms_list[:-1]) + f", and {symptoms_list[-1]}"

def generate_sentence(symptoms_list):
    """Generate a sentence from a list of symptoms using random templates."""
    if not symptoms_list:
        return ""
    
    # Select a random template
    template = random.choice(sentence_templates)
    
    # Format symptoms
    formatted_symptoms = format_symptoms(symptoms_list)
    
    # Generate the sentence
    sentence = template.format(symptoms=formatted_symptoms)
    
    return sentence

# Statistics tracking
total_rows = len(df)
rows_with_no_symptoms = 0
processed_data = []
seen_symptom_combinations = set()

print("\nProcessing rows...")

for idx, row in df.iterrows():
    # Get disease label
    disease = row[disease_col]
    
    # Extract symptoms where value == 1
    symptoms = []
    for symptom_col in symptom_columns:
        if row[symptom_col] == 1:
            # Clean symptom name: already has spaces, just ensure lowercase
            clean_symptom = symptom_col.strip().lower()
            symptoms.append(clean_symptom)
    
    # Skip rows with no symptoms
    if len(symptoms) == 0:
        rows_with_no_symptoms += 1
        continue
    
    # Limit to 15 symptoms if necessary
    if len(symptoms) > 15:
        symptoms = random.sample(symptoms, 15)
    
    # Sort symptoms to create a unique key for deduplication
    # This ensures identical symptom combinations are detected
    symptoms_key = frozenset(symptoms)
    
    # Check for duplicates based on symptom combination
    if symptoms_key in seen_symptom_combinations:
        # Skip duplicate symptom combinations
        continue
    
    # Mark this symptom combination as seen
    seen_symptom_combinations.add(symptoms_key)
    
    # Generate sentence
    input_text = generate_sentence(symptoms)
    
    # Store processed row
    processed_data.append({
        'input_text': input_text,
        'disease': disease
    })
    
    if (idx + 1) % 5000 == 0:
        print(f"Processed {idx + 1} rows...")

# Create new dataframe
result_df = pd.DataFrame(processed_data)

# Statistics
rows_removed_duplicates = total_rows - rows_with_no_symptoms - len(result_df)
final_size = len(result_df)
unique_diseases = result_df['disease'].nunique()
disease_counts = result_df['disease'].value_counts()

print("\n" + "="*60)
print("DATASET CONVERSION COMPLETE")
print("="*60)

print(f"\nStatistics:")
print(f"  Total rows processed: {total_rows}")
print(f"  Rows removed (no symptoms): {rows_with_no_symptoms}")
print(f"  Rows removed (duplicates): {rows_removed_duplicates}")
print(f"  Final dataset size: {final_size}")
print(f"  Number of unique diseases: {unique_diseases}")

print(f"\nClass Distribution (Disease Label Counts):")
print("-" * 40)
for disease, count in disease_counts.items():
    percentage = (count / final_size) * 100
    print(f"  {disease}: {count} ({percentage:.2f}%)")

print("\nSample rows from the new dataset:")
print("-" * 40)
for i in range(min(5, len(result_df))):
    print(f"\nExample {i+1}:")
    print(f"  input_text: \"{result_df.iloc[i]['input_text']}\"")
    print(f"  disease: {result_df.iloc[i]['disease']}")

# Save the dataset
result_df.to_csv(output_file, index=False)
print(f"\n✓ Dataset saved to '{output_file}'")

# Display final dataframe info
print(f"\nFinal dataset info:")
print(f"  Rows: {result_df.shape[0]}")
print(f"  Columns: {result_df.shape[1]}")
print(f"  Columns: {list(result_df.columns)}")
