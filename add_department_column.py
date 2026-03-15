import pandas as pd

# ============================================================================
# DISEASE TO DEPARTMENT MAPPING
# ============================================================================
# Organize diseases by medical specialty for easy maintenance and extension
# Simply add new diseases to the appropriate department dictionary

DISEASE_DEPARTMENT_MAPPING = {
    # Cardiology - Heart and Cardiovascular diseases
    "Cardiology": [
        "angina", "arrhythmia", "atrial fibrillation", "heart failure", 
        "hypertension", "myocardial infarction", "palpitations", "aortic valve disease",
        "mitral valve disease", "heart murmur", "aortic stenosis", "acute coronary syndrome",
        "coronary artery disease", "congestive heart failure", "cardiomyopathy",
        "pericarditis", "endocarditis", "heart block", "ventricular fibrillation",
        "abdominal aortic aneurysm", "aortic aneurysm"
    ],
    
    # Pulmonology - Lungs and Respiratory diseases
    "Pulmonology": [
        "pneumonia", "asthma", "chronic obstructive pulmonary disease", "copd",
        "bronchitis", "acute bronchitis", "chronic bronchitis", "cough",
        "shortness of breath", "acute bronchiolitis", "acute bronchospasm",
        "pulmonary edema", "tuberculosis", "lung cancer", "emphysema",
        "acute respiratory distress syndrome", "ards", "pleural effusion",
        "pneumothorax", "pleurisy", "cystic fibrosis", "scarring of the lungs",
        "wheezing", "abscess of the lung", "aspiration pneumonia"
    ],
    
    # Endocrinology - Diabetes and Hormonal disorders
    "Endocrinology": [
        "diabetes", "type 1 diabetes", "type 2 diabetes", "gestational diabetes",
        "hypoglycemia", "hyperglycemia", "thyroid disease", "hyperthyroidism",
        "hypothyroidism", "thyroiditis", "goiter", "adrenal adenoma",
        "cushings syndrome", "addisons disease", "pituitary adenoma",
        "growth hormone disorder"
    ],
    
    # Gastroenterology - Digestive system
    "Gastroenterology": [
        "gastroenteritis", "infectious gastroenteritis", "noninfectious gastroenteritis",
        "acid reflux", "gerd", "ulcer", "peptic ulcer", "appendicitis",
        "colitis", "crohns disease", "ulcerative colitis", "ibs",
        "irritable bowel syndrome", "celiac disease", "pancreatitis",
        "acute pancreatitis", "chronic pancreatitis", "cholecystitis", "gallstones",
        "cholecystolithiasis", "diverticulitis", "diverticulosis",
        "hemorrhoids", "anal fissure", "anal fistula", "constipation",
        "diarrhea", "irritable bowel", "inflammatory bowel disease",
        "esophagitis", "gastroesophageal reflux disease", "hiatal hernia",
        "hepatitis", "cirrhosis", "liver disease", "alcoholic liver disease",
        "fatty liver disease", "cholangitis", "ascending cholangitis",
        "gastrointestinal hemorrhage", "nausea", "vomiting", "abdominal hernia",
        "abdominal pain", "cramps"
    ],
    
    # Neurology - Nervous system and Brain
    "Neurology": [
        "migraine", "headache", "seizure", "epilepsy", "stroke", "ischemic stroke",
        "hemorrhagic stroke", "transient ischemic attack", "tia",
        "parkinson disease", "multiple sclerosis", "ms", "amyotrophic lateral sclerosis",
        "als", "alzheimer disease", "dementia", "neuropathy", "peripheral nerve disorder",
        "sciatica", "carpal tunnel syndrome", "concussion", "traumatic brain injury",
        "meningitis", "encephalitis", "guillain barre syndrome", "myasthenia gravis",
        "bell palsy", "vertigo", "dizziness", "syncope", "fainting",
        "tremor", "ataxia", "numbness", "tingling", "weakness",
        "balance disorder", "spinal stenosis", "spondylosis"
    ],
    
    # Psychiatry - Mental health and Behavioral disorders
    "Psychiatry": [
        "anxiety", "anxiety and nervousness", "anxiety disorder", "panic disorder",
        "depression", "depressive disorder", "major depressive disorder",
        "bipolar disorder", "schizophrenia", "psychosis", "psychotic disorder",
        "depressive or psychotic symptoms", "stress reaction", "acute stress reaction",
        "adjustment reaction", "ocd", "obsessive compulsive disorder",
        "ptsd", "post traumatic stress disorder", "sleep disorder", "insomnia",
        "substance abuse", "alcohol abuse", "alcohol intoxication", "alcohol withdrawal",
        "drug abuse", "marijuana abuse", "opioid abuse", "withdrawal syndrome"
    ],
    
    # Dermatology - Skin conditions
    "Dermatology": [
        "acne", "eczema", "psoriasis", "urticaria", "hives", "rash",
        "dermatitis", "contact dermatitis", "seborrheic dermatitis",
        "fungal infection", "ringworm", "athlete foot", "fungal infection of the hair",
        "nail fungus", "warts", "moles", "skin cancer", "melanoma",
        "basal cell carcinoma", "squamous cell carcinoma", "acanthosis nigricans",
        "alopecia", "hair loss", "vitiligo", "birthmarks", "nevus",
        "actinic keratosis", "cold sore", "shingles", "chickenpox",
        "impetigo", "cellulitis", "abscess", "boils", "folliculitis"
    ],
    
    # Orthopedics - Bones, joints, and muscles
    "Orthopedics": [
        "arthritis", "osteoarthritis", "rheumatoid arthritis", "gout",
        "sprain", "strain", "sprain or strain", "fracture", "broken bone",
        "dislocation", "tendinitis", "bursitis", "adhesive capsulitis",
        "adhesive capsulitis of the shoulder", "rotator cuff injury",
        "plantar fasciitis", "achilles tendinitis", "knee pain",
        "back pain", "lower back pain", "neck pain", "shoulder pain",
        "arthritis of the hip", "hip pain", "ankle pain", "foot pain",
        "muscle pain", "muscle strain", "muscle cramp", "weakness",
        "ankylosing spondylitis", "systemic lupus erythematosus", "sle",
        "fibromyalgia", "complex regional pain syndrome", "osteoporosis",
        "joint swelling", "joint stiffness"
    ],
    
    # Rheumatology - Autoimmune and connective tissue diseases
    "Rheumatology": [
        "rheumatoid arthritis", "systemic lupus erythematosus", "sle",
        "sjorgrens syndrome", "vasculitis", "polymyalgia rheumatica",
        "temporal arteritis", "ankylosing spondylitis", "reactive arthritis",
        "psoriatic arthritis", "scleroderma", "mixed connective tissue disease",
        "antiphospholipid syndrome", "behcets disease", "kawasaki disease",
        "amyloidosis"
    ],
    
    # Oncology - Cancer and Malignancies
    "Oncology": [
        "cancer", "lung cancer", "breast cancer", "cancer of breast",
        "skin cancer", "melanoma", "colon cancer", "colorectal cancer",
        "prostate cancer", "ovarian cancer", "cervical cancer", "uterine cancer",
        "thyroid cancer", "liver cancer", "pancreatic cancer", "stomach cancer",
        "gastric cancer", "esophageal cancer", "leukemia", "lymphoma",
        "multiple myeloma", "basal cell carcinoma", "squamous cell carcinoma",
        "anemia due to malignancy"
    ],
    
    # Ophthalmology - Eyes and Vision
    "Ophthalmology": [
        "cataracts", "glaucoma", "acute glaucoma", "macular degeneration",
        "diabetic retinopathy", "retinopathy", "conjunctivitis", "pink eye",
        "conjunctivitis due to allergy", "dry eye", "keratitis",
        "uveitis", "color blindness", "amblyopia", "presbyopia",
        "astigmatism", "myopia", "hyperopia", "nearsightedness", "farsightedness",
        "stye", "chalazion", "floaters", "flashes of light", "vision loss",
        "blurred vision", "eye pain"
    ],
    
    # Otolaryngology - Ears, Nose, and Throat
    "Otolaryngology": [
        "hearing loss", "deafness", "tinnitus", "ear infection", "otitis media",
        "acute otitis media", "otitis externa", "earache", "ear pain",
        "sinusitis", "acute sinusitis", "chronic sinusitis", "nasal congestion",
        "runny nose", "nose bleed", "epistaxis", "sore throat", "strep throat",
        "pharyngitis", "laryngitis", "hoarseness", "voice loss",
        "tonsillitis", "throat pain", "abscess of the pharynx", "abscess of nose",
        "nose disorder", "nasal polyps", "deviated septum", "sleep apnea",
        "obstructive sleep apnea", "osa"
    ],
    
    # Nephrology - Kidneys
    "Nephrology": [
        "kidney disease", "chronic kidney disease", "acute kidney injury",
        "kidney failure", "end stage renal disease", "esrd", "kidney stones",
        "nephrolithiasis", "uremia", "proteinuria", "hematuria",
        "glomerulonephritis", "pyelonephritis", "cystitis", "urinary tract infection",
        "uti", "bladder infection", "urinary incontinence", "anemia due to chronic kidney disease",
        "nephritis", "lupus nephritis", "diabetic kidney disease"
    ],
    
    # Urology - Urinary and Reproductive system (male)
    "Urology": [
        "benign prostatic hyperplasia", "bph", "prostate cancer", "prostatitis",
        "erectile dysfunction", "impotence", "infertility", "low testosterone",
        "testicular cancer", "testicular torsion", "epididymitis", "orchitis",
        "urinary retention", "urinary incontinence", "kidney stones", "nephrolithiasis",
        "bladder cancer", "prostate disease"
    ],
    
    # Gynecology - Reproductive system (female) and Obstetrics
    "Gynecology": [
        "pregnancy", "problem during pregnancy", "gestational diabetes",
        "preeclampsia", "eclampsia", "placental abruption", "placenta previa",
        "labor complications", "delivery complications", "postpartum depression",
        "miscarriage", "spontaneous abortion", "ectopic pregnancy",
        "menstrual disorder", "amenorrhea", "dysmenorrhea", "irregular periods",
        "heavy menstrual bleeding", "premenstrual syndrome", "pms",
        "menopause", "hot flashes", "night sweats", "ovarian cyst",
        "vaginal cyst", "polycystic ovary syndrome", "pcos", "uterine polyps",
        "fibroids", "endometriosis", "fibroid tumor", "breast cancer",
        "cancer of breast", "vaginitis", "vaginal infection", "yeast infection",
        "bacterial vaginosis", "vulvodynia", "cervicitis", "pelvic inflammatory disease",
        "pid", "infertility", "ovarian cancer", "cervical cancer", "uterine cancer"
    ],
    
    # Immunology/Allergy
    "Allergy & Immunology": [
        "allergy", "allergic reaction", "anaphylaxis", "anaphylactic shock",
        "food allergy", "drug allergy", "penicillin allergy",
        "hay fever", "allergic rhinitis", "contact dermatitis",
        "urticaria", "hives", "conjunctivitis due to allergy",
        "allergy to animals", "latex allergy", "asthma", "asthma attack",
        "immunodeficiency", "hiv", "aids", "immunosuppression"
    ],
    
    # Infectious Disease
    "Infectious Disease": [
        "hiv", "aids", "covid", "covid-19", "coronavirus", "influenza", "flu",
        "measles", "mumps", "rubella", "chickenpox", "shingles",
        "herpes", "herpes simplex", "herpes zoster", "cold sore",
        "mononucleosis", "ebstein barr virus", "hepatitis", "hepatitis a",
        "hepatitis b", "hepatitis c", "tuberculosis", "tb",
        "malaria", "dengue fever", "zika virus", "ebola",
        "meningitis", "encephalitis", "sepsis", "septic shock",
        "blood infection", "bacteremia", "viremia", "fungal infection",
        "fungal infection of the hair", "candida", "aspergillosis",
        "parasitic infection", "roundworm", "tapeworm", "acariasis",
        "scabies", "lice", "strep throat", "whooping cough", "pertussis",
        "diphtheria", "tetanus", "pneumococcal disease", "impetigo",
        "cellulitis", "abscess", "boils", "carbuncle"
    ],
    
    # Hematology - Blood disorders
    "Hematology": [
        "anemia", "iron deficiency anemia", "anemia due to chronic kidney disease",
        "anemia due to malignancy", "anemia of chronic disease", "aplastic anemia",
        "hemolytic anemia", "sickle cell disease", "sickle cell anemia",
        "leukemia", "lymphoma", "multiple myeloma", "bleeding disorder",
        "hemophilia", "von willebrand disease", "thrombocytopenia",
        "low platelets", "blood clot", "deep vein thrombosis", "dvt",
        "pulmonary embolism", "pe", "thrombosis", "embolism"
    ],
    
    # Trauma and General Surgery
    "Trauma/General Surgery": [
        "injury", "injury to the leg", "injury to the arm", "trauma",
        "wound", "laceration", "cut", "bruise", "contusion", "burn",
        "first degree burn", "second degree burn", "third degree burn",
        "fracture", "broken bone", "dislocation", "limp", "swelling",
        "concussion", "head injury", "spinal injury", "internal bleeding",
        "external bleeding", "hemorrhage", "gastrointestinal hemorrhage"
    ],
    
    # Pain Management
    "Pain Management": [
        "pain", "chronic pain", "acute pain", "back pain", "lower back pain",
        "neck pain", "shoulder pain", "chest pain", "sharp chest pain",
        "abdominal pain", "joint pain", "knee pain", "hip pain",
        "foot pain", "ankle pain", "headache", "migraine",
        "muscle pain", "arthritis pain", "neuropathic pain", "phantom pain",
        "cancer pain"
    ]
}

# ============================================================================
# CREATE REVERSE MAPPING - Disease to Department
# ============================================================================
DISEASE_TO_DEPARTMENT = {}
for department, diseases in DISEASE_DEPARTMENT_MAPPING.items():
    for disease in diseases:
        DISEASE_TO_DEPARTMENT[disease.lower().strip()] = department


def map_disease_to_department(disease):
    """
    Maps a disease to its appropriate medical department.
    
    Args:
        disease (str): The disease name
        
    Returns:
        str: The medical department, or "General Medicine" if not found
    """
    disease_clean = disease.lower().strip()
    return DISEASE_TO_DEPARTMENT.get(disease_clean, "General Medicine")


# ============================================================================
# MAIN PROCESSING
# ============================================================================
def main():
    print("=" * 70)
    print("DISEASE TO DEPARTMENT MAPPER")
    print("=" * 70)
    
    # Read the dataset
    print("\nReading dataset...")
    input_file = 'symptom_sentence_dataset.csv'
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df)} rows with columns: {list(df.columns)}")
    
    # Apply the mapping
    print("\nMapping diseases to departments...")
    df['department'] = df['disease'].apply(map_disease_to_department)
    
    # Statistics
    print(f"✓ Added 'department' column")
    
    # Show the mapping coverage
    unmapped_diseases = df[df['department'] == 'General Medicine']['disease'].nunique()
    total_unique_diseases = df['disease'].nunique()
    mapped_diseases = total_unique_diseases - unmapped_diseases
    mapping_percentage = (mapped_diseases / total_unique_diseases) * 100
    
    print(f"\nMapping Coverage:")
    print(f"  Total unique diseases: {total_unique_diseases}")
    print(f"  Mapped diseases: {mapped_diseases} ({mapping_percentage:.1f}%)")
    print(f"  Unmapped diseases (assigned to General Medicine): {unmapped_diseases}")
    
    # Show department distribution
    print(f"\nDepartment Distribution:")
    print("-" * 70)
    dept_counts = df['department'].value_counts().sort_values(ascending=False)
    for dept, count in dept_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {dept}: {count} ({percentage:.2f}%)")
    
    # Show sample of unmapped diseases (first 10)
    if unmapped_diseases > 0:
        print(f"\nSample of unmapped diseases (first 10):")
        print("-" * 70)
        unmapped = df[df['department'] == 'General Medicine']['disease'].unique()[:10]
        for disease in unmapped:
            print(f"  • {disease}")
        if unmapped_diseases > 10:
            print(f"  ... and {unmapped_diseases - 10} more")
    
    # Save the updated dataset
    output_file = 'symptom_sentence_dataset_with_department.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved updated dataset to '{output_file}'")
    
    # Show sample records
    print(f"\nSample records from the new dataset:")
    print("-" * 70)
    sample_df = df[['input_text', 'disease', 'department']].head(5)
    for idx, row in sample_df.iterrows():
        print(f"\nExample {idx + 1}:")
        print(f"  Disease: {row['disease']}")
        print(f"  Department: {row['department']}")
        print(f"  Text: {row['input_text'][:80]}...")
    
    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}\n")
    
    return df


if __name__ == "__main__":
    df = main()
