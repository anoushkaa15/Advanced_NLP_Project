#!/usr/bin/env python3
"""
Quick Start Guide for Medical Chatbot
======================================
Copy and paste these commands to get started.
"""

print("""
╔════════════════════════════════════════════════════════════════════╗
║         MEDICAL CHATBOT - QUICK START GUIDE                       ║
╚════════════════════════════════════════════════════════════════════╝

### STEP 1: Install Dependencies

    pip install -r requirements.txt

    This will install:
    ✓ pandas           - Data handling
    ✓ numpy            - Numerical operations
    ✓ scikit-learn     - ML algorithms & text analysis
    ✓ langdetect       - Language detection
    ✓ googletrans      - Translation API

### STEP 2: Run Interactive Chatbot

    python medical_chatbot.py

    Then type your symptoms in any language:
    • English: "I have a fever and cough"
    • Spanish: "Tengo fiebre y tos"
    • French: "J'ai une fièvre et une toux"
    • Hindi: "मुझे बुखार और खांसी है"

### STEP 3: Use Programmatically

    from medical_chatbot_api import MedicalChatbotAPI
    
    api = MedicalChatbotAPI()
    result = api.analyze_symptoms("I have severe headache")
    print(result)

### STEP 4: Run API Examples

    python medical_chatbot_api.py

    This will show you:
    ✓ How to analyze single/multiple queries
    ✓ Dataset statistics
    ✓ Disease search functionality
    ✓ Department-based queries

╔════════════════════════════════════════════════════════════════════╗
║                      USAGE EXAMPLES                               ║
╚════════════════════════════════════════════════════════════════════╝

--- EXAMPLE 1: Interactive Chat ---
$ python medical_chatbot.py

🩺 You: I have a high fever and severe cough
[1/5] Detecting language...
    ✓ Detected language: English
[2/5] Translating to English...
    ✓ English translation: "I have a high fever and severe cough"
[3/5] Searching for matching diseases...
    ✓ Found 3 match(es)
[4/5] Translating response to English...
    ✓ Response translated
[5/5] Complete!

📋 Top Match: acute bronchitis
🏥 Department: Pulmonology
🎯 Confidence: 78.5%

💬 Response:
Based on your symptoms, the most likely condition is: acute bronchitis.
This falls under the Pulmonology department. Confidence: 78.5%

--- EXAMPLE 2: Programmatic Usage ---
from medical_chatbot_api import MedicalChatbotAPI

api = MedicalChatbotAPI()

# Single analysis
result = api.analyze_symptoms("I have joint pain and swelling")
print(f"Disease: {result['predicted_disease']}")
print(f"Department: {result['predicted_department']}")
print(f"Confidence: {result['confidence']:.1%}")

# Output:
# Disease: arthritis
# Department: Orthopedics
# Confidence: 82.3%

--- EXAMPLE 3: Multilingual ---
# Spanish
result = api.analyze_symptoms("Tengo fiebre alta y tos severa")
# Response will be in Spanish!

# Hindi
result = api.analyze_symptoms("मुझे गंभीर सिरदर्द है")
# Response will be in Hindi!

# French
result = api.analyze_symptoms("J'ai une douleur thoracique")
# Response will be in French!

--- EXAMPLE 4: Batch Analysis ---
symptoms_list = [
    "fever and cough",
    "chest pain and shortness of breath",
    "joint pain and swelling"
]

results = api.analyze_symptoms_batch(symptoms_list)

for result in results:
    print(f"{result['predicted_disease']} ({result['predicted_department']})")

--- EXAMPLE 5: Get Statistics ---
stats = api.get_statistics()

print(f"Total Diseases: {stats['total_unique_diseases']}")
print(f"Total Departments: {stats['total_departments']}")
print(f"Records in Database: {stats['total_records']}")

--- EXAMPLE 6: Search for Diseases ---
results = api.search_diseases("fever")
for disease in results:
    print(f"{disease['disease']} → {disease['department']}")

--- EXAMPLE 7: Get All Diseases in a Department ---
cardio_diseases = api.get_department_diseases("Cardiology")
print(f"Cardiology diseases: {cardio_diseases}")

╔════════════════════════════════════════════════════════════════════╗
║                    TROUBLESHOOTING                                ║
╚════════════════════════════════════════════════════════════════════╝

Q: "ModuleNotFoundError: No module named 'sklearn'"
A: Run: pip install scikit-learn

Q: "ModuleNotFoundError: No module named 'langdetect'"
A: Run: pip install langdetect

Q: "ModuleNotFoundError: No module named 'googletrans'"
A: Run: pip install googletrans

Q: Dataset file not found error
A: Make sure 'symptom_sentence_dataset_with_department.csv' exists
   in the same directory as the script

Q: Translation is slow
A: First query takes longer due to API initialization. Subsequent
   queries are faster. Consider caching frequent queries.

Q: Accuracy is low for some queries
A: Medical chatbots are not 100% accurate. Always recommend users
   to consult actual medical professionals for diagnosis.

Q: No matches found
A: The threshold is set to 0.3 (30% similarity).
   Modify SIMILARITY_THRESHOLD in medical_chatbot.py if needed.

╔════════════════════════════════════════════════════════════════════╗
║                     NEXT STEPS                                    ║
╚════════════════════════════════════════════════════════════════════╝

1. PRODUCTION DEPLOYMENT
   - Build a web API using Flask or FastAPI
   - Deploy to cloud (AWS, GCP, Azure)
   - Add caching for faster responses

2. ACCURACY IMPROVEMENT
   - Use BERT embeddings instead of TF-IDF
   - Fine-tune on medical corpora
   - Add feedback mechanism for corrections

3. FEATURE ADDITIONS
   - Multi-turn conversations
   - Ask clarifying questions
   - Show related conditions
   - Provide treatment recommendations

4. ACCESSIBILITY
   - Add symptom picker UI
   - Mobile app integration
   - Voice input/output support

╔════════════════════════════════════════════════════════════════════╗
║              FOR MORE DETAILS, READ:                              ║
║          CHATBOT_DOCUMENTATION.md                                 ║
╚════════════════════════════════════════════════════════════════════╝
""")
