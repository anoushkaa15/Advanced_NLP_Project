"""
Medical Chatbot API Wrapper
============================
This module provides a programmatic interface to the medical chatbot.
Can be used to build REST APIs, web applications, or integrate into other systems.

Example Usage:
    from medical_chatbot_api import MedicalChatbotAPI
    
    # Initialize
    api = MedicalChatbotAPI('symptom_sentence_dataset_with_department.csv')
    
    # Single query
    result = api.analyze_symptoms("I have a fever and cough")
    print(result)
    
    # Batch processing
    queries = ["fever and headache", "chest pain", "joint pain"]
    results = api.analyze_symptoms_batch(queries)
"""

import json
from typing import Dict, List, Union
import pandas as pd
from medical_chatbot import MedicalChatbot


class MedicalChatbotAPI:
    """
    API wrapper for the Medical Chatbot.
    Provides simple methods for symptom analysis and disease prediction.
    """
    
    def __init__(self, dataset_path: str = 'symptom_sentence_dataset_with_department.csv'):
        """
        Initialize the API.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.chatbot = MedicalChatbot(dataset_path)
    
    def analyze_symptoms(self, symptoms: str, language: str = None) -> Dict:
        """
        Analyze symptoms and return disease predictions.
        
        Args:
            symptoms (str): Description of symptoms in any language
            language (str): Optional language code (if None, auto-detected)
            
        Returns:
            Dict: Results containing disease, department, confidence, and response
        """
        result = self.chatbot.process_query(symptoms)
        
        # Simplified output
        return {
            'status': 'success' if result['matches'] else 'no_match',
            'input_language': result['user_language'],
            'predicted_disease': result['matches'][0]['disease'] if result['matches'] else None,
            'predicted_department': result['matches'][0]['department'] if result['matches'] else None,
            'confidence': result['confidence'],
            'alternative_matches': [
                {
                    'disease': m['disease'],
                    'department': m['department'],
                    'confidence': m['similarity']
                }
                for m in result['matches'][1:]
            ],
            'response': result['response']
        }
    
    def analyze_symptoms_batch(self, symptoms_list: List[str]) -> List[Dict]:
        """
        Analyze multiple symptom descriptions.
        
        Args:
            symptoms_list (List[str]): List of symptom descriptions
            
        Returns:
            List[Dict]: List of analysis results
        """
        results = []
        for symptoms in symptoms_list:
            result = self.analyze_symptoms(symptoms)
            results.append(result)
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the dataset and model.
        
        Returns:
            Dict: Dataset statistics
        """
        df = self.chatbot.symptom_matcher.df
        
        return {
            'total_records': len(df),
            'total_unique_diseases': df['disease'].nunique(),
            'total_departments': df['department'].nunique(),
            'departments': df['department'].value_counts().to_dict(),
            'top_diseases': df['disease'].value_counts().head(10).to_dict()
        }
    
    def search_diseases(self, disease_name: str) -> List[Dict]:
        """
        Search for a specific disease in the database.
        
        Args:
            disease_name (str): Disease name (partial match supported)
            
        Returns:
            List[Dict]: Matching diseases with departments
        """
        df = self.chatbot.symptom_matcher.df
        
        # Case-insensitive partial match
        disease_name_lower = disease_name.lower()
        matches = df[df['disease'].str.lower().str.contains(disease_name_lower, na=False)]
        
        # Return unique disease-department pairs
        unique_matches = matches[['disease', 'department']].drop_duplicates()
        
        return unique_matches.to_dict('records')
    
    def get_department_diseases(self, department: str) -> List[str]:
        """
        Get all diseases in a specific department.
        
        Args:
            department (str): Department name
            
        Returns:
            List[str]: Unique diseases in that department
        """
        df = self.chatbot.symptom_matcher.df
        diseases = df[df['department'] == department]['disease'].unique()
        return sorted(diseases.tolist())


# ============================================================================
# FLASK API EXAMPLE (Optional - not executed, for reference)
# ============================================================================
"""
Example Flask API using this wrapper:

from flask import Flask, request, jsonify
from medical_chatbot_api import MedicalChatbotAPI

app = Flask(__name__)
api = MedicalChatbotAPI()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    symptoms = data.get('symptoms', '')
    
    result = api.analyze_symptoms(symptoms)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def stats():
    return jsonify(api.get_statistics())

@app.route('/api/search/<disease>', methods=['GET'])
def search(disease):
    results = api.search_diseases(disease)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
"""


# ============================================================================
# COMMAND-LINE EXAMPLES
# ============================================================================
def example_usage():
    """Run example usage of the API."""
    
    print("=" * 70)
    print("MEDICAL CHATBOT API - USAGE EXAMPLES")
    print("=" * 70)
    
    # Initialize API
    print("\n[1] Initializing API...")
    api = MedicalChatbotAPI()
    
    # Example 1: Single analysis
    print("\n[2] Analyzing symptom query...")
    result = api.analyze_symptoms("I have a severe headache and fever")
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
    # Example 2: Get statistics
    print("\n[3] Getting dataset statistics...")
    stats = api.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total Records: {stats['total_records']}")
    print(f"  Unique Diseases: {stats['total_unique_diseases']}")
    print(f"  Total Departments: {stats['total_departments']}")
    print(f"\nTop 5 Diseases:")
    for disease, count in list(stats['top_diseases'].items())[:5]:
        print(f"  • {disease}: {count}")
    
    # Example 3: Search diseases
    print("\n[4] Searching for 'fever' diseases...")
    fever_diseases = api.search_diseases("fever")
    print(f"Found {len(fever_diseases)} diseases:")
    for disease_info in fever_diseases[:5]:
        print(f"  • {disease_info['disease']} → {disease_info['department']}")
    
    # Example 4: Get department diseases
    print("\n[5] Getting all Cardiology diseases...")
    cardio_diseases = api.get_department_diseases("Cardiology")
    print(f"Found {len(cardio_diseases)} diseases in Cardiology:")
    for disease in cardio_diseases[:5]:
        print(f"  • {disease}")


if __name__ == "__main__":
    example_usage()
