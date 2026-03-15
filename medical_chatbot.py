"""
Multilingual Medical Symptom Chatbot
======================================
An intelligent chatbot that detects your input language, translates it to English,
matches your symptoms with medical conditions, and responds in your original language.

Libraries Used:
- langdetect: Fast language detection (supports 55+ languages)
- googletrans: Google Translate API for free translation
- scikit-learn: TF-IDF vectorizer and cosine similarity for text matching
- pandas: Data manipulation and analysis
- numpy: Numerical operations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from typing import Tuple, Dict, List

# Language detection and translation
try:
    from langdetect import detect, detect_langs
    from googletrans import Translator
except ImportError:
    print("ERROR: Required libraries not installed.")
    print("Install them using: pip install langdetect googletrans")
    exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score (0-1) to consider a match
TOP_N_MATCHES = 3  # Number of top matches to consider
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ja': 'Japanese',
    'ar': 'Arabic',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'it': 'Italian',
}


# ============================================================================
# LANGUAGE DETECTION AND TRANSLATION
# ============================================================================
class LanguageProcessor:
    """Handles language detection and translation."""
    
    def __init__(self):
        self.translator = Translator()
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Language code (e.g., 'en', 'es', 'fr')
        """
        try:
            lang = detect(text)
            return lang
        except Exception as e:
            print(f"Error detecting language: {e}")
            return 'en'  # Default to English
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text to English if it's not already.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            
        Returns:
            str: Translated text in English
        """
        if source_lang == 'en':
            return text
        
        try:
            translation = self.translator.translate(text, src_language=source_lang, dest_language='en')
            return translation['text']
        except Exception as e:
            print(f"Error translating text: {e}")
            return text  # Return original text if translation fails
    
    def translate_to_language(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text (str): English text
            target_lang (str): Target language code
            
        Returns:
            str: Translated text
        """
        if target_lang == 'en':
            return text
        
        try:
            translation = self.translator.translate(text, src_language='en', dest_language=target_lang)
            return translation['text']
        except Exception as e:
            print(f"Error translating response: {e}")
            return text  # Return original text if translation fails


# ============================================================================
# SYMPTOM MATCHING ENGINE
# ============================================================================
class SymptomMatcher:
    """Handles symptom matching using TF-IDF and cosine similarity."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the matcher with dataset.
        
        Args:
            dataset_path (str): Path to the CSV dataset
        """
        print("Loading dataset...")
        self.df = pd.read_csv(dataset_path)
        
        # Create TF-IDF vectorizer for symptom descriptions
        print("Building TF-IDF vectors (this may take a moment)...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,           # Limit vocabulary size
            min_df=1,                    # Minimum document frequency
            max_df=0.9,                  # Maximum document frequency
            ngram_range=(1, 2),          # Use unigrams and bigrams
            stop_words='english',        # Remove common English words
            lowercase=True
        )
        
        # Fit vectorizer on symptom descriptions
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['input_text'])
        print(f"✓ Dataset loaded: {len(self.df)} records")
        print(f"✓ TF-IDF matrix built: {self.tfidf_matrix.shape}")
    
    def find_matching_diseases(self, query: str, top_n: int = TOP_N_MATCHES) -> List[Dict]:
        """
        Find the most relevant diseases based on symptom query.
        
        Args:
            query (str): User's symptom description (in English)
            top_n (int): Number of top matches to return
            
        Returns:
            List[Dict]: List of matches with disease, department, and similarity score
        """
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity with all dataset entries
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top N matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        matches = []
        
        for idx in top_indices:
            similarity_score = similarities[idx]
            
            # Only include matches above threshold
            if similarity_score >= SIMILARITY_THRESHOLD:
                matches.append({
                    'disease': self.df.iloc[idx]['disease'],
                    'department': self.df.iloc[idx]['department'],
                    'similarity': float(similarity_score),
                    'description': self.df.iloc[idx]['input_text'][:100] + "..."
                })
        
        return matches


# ============================================================================
# MEDICAL CHATBOT
# ============================================================================
class MedicalChatbot:
    """Main chatbot class that orchestrates language processing and symptom matching."""
    
    def __init__(self, dataset_path: str):
        """
        Initialize the chatbot.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.language_processor = LanguageProcessor()
        self.symptom_matcher = SymptomMatcher(dataset_path)
        print("\n✓ Chatbot initialized and ready!")
    
    def process_query(self, user_input: str) -> Dict:
        """
        Process user query end-to-end.
        
        Args:
            user_input (str): User's query in any language
            
        Returns:
            Dict: Response containing disease, department, and translated response
        """
        # Step 1: Detect language
        print(f"\n[1/5] Detecting language...")
        detected_lang = self.language_processor.detect_language(user_input)
        print(f"    ✓ Detected language: {SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}")
        
        # Step 2: Translate to English
        print(f"[2/5] Translating to English...")
        english_text = self.language_processor.translate_to_english(user_input, detected_lang)
        print(f"    ✓ English translation: \"{english_text}\"")
        
        # Step 3: Match symptoms
        print(f"[3/5] Searching for matching diseases...")
        matches = self.symptom_matcher.find_matching_diseases(english_text)
        
        if not matches:
            print(f"    ✗ No suitable matches found")
            response = self._generate_no_match_response()
        else:
            print(f"    ✓ Found {len(matches)} match(es)")
            response = self._generate_response(matches)
        
        # Step 4: Translate response back to original language
        print(f"[4/5] Translating response to {SUPPORTED_LANGUAGES.get(detected_lang, detected_lang)}...")
        translated_response = self.language_processor.translate_to_language(
            response['message'], 
            detected_lang
        )
        response['message'] = translated_response
        print(f"    ✓ Response translated")
        
        # Step 5: Return result
        print(f"[5/5] Complete!")
        
        return {
            'user_language': SUPPORTED_LANGUAGES.get(detected_lang, detected_lang),
            'original_input': user_input,
            'english_translation': english_text,
            'matches': matches,
            'response': response['message'],
            'confidence': matches[0]['similarity'] if matches else 0.0
        }
    
    def _generate_response(self, matches: List[Dict]) -> Dict:
        """Generate a response based on matched diseases."""
        top_match = matches[0]
        
        if len(matches) == 1:
            message = (
                f"Based on your symptoms, the most likely condition is: {top_match['disease']}. "
                f"This falls under the {top_match['department']} department. "
                f"Confidence: {top_match['similarity']:.1%}"
            )
        else:
            message = (
                f"Based on your symptoms, the most likely conditions are:\n"
            )
            for i, match in enumerate(matches, 1):
                message += f"{i}. {match['disease']} ({match['department']}) - Confidence: {match['similarity']:.1%}\n"
            message += (
                f"\nThe top recommendation is: {top_match['disease']} "
                f"(Department: {top_match['department']})"
            )
        
        return {'message': message}
    
    def _generate_no_match_response(self) -> Dict:
        """Generate a response when no matches are found."""
        message = (
            "I couldn't find a close match for your symptoms in the database. "
            "Please consult with a medical professional or try describing your "
            "symptoms in more detail. For urgent concerns, please visit an Emergency Department."
        )
        return {'message': message}
    
    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "=" * 70)
        print("MULTILINGUAL MEDICAL SYMPTOM CHATBOT")
        print("=" * 70)
        print("\nWelcome! I can understand and respond in multiple languages.")
        print("Describe your symptoms in any language you prefer.")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("-" * 70)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🩺 You: ").strip()
                
                if not user_input:
                    print("Please describe your symptoms.")
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Thank you for using the Medical Chatbot. Stay healthy!")
                    break
                
                # Process query
                result = self.process_query(user_input)
                
                # Display results
                print("\n" + "-" * 70)
                if result['matches']:
                    print(f"\n📋 Top Match: {result['matches'][0]['disease']}")
                    print(f"🏥 Department: {result['matches'][0]['department']}")
                    print(f"🎯 Confidence: {result['confidence']:.1%}")
                
                print(f"\n💬 Response:\n{result['response']}")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                continue


# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main function to run the chatbot."""
    import sys
    
    # Initialize chatbot
    dataset_path = 'symptom_sentence_dataset_with_department.csv'
    
    try:
        chatbot = MedicalChatbot(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found.")
        print("Make sure you've run 'add_department_column.py' first.")
        sys.exit(1)
    
    # Start interactive chat
    chatbot.chat()


if __name__ == "__main__":
    main()
