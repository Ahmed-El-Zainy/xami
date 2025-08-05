# utils/arabic_utils.py
import re
import string
from typing import List, Dict
import pyarabic.araby as araby
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class ArabicTextProcessor:
    def __init__(self):
        self.arabic_stopwords = set(stopwords.words('arabic')) if 'arabic' in stopwords.fileids() else set()
        # Add common Arabic stopwords manually if needed
        self.arabic_stopwords.update([
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'هذا', 'هذه', 'ذلك', 'تلك',
            'الذي', 'التي', 'التي', 'اللذان', 'اللتان', 'الذين', 'اللاتي', 'اللواتي',
            'وهو', 'وهي', 'وهم', 'وهن', 'له', 'لها', 'لهم', 'لهن', 'به', 'بها', 'بهم', 'بهن',
            'فيه', 'فيها', 'فيهم', 'فيهن', 'عليه', 'عليها', 'عليهم', 'عليهن',
            'كان', 'كانت', 'كانوا', 'كن', 'يكون', 'تكون', 'يكونوا', 'يكن',
            'هناك', 'هنا', 'حيث', 'كيف', 'متى', 'أين', 'ماذا', 'لماذا', 'كم'
        ])
        
        self.english_stopwords = set(stopwords.words('english'))
    
    def clean_arabic_text(self, text: str) -> str:
        """Clean Arabic text by removing diacritics, normalizing, etc."""
        if not text:
            return ""
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove diacritics
        text = araby.strip_diacritics(text)
        
        # Normalize Arabic text
        text = araby.normalize_hamza(text)
        text = araby.normalize_alef(text)
        text = araby.normalize_teh(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Arabic punctuation
        text = re.sub(r'[^\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\u0590-\u05ff\w\s\.\،\؛\؟\!\-\(\)]', '', text)
        
        return text.strip()
    
    def remove_stopwords(self, text: str, language: str = "ar") -> str:
        """Remove stopwords from text"""
        words = text.split()
        
        if language == "ar":
            stopwords_set = self.arabic_stopwords
        elif language == "en":
            stopwords_set = self.english_stopwords
        else:
            # Mixed language - remove both
            stopwords_set = self.arabic_stopwords.union(self.english_stopwords)
        
        filtered_words = [word for word in words if word.lower() not in stopwords_set]
        return ' '.join(filtered_words)
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Arabic or English"""
        if not text:
            return "unknown"
        
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06ff')
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.7:
            return "ar"
        elif arabic_ratio < 0.3:
            return "en"
        else:
            return "mixed"
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences, handling both Arabic and English"""
        # Arabic sentence endings
        arabic_endings = r'[\.!\?؟\.\!\؟]'
        
        # Split by sentence endings
        sentences = re.split(arabic_endings, text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from Arabic text"""
        # Clean text
        cleaned_text = self.clean_arabic_text(text)
        
        # Remove stopwords
        language = self.detect_language(cleaned_text)
        text_without_stopwords = self.remove_stopwords(cleaned_text, language)
        
        # Split into words
        words = text_without_stopwords.split()
        
        # Filter words (minimum length 3, maximum length 20)
        filtered_words = [
            word for word in words 
            if 3 <= len(word) <= 20 and not word.isdigit()
        ]
        
        # Count frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top_k
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def normalize_for_search(self, text: str) -> str:
        """Normalize text for search queries"""
        # Clean text
        text = self.clean_arabic_text(text)
        
        # Convert to lowercase (for mixed content)
        text = text.lower()
        
        # Additional normalization for search
        text = re.sub(r'[أإآ]', 'ا', text)  # Normalize Alef variations
        text = re.sub(r'[ىي]', 'ي', text)   # Normalize Yeh variations
        text = re.sub(r'[ةه]', 'ه', text)   # Normalize Teh Marbuta
        
        return text
    
    def is_arabic_text(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        return bool(re.search(r'[\u0600-\u06ff]', text))
    
    def get_text_statistics(self, text: str) -> Dict:
        """Get basic statistics about the text"""
        if not text:
            return {
                "total_chars": 0,
                "total_words": 0,
                "arabic_chars": 0,
                "english_chars": 0,
                "sentences": 0,
                "language": "unknown"
            }
        
        words = text.split()
        sentences = self.segment_sentences(text)
        arabic_chars = sum(1 for char in text if '\u0600' <= char <= '\u06ff')
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        return {
            "total_chars": len(text),
            "total_words": len(words),
            "arabic_chars": arabic_chars,
            "english_chars": english_chars,
            "sentences": len(sentences),
            "language": self.detect_language(text)
        }

# Global text processor instance
arabic_processor = ArabicTextProcessor()