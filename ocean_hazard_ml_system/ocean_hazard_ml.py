# Ocean Hazard Detection ML System
# Complete implementation for SIH Hackathon

import os
import re
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

# ML Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Flask API
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HazardAnalysis:
    """Data structure for ML analysis results"""
    hazard_type: str
    confidence: float
    severity_score: int  # 1-10 scale
    urgency_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    credibility_score: float  # 0-1 scale
    keywords_found: List[str]
    sentiment: str
    location_extracted: Optional[str] = None

class OceanHazardML:
    """Complete ML system for ocean hazard detection and analysis"""
    
    def __init__(self):
        self.setup_models()
        self.setup_keywords()
        self.setup_location_patterns()
        
    def setup_models(self):
        """Initialize all ML models"""
        try:
            # Sentiment Analysis Model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # Text Classification Model for Hazard Types
            self.hazard_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Load spaCy for NLP preprocessing
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
            logger.info("All ML models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def setup_keywords(self):
        """Define hazard-specific keywords with weights"""
        self.hazard_keywords = {
            "tsunami": {
                "keywords": ["tsunami", "tidal wave", "sea wall", "giant wave", "ocean wave", "seismic wave"],
                "weight": 1.0,
                "urgent_words": ["evacuate", "run", "high ground", "warning", "siren"]
            },
            "flood": {
                "keywords": ["flood", "flooding", "waterlogging", "inundation", "overflow", "submerged", "drowning"],
                "weight": 0.9,
                "urgent_words": ["trapped", "rescue", "emergency", "help", "stuck"]
            },
            "storm_surge": {
                "keywords": ["storm surge", "storm", "cyclone", "hurricane", "typhoon", "wind", "storm tide"],
                "weight": 0.8,
                "urgent_words": ["shelter", "evacuate", "dangerous", "category", "landfall"]
            },
            "high_waves": {
                "keywords": ["high waves", "rough sea", "big waves", "choppy", "swells", "breakers", "surf"],
                "weight": 0.7,
                "urgent_words": ["avoid", "dangerous", "warning", "stay away"]
            },
            "coastal_erosion": {
                "keywords": ["erosion", "beach erosion", "cliff collapse", "shoreline", "retreat", "sand loss"],
                "weight": 0.6,
                "urgent_words": ["unstable", "collapse", "danger zone"]
            },
            "normal": {
                "keywords": ["calm", "peaceful", "clear", "sunny", "beautiful", "pleasant"],
                "weight": 0.1,
                "urgent_words": []
            }
        }
        
        # Critical urgency indicators
        self.critical_words = [
            "emergency", "disaster", "catastrophe", "help", "rescue", "trapped", 
            "evacuate", "danger", "warning", "alert", "critical", "urgent",
            "death", "casualties", "missing", "destroyed", "devastation"
        ]
        
        # Credibility indicators
        self.credibility_boosters = [
            "witnessed", "saw", "observed", "confirmed", "official", "reported",
            "measured", "recorded", "verified", "authenticated"
        ]
        
        self.credibility_reducers = [
            "heard", "rumor", "maybe", "possibly", "might", "could be",
            "unconfirmed", "allegedly", "supposedly"
        ]
    
    def setup_location_patterns(self):
        """Setup patterns for location extraction"""
        self.location_patterns = [
            r'\b(?:at|in|near|around|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:coast|beach|shore|harbor|port)\b',
            r'\b(?:coast|beach|shore)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = text.lower().strip()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def extract_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract hazard-specific keywords from text"""
        text_lower = text.lower()
        found_keywords = {}
        
        for hazard_type, hazard_info in self.hazard_keywords.items():
            keywords = []
            
            # Check main keywords
            for keyword in hazard_info["keywords"]:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            # Check urgent words
            urgent_found = []
            for urgent_word in hazard_info["urgent_words"]:
                if urgent_word in text_lower:
                    urgent_found.append(urgent_word)
            
            if keywords or urgent_found:
                found_keywords[hazard_type] = {
                    "main_keywords": keywords,
                    "urgent_keywords": urgent_found,
                    "weight": hazard_info["weight"]
                }
        
        return found_keywords
    
    def calculate_credibility_score(self, text: str) -> float:
        """Calculate credibility score based on text content"""
        text_lower = text.lower()
        
        boost_count = sum(1 for word in self.credibility_boosters if word in text_lower)
        reduce_count = sum(1 for word in self.credibility_reducers if word in text_lower)
        
        # Base credibility
        base_score = 0.5
        
        # Adjust based on indicators
        credibility = base_score + (boost_count * 0.15) - (reduce_count * 0.1)
        
        # Check for specific patterns that increase credibility
        if any(pattern in text_lower for pattern in ["i saw", "i witnessed", "happening now"]):
            credibility += 0.2
        
        if any(pattern in text_lower for pattern in ["photo attached", "video", "image"]):
            credibility += 0.15
        
        return max(0.0, min(1.0, credibility))
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract location information from text"""
        if not self.nlp:
            return None
            
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        
        if locations:
            return locations[0]
        
        # Fallback to regex patterns
        for pattern in self.location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """Analyze sentiment and return label with confidence"""
        try:
            results = self.sentiment_analyzer(text)[0]
            
            # Find the highest scoring sentiment
            best_sentiment = max(results, key=lambda x: x['score'])
            
            # Map to simpler categories
            sentiment_mapping = {
                'LABEL_0': 'negative',  # Often indicates urgency/danger
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive',
                'negative': 'negative',
                'neutral': 'neutral', 
                'positive': 'positive'
            }
            
            sentiment = sentiment_mapping.get(best_sentiment['label'].lower(), best_sentiment['label'].lower())
            confidence = best_sentiment['score']
            
            return sentiment, confidence
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "neutral", 0.5
    
    def classify_hazard_type(self, text: str, keywords_found: Dict) -> Tuple[str, float]:
        """Determine the most likely hazard type"""
        
        # If we have keyword matches, use them
        if keywords_found:
            # Calculate weighted scores for each hazard type
            hazard_scores = {}
            for hazard_type, info in keywords_found.items():
                keyword_count = len(info["main_keywords"]) + len(info["urgent_keywords"]) * 2
                hazard_scores[hazard_type] = keyword_count * info["weight"]
            
            if hazard_scores:
                best_hazard = max(hazard_scores, key=hazard_scores.get)
                confidence = min(0.95, hazard_scores[best_hazard] / 10)  # Normalize
                return best_hazard, confidence
        
        # Fallback to zero-shot classification
        try:
            candidate_labels = list(self.hazard_keywords.keys())
            result = self.hazard_classifier(text, candidate_labels)
            return result['labels'][0], result['scores'][0]
        except Exception as e:
            logger.error(f"Hazard classification error: {e}")
            return "normal", 0.3
    
    def calculate_severity_score(self, text: str, hazard_type: str, sentiment: str, 
                               keywords_found: Dict, credibility: float) -> int:
        """Calculate severity score from 1-10"""
        
        base_severity = {
            "tsunami": 9,
            "flood": 7,
            "storm_surge": 8,
            "high_waves": 5,
            "coastal_erosion": 4,
            "normal": 1
        }
        
        severity = base_severity.get(hazard_type, 3)
        
        # Adjust based on sentiment (negative = more severe)
        if sentiment == "negative":
            severity += 1
        elif sentiment == "positive":
            severity -= 1
        
        # Check for critical words
        text_lower = text.lower()
        critical_count = sum(1 for word in self.critical_words if word in text_lower)
        severity += min(3, critical_count)
        
        # Adjust based on credibility
        if credibility > 0.8:
            severity += 1
        elif credibility < 0.3:
            severity -= 1
        
        # Check for urgent keywords in the hazard type
        if hazard_type in keywords_found:
            urgent_keywords = keywords_found[hazard_type]["urgent_keywords"]
            if urgent_keywords:
                severity += len(urgent_keywords)
        
        return max(1, min(10, severity))
    
    def determine_urgency_level(self, severity_score: int, hazard_type: str) -> str:
        """Determine urgency level based on severity and hazard type"""
        
        high_priority_hazards = ["tsunami", "storm_surge"]
        
        if hazard_type in high_priority_hazards:
            if severity_score >= 8:
                return "CRITICAL"
            elif severity_score >= 6:
                return "HIGH"
            elif severity_score >= 4:
                return "MEDIUM"
            else:
                return "LOW"
        else:
            if severity_score >= 9:
                return "CRITICAL"
            elif severity_score >= 7:
                return "HIGH"
            elif severity_score >= 5:
                return "MEDIUM"
            else:
                return "LOW"
    
    def analyze_text(self, text: str, include_location: bool = True) -> HazardAnalysis:
        """Main function to analyze text and return complete analysis"""
        
        if not text or len(text.strip()) < 5:
            return HazardAnalysis(
                hazard_type="normal",
                confidence=0.9,
                severity_score=1,
                urgency_level="LOW",
                credibility_score=0.5,
                keywords_found=[],
                sentiment="neutral"
            )
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract keywords
        keywords_found = self.extract_keywords(processed_text)
        
        # Analyze sentiment
        sentiment, sentiment_confidence = self.analyze_sentiment(processed_text)
        
        # Classify hazard type
        hazard_type, hazard_confidence = self.classify_hazard_type(processed_text, keywords_found)
        
        # Calculate credibility
        credibility_score = self.calculate_credibility_score(processed_text)
        
        # Calculate severity
        severity_score = self.calculate_severity_score(
            processed_text, hazard_type, sentiment, keywords_found, credibility_score
        )
        
        # Determine urgency
        urgency_level = self.determine_urgency_level(severity_score, hazard_type)
        
        # Extract location if requested
        location = None
        if include_location:
            location = self.extract_location(text)
        
        # Compile all found keywords
        all_keywords = []
        for hazard_info in keywords_found.values():
            all_keywords.extend(hazard_info["main_keywords"])
            all_keywords.extend(hazard_info["urgent_keywords"])
        
        return HazardAnalysis(
            hazard_type=hazard_type,
            confidence=hazard_confidence,
            severity_score=severity_score,
            urgency_level=urgency_level,
            credibility_score=credibility_score,
            keywords_found=all_keywords,
            sentiment=sentiment,
            location_extracted=location
        )
    
    def analyze_image_basic(self, image_path: str) -> Dict:
        """Basic image analysis for water/flooding detection"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define water color ranges (blue tones)
            water_ranges = [
                # Light blue water
                {"lower": np.array([100, 50, 50]), "upper": np.array([130, 255, 255])},
                # Darker water
                {"lower": np.array([90, 30, 30]), "upper": np.array([120, 255, 200])},
                # Muddy water (brown-ish)
                {"lower": np.array([10, 50, 20]), "upper": np.array([20, 255, 200])}
            ]
            
            total_water_pixels = 0
            total_pixels = image.shape[0] * image.shape[1]
            
            for water_range in water_ranges:
                mask = cv2.inRange(hsv, water_range["lower"], water_range["upper"])
                total_water_pixels += cv2.countNonZero(mask)
            
            water_percentage = (total_water_pixels / total_pixels) * 100
            
            # Basic analysis
            analysis = {
                "water_percentage": round(water_percentage, 2),
                "likely_flood": water_percentage > 25,
                "severity": "high" if water_percentage > 40 else "medium" if water_percentage > 15 else "low",
                "confidence": 0.7 if water_percentage > 30 else 0.5
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {"error": str(e)}

# Flask API Implementation
app = Flask(__name__)
CORS(app)

# Initialize ML system
ml_system = OceanHazardML()

@app.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/ml/analyze-text', methods=['POST'])
def analyze_text_endpoint():
    """Endpoint for text analysis"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        include_location = data.get('include_location', True)
        
        # Perform analysis
        analysis = ml_system.analyze_text(text, include_location)
        
        # Convert to dict for JSON response
        result = {
            "hazard_type": analysis.hazard_type,
            "confidence": round(analysis.confidence, 3),
            "severity_score": analysis.severity_score,
            "urgency_level": analysis.urgency_level,
            "credibility_score": round(analysis.credibility_score, 3),
            "keywords_found": analysis.keywords_found,
            "sentiment": analysis.sentiment,
            "location_extracted": analysis.location_extracted,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Text analysis endpoint error: {e}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route('/ml/analyze-image', methods=['POST'])
def analyze_image_endpoint():
    """Endpoint for image analysis"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "Image file is required"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Save temporarily
        temp_path = f"temp_{datetime.now().timestamp()}_{file.filename}"
        file.save(temp_path)
        
        try:
            # Analyze image
            analysis = ml_system.analyze_image_basic(temp_path)
            analysis["timestamp"] = datetime.now().isoformat()
            
            return jsonify(analysis)
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Image analysis endpoint error: {e}")
        return jsonify({"error": "Image analysis failed", "details": str(e)}), 500

@app.route('/ml/batch-analyze', methods=['POST'])
def batch_analyze_endpoint():
    """Endpoint for analyzing multiple texts at once"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "Texts array is required"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "Texts must be an array"}), 400
        
        results = []
        for i, text in enumerate(texts):
            try:
                analysis = ml_system.analyze_text(text)
                result = {
                    "id": i,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "analysis": {
                        "hazard_type": analysis.hazard_type,
                        "confidence": round(analysis.confidence, 3),
                        "severity_score": analysis.severity_score,
                        "urgency_level": analysis.urgency_level,
                        "credibility_score": round(analysis.credibility_score, 3),
                        "keywords_found": analysis.keywords_found,
                        "sentiment": analysis.sentiment
                    }
                }
                results.append(result)
            except Exception as e:
                results.append({
                    "id": i,
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "error": str(e)
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(results),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch analysis endpoint error: {e}")
        return jsonify({"error": "Batch analysis failed", "details": str(e)}), 500

if __name__ == '__main__':
    print("🌊 Ocean Hazard ML System Starting...")
    print("📊 Available endpoints:")
    print("  - POST /ml/analyze-text")
    print("  - POST /ml/analyze-image") 
    print("  - POST /ml/batch-analyze")
    print("  - GET /health")
    print("\n🚀 Starting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)