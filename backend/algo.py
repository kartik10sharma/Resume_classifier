import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import re
import os
from datetime import datetime
import logging
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Flask imports with latest versions
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Document processing imports
import PyPDF2
import docx2txt
import io
from typing import Dict, List, Optional, Tuple

class ResumeJobMatcher:
    """Advanced ML-based Resume-Job Matching System"""
    
    def __init__(self, model_path: str = 'models/'):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        # Enhanced Random Forest with latest sklearn
        self.rf_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )
        
        # Enhanced TF-IDF with better parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            use_idf=True
        )
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.training_data = []
        self.model_version = "2.0"
        
        # Enhanced logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load existing model
        self.load_model()
        
        # Extended skills database
        self.skills_database = {
            'programming': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby'],
            'web': ['react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'html', 'css'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'gitlab'],
            'data_science': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'matplotlib'],
            'soft_skills': ['leadership', 'communication', 'project management', 'agile', 'scrum', 'teamwork']
        }
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, emails, phone numbers
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s+#.-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_skills(self, text: str) -> Dict[str, int]:
        """Extract and categorize skills from text"""
        processed_text = self.preprocess_text(text)
        skill_counts = {}
        
        for category, skills in self.skills_database.items():
            count = sum(1 for skill in skills if skill in processed_text)
            skill_counts[f"{category}_skills"] = count
        
        return skill_counts
    
    def extract_features(self, job_desc: str, resume_text: str) -> Dict:
        """Enhanced feature extraction"""
        # Preprocess texts
        job_processed = self.preprocess_text(job_desc)
        resume_processed = self.preprocess_text(resume_text)
        combined_text = f"{job_processed} {resume_processed}"
        
        # Extract skills
        job_skills = self.extract_skills(job_desc)
        resume_skills = self.extract_skills(resume_text)
        
        # Calculate skill overlap
        skill_overlap = {}
        for category in self.skills_database.keys():
            job_count = job_skills.get(f"{category}_skills", 0)
            resume_count = resume_skills.get(f"{category}_skills", 0)
            
            if job_count > 0:
                overlap = min(resume_count / job_count, 1.0)
            else:
                overlap = 1.0 if resume_count == 0 else 0.5
            
            skill_overlap[f"{category}_overlap"] = overlap
        
        # Text similarity metrics
        job_words = set(job_processed.split())
        resume_words = set(resume_processed.split())
        
        if job_words:
            keyword_overlap = len(job_words.intersection(resume_words)) / len(job_words)
        else:
            keyword_overlap = 0
        
        # Length and complexity metrics
        length_ratio = len(resume_text) / max(len(job_desc), 1)
        word_diversity = len(set(resume_processed.split())) / max(len(resume_processed.split()), 1)
        
        return {
            'text': combined_text,
            'keyword_overlap': keyword_overlap,
            'length_ratio': min(length_ratio, 3.0),
            'word_diversity': word_diversity,
            **skill_overlap,
            **resume_skills
        }
    
    def prepare_training_data(self, data_list: List[Dict]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Prepare enhanced training data"""
        if not data_list:
            return None, None
        
        texts = []
        manual_features = []
        labels = []
        
        for item in data_list:
            features = self.extract_features(item['job_desc'], item['resume'])
            texts.append(features['text'])
            
            # Manual features vector
            feature_vector = [
                features['keyword_overlap'],
                features['length_ratio'],
                features['word_diversity'],
                features['programming_overlap'],
                features['web_overlap'],
                features['database_overlap'],
                features['cloud_overlap'],
                features['data_science_overlap'],
                features['soft_skills_overlap'],
                features['programming_skills'],
                features['web_skills'],
                features['database_skills']
            ]
            
            manual_features.append(feature_vector)
            labels.append(item['match_score'])
        
        # Fit TF-IDF if not already fitted
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        # Combine features
        manual_features = np.array(manual_features)
        combined_features = np.hstack([
            tfidf_features.toarray(),
            manual_features
        ])
        
        return combined_features, labels
    
    def train_model(self, training_data: Optional[List[Dict]] = None) -> bool:
        """Train the Random Forest model with enhanced features"""
        if training_data:
            self.training_data.extend(training_data)
        
        if len(self.training_data) < 15:
            self.logger.warning(f"Need at least 15 training samples, have {len(self.training_data)}")
            return False
        
        try:
            features, labels = self.prepare_training_data(self.training_data)
            if features is None:
                return False
            
            # Encode labels if needed
            if not hasattr(self.label_encoder, 'classes_'):
                labels_encoded = self.label_encoder.fit_transform(labels)
            else:
                labels_encoded = self.label_encoder.transform(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels_encoded, test_size=0.25, random_state=42, stratify=labels_encoded
            )
            
            # Train model
            self.rf_classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.rf_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Model trained successfully!")
            self.logger.info(f"Accuracy: {accuracy:.3f}")
            self.logger.info(f"OOB Score: {self.rf_classifier.oob_score_:.3f}")
            self.logger.info(f"Training samples: {len(self.training_data)}")
            
            self.is_trained = True
            self.save_model()
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def predict_match(self, job_description: str, resume_text: str) -> Dict:
        """Enhanced prediction with detailed analysis"""
        try:
            features = self.extract_features(job_description, resume_text)
            
            if not self.is_trained:
                return self.rule_based_match(features)
            
            # Prepare features for prediction
            text_features = self.tfidf_vectorizer.transform([features['text']])
            manual_features = np.array([[
                features['keyword_overlap'],
                features['length_ratio'],
                features['word_diversity'],
                features['programming_overlap'],
                features['web_overlap'],
                features['database_overlap'],
                features['cloud_overlap'],
                features['data_science_overlap'],
                features['soft_skills_overlap'],
                features['programming_skills'],
                features['web_skills'],
                features['database_skills']
            ]])
            
            combined_features = np.hstack([
                text_features.toarray(),
                manual_features
            ])
            
            # Get predictions
            probabilities = self.rf_classifier.predict_proba(combined_features)[0]
            predicted_class_idx = self.rf_classifier.predict(combined_features)[0]
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            # Get feature importance
            top_features = self.get_feature_importance()
            
            return {
                'match_probability': max(probabilities),
                'predicted_score': predicted_class,
                'class_probabilities': dict(zip(self.label_encoder.classes_, probabilities)),
                'skill_analysis': self.analyze_skills(features),
                'confidence': max(probabilities),
                'top_features': top_features,
                'model_version': self.model_version
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return self.rule_based_match(features)
    
    def analyze_skills(self, features: Dict) -> Dict:
        """Detailed skill analysis"""
        return {
            'programming_match': features['programming_overlap'],
            'web_tech_match': features['web_overlap'],
            'database_match': features['database_overlap'],
            'cloud_match': features['cloud_overlap'],
            'data_science_match': features['data_science_overlap'],
            'soft_skills_match': features['soft_skills_overlap'],
            'keyword_overlap': features['keyword_overlap'],
            'resume_completeness': min(features['length_ratio'], 1.0)
        }
    
    def get_feature_importance(self) -> List[Dict]:
        """Get top feature importance from trained model"""
        if not self.is_trained:
            return []
        
        try:
            importances = self.rf_classifier.feature_importances_
            # Get top 5 features
            top_indices = np.argsort(importances)[-5:][::-1]
            
            feature_names = ['tfidf_features'] * len(importances[:-12]) + [
                'keyword_overlap', 'length_ratio', 'word_diversity',
                'programming_overlap', 'web_overlap', 'database_overlap',
                'cloud_overlap', 'data_science_overlap', 'soft_skills_overlap',
                'programming_skills', 'web_skills', 'database_skills'
            ]
            
            return [
                {'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                 'importance': float(importances[i])}
                for i in top_indices
            ]
        except:
            return []
    
    def rule_based_match(self, features: Dict) -> Dict:
        """Enhanced rule-based fallback matching"""
        # Weighted scoring
        weights = {
            'keyword_overlap': 0.3,
            'programming_overlap': 0.25,
            'web_overlap': 0.15,
            'database_overlap': 0.1,
            'cloud_overlap': 0.1,
            'soft_skills_overlap': 0.1
        }
        
        score = sum(features.get(key, 0) * weight for key, weight in weights.items())
        
        # Apply bonuses and penalties
        if features['word_diversity'] > 0.3:
            score += 0.1
        if features['length_ratio'] > 2.5:
            score -= 0.05
        
        score = max(0.05, min(0.95, score))
        
        # Determine class
        if score > 0.75:
            predicted_score = 'high'
        elif score > 0.45:
            predicted_score = 'medium'
        else:
            predicted_score = 'low'
        
        return {
            'match_probability': score,
            'predicted_score': predicted_score,
            'skill_analysis': self.analyze_skills(features),
            'confidence': score,
            'model_version': 'rule_based',
            'top_features': []
        }
    
    def add_feedback(self, job_desc: str, resume: str, actual_score: str) -> None:
        """Add feedback and trigger retraining"""
        new_data = {
            'job_desc': job_desc,
            'resume': resume,
            'match_score': actual_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_data.append(new_data)
        self.logger.info(f"Added feedback. Total samples: {len(self.training_data)}")
        
        # Retrain every 8 samples for faster adaptation
        if len(self.training_data) % 8 == 0 and len(self.training_data) >= 15:
            self.logger.info("Triggering model retraining...")
            self.train_model()
    
    def save_model(self) -> None:
        """Save model using joblib for better performance"""
        try:
            model_data = {
                'rf_classifier': self.rf_classifier,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'label_encoder': self.label_encoder,
                'training_data': self.training_data,
                'is_trained': self.is_trained,
                'model_version': self.model_version,
                'skills_database': self.skills_database,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, self.model_path / 'resume_matcher_model.joblib')
            self.logger.info("Model saved successfully with joblib")
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> None:
        """Load model using joblib"""
        try:
            model_file = self.model_path / 'resume_matcher_model.joblib'
            if model_file.exists():
                model_data = joblib.load(model_file)
                
                self.rf_classifier = model_data['rf_classifier']
                self.tfidf_vectorizer = model_data['tfidf_vectorizer']
                self.label_encoder = model_data['label_encoder']
                self.training_data = model_data.get('training_data', [])
                self.is_trained = model_data.get('is_trained', False)
                self.skills_database = model_data.get('skills_database', self.skills_database)
                
                self.logger.info(f"Model loaded successfully. Samples: {len(self.training_data)}")
                
        except Exception as e:
            self.logger.info(f"No existing model found or failed to load: {e}")

# Flask Application with latest version
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Enhanced CORS configuration
CORS(app, origins=["http://localhost:3000", "http://localhost:5173"])

# Create upload directory
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Initialize matcher
matcher = ResumeJobMatcher()

def extract_text_from_file(file) -> str:
    """Enhanced file text extraction"""
    filename = secure_filename(file.filename).lower()
    
    try:
        file_content = file.read()
        file.seek(0)  # Reset file pointer
        
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif filename.endswith('.docx'):
            return docx2txt.process(io.BytesIO(file_content))
        
        elif filename.endswith('.doc'):
            # Basic DOC support - might need python-docx2txt for better support
            return file_content.decode('utf-8', errors='ignore')
        
        elif filename.endswith('.txt'):
            return file_content.decode('utf-8')
        
        else:
            return "Unsupported file format. Please use PDF, DOCX, or TXT files."
    
    except Exception as e:
        app.logger.error(f"Error extracting text from {filename}: {e}")
        return f"Error reading file: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': matcher.is_trained,
        'training_samples': len(matcher.training_data),
        'model_version': matcher.model_version
    })

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Validate inputs
        if 'job_description' not in request.form:
            return jsonify({'error': 'Job description is required'}), 400

        # BEFORE (buggy): if 'file' not in request.files:
        # AFTER (fixed):
        if 'resume' not in request.files:
            return jsonify({'error': 'Resume file is required'}), 400

        job_description = request.form['job_description'].strip()

        # BEFORE (buggy): file = request.files['file']
        # AFTER (fixed):
        file = request.files['resume']

        if not job_description:
            return jsonify({'error': 'Job description cannot be empty'}), 400

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Extract text from uploaded file
        resume_text = extract_text_from_file(file)

        if not resume_text or len(resume_text.strip()) < 50:
            return jsonify({'error': 'Could not extract sufficient text from resume'}), 400

        # Get ML prediction
        result = matcher.predict_match(job_description, resume_text)

        response = {
            'success': True,
            'analysis': result,
            'recommendation': get_detailed_recommendation(result),
            'timestamp': datetime.now().isoformat(),
            'file_processed': file.filename
        }

        app.logger.info(f"Analysis completed for {file.filename}")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Analysis error: {e}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/feedback', methods=['POST'])
def add_feedback():
    """Enhanced feedback endpoint"""
    try:
        data = request.get_json()
        
        required_fields = ['job_desc', 'resume', 'actual_score']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        matcher.add_feedback(
            data['job_desc'],
            data['resume'],
            data['actual_score']
        )
        
        return jsonify({
            'success': True,
            'message': 'Feedback added successfully',
            'total_samples': len(matcher.training_data),
            'model_trained': matcher.is_trained
        })
    
    except Exception as e:
        app.logger.error(f"Feedback error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model-stats', methods=['GET'])
def get_model_stats():
    """Get model statistics and performance"""
    return jsonify({
        'is_trained': matcher.is_trained,
        'training_samples': len(matcher.training_data),
        'model_version': matcher.model_version,
        'last_training': datetime.now().isoformat(),
        'feature_count': len(matcher.skills_database),
        'supported_formats': ['.pdf', '.docx', '.txt']
    })

def get_detailed_recommendation(result: Dict) -> Dict:
    """Generate detailed recommendations"""
    prob = result['match_probability']
    skills = result.get('skill_analysis', {})
    
    if prob > 0.8:
        level = "excellent"
        action = "Schedule interview immediately"
    elif prob > 0.65:
        level = "good"
        action = "Consider for interview"
    elif prob > 0.4:
        level = "moderate"
        action = "Review qualifications carefully"
    else:
        level = "poor"
        action = "Not recommended for this position"
    
    # Identify improvement areas
    improvement_areas = []
    for skill, score in skills.items():
        if 'overlap' in skill and score < 0.3:
            improvement_areas.append(skill.replace('_overlap', '').replace('_', ' '))
    
    return {
        'level': level,
        'action': action,
        'probability_percent': round(prob * 100, 1),
        'improvement_areas': improvement_areas[:3],  # Top 3 areas
        'strengths': [k.replace('_', ' ') for k, v in skills.items() 
                     if 'overlap' in k and v > 0.7][:3]
    }

def generate_enhanced_sample_data() -> List[Dict]:
    """Generate comprehensive training data"""
    return [
        {
            'job_desc': 'Senior Python Developer with machine learning experience. React and AWS knowledge preferred. 5+ years experience required.',
            'resume': 'Senior Software Engineer with 6 years Python development. Built ML models using scikit-learn and TensorFlow. React frontend development for 3 years. AWS certified.',
            'match_score': 'high'
        },
        {
            'job_desc': 'Data Scientist position requiring Python, SQL, machine learning, and statistical analysis skills.',
            'resume': 'Marketing Specialist with Excel and PowerBI experience. No programming background. Strong analytical skills.',
            'match_score': 'low'
        },
        {
            'job_desc': 'Full Stack Developer - JavaScript, Node.js, MongoDB, React required. Docker experience preferred.',
            'resume': 'Full Stack Developer with 4 years JavaScript experience. Expert in Node.js, MongoDB, and React. Docker containerization experience.',
            'match_score': 'high'
        },
        {
            'job_desc': 'DevOps Engineer - Kubernetes, Docker, AWS, CI/CD pipelines, infrastructure automation.',
            'resume': 'Software Developer with some Docker experience. Basic AWS knowledge. No Kubernetes experience.',
            'match_score': 'medium'
        },
        {
            'job_desc': 'Frontend Developer - React, TypeScript, modern CSS, responsive design, testing frameworks.',
            'resume': 'Frontend Developer specializing in React and TypeScript. 3+ years building responsive web applications. Jest and Cypress testing.',
            'match_score': 'high'
        },
        {
            'job_desc': 'Project Manager - Agile methodologies, Scrum master certification, team leadership, stakeholder management.',
            'resume': 'Experienced Project Manager with PMP certification. Led agile teams for 5+ years. Strong stakeholder communication skills.',
            'match_score': 'high'
        },
        {
            'job_desc': 'Mobile App Developer - React Native, iOS/Android development, REST APIs, mobile UI/UX.',
            'resume': 'Web Developer with HTML, CSS, JavaScript skills. No mobile development experience. Some REST API knowledge.',
            'match_score': 'low'
        },
        {
            'job_desc': 'Database Administrator - PostgreSQL, MySQL, performance tuning, backup strategies, security.',
            'resume': 'Database Administrator with 4 years PostgreSQL experience. Performance optimization and backup management expertise.',
            'match_score': 'medium'
        }
    ]

if __name__ == '__main__':
    # Initialize with sample data if model doesn't exist
    if not matcher.is_trained:
        app.logger.info("Training initial model with sample data...")
        sample_data = generate_enhanced_sample_data()
        matcher.train_model(sample_data)
    
    # Run Flask app with latest configuration
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True
    )