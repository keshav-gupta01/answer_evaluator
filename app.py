from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Teacher credentials
TEACHER_PASSWORD = "admin123"  # Change this to your desired password

class AnswerEvaluator:
    def __init__(self, data_file='answers.json'):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.data_file = data_file
        self.sample_answers = {}  # Initialize empty dict
        self.load_answers()
    
    def load_answers(self):
        try:
            if os.path.exists(self.data_file) and os.path.getsize(self.data_file) > 0:
                with open(self.data_file, 'r') as f:
                    self.sample_answers = json.load(f)
            else:
                # Create a new file with empty dictionary if it doesn't exist
                with open(self.data_file, 'w') as f:
                    json.dump({}, f)
                self.sample_answers = {}
        except json.JSONDecodeError:
            # If file is corrupted, initialize with empty dictionary
            self.sample_answers = {}
            with open(self.data_file, 'w') as f:
                json.dump({}, f)
    
    def save_answers(self):
        with open(self.data_file, 'w') as f:
            json.dump(self.sample_answers, f, indent=4)
    
    def add_answer(self, subject, question, answer, is_paragraph=False):
        if subject not in self.sample_answers:
            self.sample_answers[subject] = {}
        
        self.sample_answers[subject][question] = {
            'answer': answer,
            'is_paragraph': is_paragraph
        }
        self.save_answers()
        
    def evaluate(self, subject, question, student_answer):
        if subject not in self.sample_answers or question not in self.sample_answers[subject]:
            return {'error': 'Question not found'}
        
        sample = self.sample_answers[subject][question]
        correct_answer = sample['answer']
        
        if not sample['is_paragraph']:
            # For single word/short answers
            score = 1.0 if student_answer.lower().strip() == correct_answer.lower().strip() else 0.0
            feedback = "Correct!" if score > 0.9 else "Incorrect. The right answer was: " + correct_answer
        else:
            # For paragraph answers using TF-IDF
            try:
                tfidf = self.vectorizer.fit_transform([correct_answer, student_answer])
                score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
                
                if score >= 0.8:
                    feedback = "Excellent answer! Very close to the expected response."
                elif score >= 0.6:
                    feedback = "Good answer, but could use some improvement."
                else:
                    feedback = "Your answer needs significant improvement. Key points missing."
            except Exception as e:
                return {'error': str(e)}
        
        return {
            'score': round(score * 100, 2),
            'feedback': feedback
        }

# Initialize the evaluator
evaluator = AnswerEvaluator()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == TEACHER_PASSWORD:
            session['is_teacher'] = True
            return redirect(url_for('teacher_panel'))
        else:
            return render_template('login.html', error="Incorrect password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('is_teacher', None)
    return redirect(url_for('home'))

@app.route('/teacher')
def teacher_panel():
    if not session.get('is_teacher'):
        return redirect(url_for('login'))
    return render_template('teacher.html')

@app.route('/student')
def student_panel():
    return render_template('student.html')

@app.route('/api/add_answer', methods=['POST'])
def add_answer():
    if not session.get('is_teacher'):
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json()
    evaluator.add_answer(
        data['subject'],
        data['question'],
        data['answer'],
        data.get('is_paragraph', False)
    )
    return jsonify({'status': 'success'})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_answer():
    data = request.get_json()
    result = evaluator.evaluate(
        data['subject'],
        data['question'],
        data['answer']
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)