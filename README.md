# 📄 Resume Classifier

Automated resume classification tool built using machine learning and NLP techniques.

This project aims to **classify resumes** into predefined categories based on their content — using text processing, feature extraction and machine learning models — helping to automate part of the HR screening process and accelerate candidate evaluation.

---

## 🚀 Features

-  Parses resume text (from PDF/TXT).
-  Applies text preprocessing (tokenization, cleaning, vectorization).
-  Trains a classification model to categorize resumes.
-  Easily extensible to add more categories or models.

---

## 🧠 How It Works

1. **Data Collection & Cleaning** – Extract text from resumes
2. **Preprocessing** – Remove punctuation, lowercasing, stopwords, tokenization
3. **Feature Extraction** – Convert text to numerical features (e.g., TF-IDF)
4. **Model Training** – Use a classifier (e.g., SVM, Naive Bayes, Random Forest)
5. **Prediction** – Classify new resume text into job categories

---

## 📁 Project Structure

```
Resume_classifier/
├── app/                # Application logic (parsing & model inference)
├── backend/            # Backend scripts (training / API / utilities)
├── requirements.txt    # Python dependencies
├── data/               # (Optional) sample/processed datasets
├── models/             # Trained model files
├── notebooks/          # Jupyter notebooks (EDA, experimentation)
├── scripts/            # Helper scripts (preprocessing, evaluation)
└── README.md           # Project documentation
```

---

## 🛠 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/kartik10sharma/Resume_classifier.git
   cd Resume_classifier
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data & train model**
   Customize/train with your dataset:

   ```bash
   python backend/train.py
   ```

5. **Run classification**

   ```bash
   python app/predict.py --resume path/to/resume.pdf
   ```

---

## 🧪 Usage Example

```bash
# Example command to classify a resume
python app/predict.py --resume resumes/john_doe_resume.pdf
```

Output:

```
Resume: john_doe_resume.pdf
Predicted Category: Software Engineer
Confidence: 92%
```

---

## 📦 Dependencies

| Package                              | Purpose                 |
| ------------------------------------ | ----------------------- |
| `scikit-learn`                       | Machine learning models |
| `nltk` / `spaCy`                     | NLP preprocessing       |
| `pandas`, `numpy`                    | Data handling           |
| `PyPDF2` / text extraction libraries | Resume text parsing     |

Install all by running:

```bash
pip install -r requirements.txt
```

---

## 📈 Model Performance

You can evaluate performance using standard metrics like:

* **Accuracy**
* **Precision / Recall**
* **F1-Score**
* **Confusion Matrix**
---

## 🧩 Customization

Want to improve accuracy or add features? Consider:

* Adding support for more file formats (DOCX, TXT)
* Using advanced NLP models (BERT, transformers)
* Integrating with a web interface/API
* Supporting multiple languages

---

## 🤝 Contributing

Contributions are welcome! Here’s how you can help:

1. Fork the repository
2. Create a new branch

   ```bash
   git checkout -b feature/your-feature
   ```
3. Make your changes
4. Commit them

   ```bash
   git commit -m "Add new feature"
   ```
5. Push and open a pull request

---

## 📄 License

This project is **open-source** and available under the **MIT License**.


## Contribution Guidelines
- **Fork the repository**: Create a fork of the repository on your GitHub account.
- **Clone your fork**: Clone the forked repository and create a new branch for your feature or bug fix.
- **Make your changes**: Implement your changes and ensure to write tests if applicable.
- **Commit and Push**: Commit your changes and push to your fork.
- **Create a Pull Request**: Submit a pull request to the original repository.
