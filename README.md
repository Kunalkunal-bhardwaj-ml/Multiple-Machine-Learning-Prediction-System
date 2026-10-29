# Multiple-Machine-Learning-Prediction-System
Machine Learning web app for Diabetes, Heart Disease, and Spam Mail Prediction using Python, Scikit-learn, and Streamlit.

A machine learning web application built using Python, Scikit-learn, and Streamlit. This project combines three machine learning prediction systems into a single web application.

## Projects Included

### 1. Diabetes Prediction
Predicts whether a person may have diabetes based on medical input features using a trained machine learning model.

### 2. Heart Disease Prediction
Predicts the possibility of heart disease based on selected medical parameters using a trained machine learning model and saved encoders.

### 3. Spam Mail Detection
Classifies text messages or emails as spam or not spam using a trained machine learning model and TF-IDF vectorization.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Streamlit Option Menu
- TF-IDF Vectorizer
- Pickle

## Project Structure

```text
Multiple-Machine-Learning-Prediction-System/
│
├── Diabetes/
│   ├── README.md
│   ├── Diabetes_Prediction.ipynb
│   └── Diabetes_model.sav
│
├── Heart-Disease/
│   ├── README.md
│   ├── Heart_Disease_Prediction.ipynb
│   ├── heart_disease_model.sav
│   └── heart_disease_encoders.sav
│
├── Spam-Mail/
│   ├── README.md
│   ├── Spam_Mail_Prediction.ipynb
│   ├── spam_mail_model.sav
│   └── spam_mail_vectorizer.sav
│
├── Multiple_Ml_model_Deploy.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Kunalkunal-bhardwaj-ml/Multiple-Machine-Learning-Prediction-System.git
```

### 2. Navigate to the Project Folder

```bash
cd Multiple-Machine-Learning-Prediction-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run Multiple_Ml_model_Deploy.py
```

The application will open in your web browser.

## Model Files

The trained machine learning models, heart disease encoders, and spam mail TF-IDF vectorizer are saved as `.sav` files and used for prediction.

## Learning Objectives

- Understand machine learning model training and prediction.
- Work with data preprocessing and feature encoding.
- Apply TF-IDF vectorization to text data.
- Save and load trained models using Pickle.
- Deploy machine learning models using Streamlit.

## Disclaimer

This project is intended for educational and learning purposes. Medical predictions are not a substitute for professional medical diagnosis.

## Author

**Kunal Bhardwaj**

GitHub: [Kunalkunal-bhardwaj-ml](https://github.com/Kunalkunal-bhardwaj-ml)
