# Practical Implementation: ML, Deep Learning, and NLP

This repository contains a hands-on implementation of machine learning, deep learning, and natural language processing tasks using Python libraries such as **Scikit-learn**, **TensorFlow**, and **spaCy**. The project demonstrates preprocessing, model training, evaluation, and basic ethical considerations in AI.

---

## **Contents**

1. **Classical ML with Scikit-learn**
   - Dataset: Iris Species Dataset
   - Task: Predict iris species using a Decision Tree Classifier
   - Steps:
     - Handle missing values
     - Encode target labels
     - Train/Test split
     - Evaluate using accuracy, precision, and recall

2. **Deep Learning with TensorFlow**
   - Dataset: MNIST Handwritten Digits
   - Task: Classify handwritten digits using a Convolutional Neural Network (CNN)
   - Features:
     - CNN architecture
     - Model training and evaluation
     - Visualization of predictions
     - Achieves >95% test accuracy

3. **Natural Language Processing with spaCy**
   - Dataset: Sample Amazon Product Reviews
   - Task: Extract product names and brands (Named Entity Recognition)
   - Additional: Simple rule-based sentiment analysis (positive/negative)

4. **Ethics & Optimization**
   - Discusses potential biases in ML/DL/NLP models
   - Provides debugging tips and best practices

---

## **Requirements**

- Python 3.7+
- Libraries:
  ```bash
  pip install pandas scikit-learn numpy matplotlib tensorflow spacy
  python -m spacy download en_core_web_sm
  # Jupyter-notebooks-answ
