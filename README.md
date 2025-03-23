# 📩 SMS Spam Classifier

![Spam Detection](https://img.shields.io/badge/Spam-Detection-red) ![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Sklearn-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-orange)

## 🚀 Project Overview
This **SMS Spam Classifier** is a machine learning application that detects whether a given SMS message is **Spam** or **Not Spam (Ham)**. It is built using **Natural Language Processing (NLP)** techniques and deployed as a user-friendly web app using **Streamlit**.

## 🔥 Features
✅ Classifies SMS messages as **Spam** or **Not Spam**  
✅ Uses **TF-IDF Vectorization** and **Machine Learning Models**  
✅ Interactive **Streamlit UI** for real-time classification  
✅ Lightweight and fast execution  

## 📂 Dataset
The model is trained on the famous **SMS Spam Collection Dataset**, which consists of 5,572 SMS messages labeled as **spam** or **ham** (non-spam). The dataset includes various types of spam messages, such as:
- Lottery and prize-winning scams
- Fake promotional offers
- Fraudulent alerts and phishing messages

## 🔄 Workflow & Model Training

### 📝 Data Cleaning & Preprocessing  
- Removed unnecessary columns and handled missing values.  
- Tokenized text and applied **lowercasing, stopword removal, punctuation removal, and lemmatization**.  
- Extracted text-based features like **word count, character count, and sentence count**.  

### 📊 Feature Engineering  
- Used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical vectors.  

### 🤖 Model Training & Evaluation  
- Trained multiple models:  
  ✅ **Naïve Bayes** (Best Performing Model)  
  ✅ **Support Vector Machine (SVM)**  
  ✅ **Decision Trees, Random Forest, XGBoost**  
  ✅ **Voting Classifier** (SVM + Naïve Bayes + Extra Trees)  

- **Evaluation Metrics:** Accuracy, Precision, Recall, Confusion Matrix  

## 📊 Model Comparison & Selection

To determine the best-performing model, we trained multiple classifiers and compared their performance using **Accuracy, Precision, Recall, and F1-score**.

| Model          | Accuracy | Precision (Spam) | Precision (Not Spam) | Recall (Spam) | Recall (Not Spam) | F1-score (Spam) | F1-score (Not Spam) |
|---------------|----------|------------------|----------------------|--------------|------------------|--------------|------------------|
| Gaussian NB   | 0.8646   | 0.4815           | 0.98                 | 0.89         | 0.86             | 0.63         | 0.92             |
| Multinomial NB | 0.9768  | 1.0              | 0.97                 | 0.82         | 1.00             | 0.90         | 0.99             |
| Bernoulli NB  | 0.9874   | 1.0              | 0.99                 | 0.90         | 1.00             | 0.95         | 0.99             |
| SVC           | 0.9836   | 0.9524           | 0.99                 | 0.92         | 0.99             | 0.93         | 0.99             |
| K-Neighbors   | 0.9178   | 1.0              | 0.91                 | 0.35         | 1.00             | 0.52         | 0.96             |
| Decision Tree | 0.9371   | 0.8438           | 0.95                 | 0.62         | 0.98             | 0.71         | 0.96             |
| Logistic Reg. | 0.9613   | 0.9252           | 0.97                 | 0.76         | 0.99             | 0.83         | 0.98             |
| Random Forest | 0.9845   | 1.0              | 0.98                 | 0.88         | 1.00             | 0.93         | 0.99             |
| AdaBoost      | 0.9642   | 0.8983           | 0.97                 | 0.81         | 0.99             | 0.85         | 0.98             |
| Bagging Classifier | 0.9642 | 0.8790        | 0.98                 | 0.83         | 0.98             | 0.85         | 0.98             |
| Extra Trees   | 0.9836   | 0.9672           | 0.99                 | 0.90         | 1.00             | 0.93         | 0.99             |
| GBDT          | 0.9507   | 0.8922           | 0.96                 | 0.69         | 0.99             | 0.78         | 0.97             |
| XGBoost       | 0.9691   | 0.9304           | 0.97                 | 0.82         | 0.99             | 0.87         | 0.98             |


### ✅ **Why Bernoulli Naïve Bayes?**
- **Best Performance:** It achieved the highest overall accuracy (**98.7%**) with a great balance between precision and recall.
- **Lightweight & Fast:** Works efficiently even with minimal computational resources.
- **Robust for Text Data:** Bernoulli NB works well with **binary word presence** features (which is common in spam detection).
- **Better Recall:** Ensures fewer false negatives, meaning **fewer spam messages are misclassified as ham**.

### 🎯 **Final Decision**
Thus, **Bernoulli Naïve Bayes (TF-IDF)** was selected as the final model for deployment. 🚀


### 🚀 Deployment  
- Integrated the best model into a **Streamlit Web App**.  
- Hosted the application on **Render** for public use.  

## 📌 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

### 3️⃣ Enter a message and classify it!

## 🖼️ UI Preview
![Spam Classifier UI] https://github.com/Ranjeet-Kumar60/sms-spam-detector/blob/main/Screenshot%202025-03-22%20215000.png

## 📝 Future Improvements
- Improve model accuracy with **Deep Learning (LSTM/BERT)**
- Handle **emojis and special characters** better

## 🤝 Contributing
Want to improve this project? Feel free to fork, submit pull requests, or open issues!

## 📜 License
This project is open-source and available under the **MIT License**.

---
🚀 **Built with passion for NLP and AI!**
