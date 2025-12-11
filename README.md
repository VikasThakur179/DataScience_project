# DataScience_project

#  Stock price 
This project predicts short-term stock price movements using Machine Learning and Deep Learning (LSTM).  
It fetches **live stock data**, preprocesses it, and predicts the next closing price or price trend.

---

##  Tech Stack
- **Language:** Python 3.11.9 
- **Libraries:** pandas, numpy, scikit-learn, tensorflow/keras, matplotlib, yfinance  
- **API Framework:** FastAPI  
- **UI:** Streamlit  

## Install dependices
pip install -r requirements.txt

## How to run 
streamlit run app.py --server.port 5000
<img width="1900" height="883" alt="image" src="https://github.com/user-attachments/assets/d5d43572-1b8d-4cee-a3c8-6a97e2daa676" />




#  LinguaScan – Multi-Language Detection App  
A powerful and lightweight **language detection web application** built using **Python, Streamlit, LangDetect, and a custom N-gram ML model**.  
LinguaScan can detect **50+ languages** from both **single text input** and **batch CSV/TXT datasets**.

##  Features
###  **1. Single Text Language Detection**
- Enter or paste any text.
- Detect language using:
  - **LangDetect algorithm**
  - **Custom N-gram Logistic Regression model**
- Shows:
  - Predicted language  
  - Language confidence  
  - Probabilities for all detected languages  
  - Detection time  
  - Clean UI with sample loader  

---

###  **2. Batch Language Detection**
- Upload `.txt` or `.csv` file
- Automatically detects language **for every row**
- Supports multilingual & noisy datasets  
- Displays:
  - Characters count  
  - Word count  
  - Detected language  
  - Confidence  
  - Errors for empty texts  
- Download analyzed results as CSV

---

###  **3. Preloaded Sample Texts**
- Choose samples (English, Hindi, Spanish, French, etc.)
- Click **Load Sample** to auto-insert into the text box
- Uses **Streamlit Session State** to avoid widget modification errors


## Technology Stack
| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| Backend | Python |
| ML Model | Logistic Regression (scikit-learn) |
| Language Detection | LangDetect |
| Data Processing | Pandas, NumPy |

## Install dependices
pip install -r requirements.txt

## How to run 
streamlit run app.py
<img width="1849" height="880" alt="image" src="https://github.com/user-attachments/assets/bd03fc1d-9445-41ab-9d96-915222c21739" />



## Student Performance Predictor
The Student Performance Predictor is a machine learning–powered web application designed to predict a student’s exam score and recommend improvements based on their personal and academic factors.

This project includes:
📊 Single Student Prediction
👥 Batch Analysis
📁 Dataset Upload & Insights
🔍 Model Explainability
📂 Dataset Explorer

Users can enter various student details to generate predictions:
Attendance Rate (%)
Study Hours/Week
Previous Grade
Internet Usage (hrs/day)
Sleep Hours/Night
Extracurricular Activities
Parental Involvement
Tutoring Sessions/Month
Assignment Completion (%)

## Machine Learning Model Selection
The app allows users to choose from different ML models, such as:
Random Forest
XGBoost
Linear Regression
Decision Tree
Gradient Boosting

## Install dependices
pip install -r requirements.txt

## How to run 
streamlit run app.py

<img width="1853" height="833" alt="image" src="https://github.com/user-attachments/assets/5e8550df-e94d-46d1-b40b-f35b0740ab18" />
