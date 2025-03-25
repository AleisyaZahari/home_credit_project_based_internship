
# Home Credit Indonesia - Machine Learning Credit Scoring

<b>Aleisya Zahari Salam</b> 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](http://linkedin.com/in/aleisyazaharisalam)

**Tool**

[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)


## 🚀 Project Overview
This project was completed as part of a **Project-Based Internship** at **Home Credit Indonesia**, where we were tasked with developing an **end-to-end Machine Learning model** for credit scoring. The objective was to improve **loan approval decisions** by predicting a customer's ability to repay a loan.

By leveraging statistical methods and **Machine Learning algorithms**, we aimed to maximize the **potential of Home Credit's data**, ensuring that creditworthy customers are **not mistakenly rejected** and that loans are provided with appropriate **terms and repayment plans**.

---

## 📌 Table of Contents
1. Import Library
2. Read Dataset
3. Exploratory Data Analysis (EDA)
   - Data Quality Check
   - Descriptive Statistics
   - Univariate Analysis
   - Bivariate Analysis
   - Outlier Detection & Correlation Analysis
4. Data Preprocessing
   - Dropping Unnecessary Columns
   - Handling Missing Values
   - Feature Encoding
   - One-Hot Encoding
   - Feature Scaling (StandardScaler & MinMaxScaler)
5. Modeling
   - Data Splitting
   - Model Training (including Logistic Regression)
   - Hyperparameter Tuning (Grid Search)
6. Evaluation
   - Model Performance Metrics
   - Feature Importance Analysis
7. Business Insights & Recommendations
8. Conclusion & Next Steps

---

## 📊 Key Insights
- **Top Features Influencing Loan Repayment:**
  - **AMT_ANNUITY**: Total annuity in the range of **20k-30k** significantly impacts repayment ability.
  - **DEF_60_CNT_SOCIAL_CIRCLE & DEF_30_CNT_SOCIAL_CIRCLE**: Number of times a customer defaulted in their **social circle**.
  - **FLAG_DOCUMENT_3**: Presence of specific documents in customer applications.
  - **INCOME_UNEMPLOYED**: Correlation between employment status and repayment behavior.
- Logistic Regression was chosen as one of the models to meet the **internship requirement** and provide a strong baseline for interpretability.
- A combination of **feature engineering** and **data preprocessing** played a crucial role in improving the accuracy of predictions.

---

## 📌 Tools & Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Machine Learning Algorithms**: Logistic Regression, Hyperparameter Tuning (Grid Search)
- **Feature Engineering & Data Cleaning Techniques**
- **Jupyter Notebook** for analysis


<!-- ## 📅 **Project Date:** [Insert Date] -->  

---

## 📌 Business Impact & Recommendations
- The project helps Home Credit optimize its **loan approval process**, reducing rejection rates for **creditworthy customers**.
- Insights from **feature importance analysis** can improve **risk assessment strategies**.
- Future improvements can include testing **more complex models** (e.g., Random Forest, XGBoost) and adding **external data sources**.


