# Description of my project

This project focuses on identifying potentially fraudulent insurance claims using machine learning techniques, with particular emphasis on addressing class imbalance and improving fraud detection.

The analysis was conducted on a dataset of 15,420 insurance claims, which was cleaned to obtain 15,100 usable records. Since fraudulent claims represented only 5.91% of the dataset, accuracy alone was insufficient to evaluate model performance.

### Objectives
- Identify potentially fraudulent insurance claims.
- Compare machine learning models for fraud classification.
- Address class imbalance using Synthetic Minority Oversampling Technique (SMOTE).
- Identify key variables associated with insurance fraud.
- Segment claims into risk groups to support targeted investigations.

### Methodology
- **Data Preprocessing:** Data cleaning, duplicate removal, categorical encoding, standardisation, and exploratory data analysis (EDA).
- **Feature Selection:** Backward elimination using statistical significance and multicollinearity checks.
- **Class Imbalance Handling:** Applied SMOTE at different fraud-to-genuine ratios.
- **Model Development:** Built and compared Logistic Regression, Decision Tree, and Random Forest classifiers.
- **Model Evaluation:** Assessed performance using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and precision-recall curves.
- **Risk Segmentation:** Applied K-Means clustering to identify groups with different concentrations of fraudulent claims.

### Key Results
Logistic Regression with 50:50 SMOTE balancing achieved **87.6% fraud recall**, detecting 156 out of 178 fraudulent claims in the test set. Random Forest achieved the highest ROC-AUC of 0.789 among the evaluated models. Feature importance analysis identified vehicle age and the number of claim supplements among the leading fraud-associated variables. K-Means segmentation identified a high-risk cluster with an approximately 14% fraud rate, compared with the overall rate of 5.91%.

### Business Application
The findings demonstrate how machine learning can support risk-based claim screening, prioritise suspicious cases for investigation, and improve the allocation of insurance fraud investigation resources. Model predictions are intended to support human review rather than automatic claim rejection.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Statsmodels, Imbalanced-learn (SMOTE), Matplotlib, Seaborn, Logistic Regression, Decision Tree, Random Forest, K-Means Clustering.
