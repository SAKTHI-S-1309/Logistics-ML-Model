\# Model Card: Logistics Late Delivery Prediction



\## 1. Model Overview



\*\*Project Name:\*\* Logistics ML Model with API  

\*\*Model Name:\*\* Logistic Regression  

\*\*Model Type:\*\* Binary Classification  

\*\*Framework:\*\* Scikit-learn  

\*\*Version:\*\* 1.0  

\*\*Author:\*\* SAKTHI S  

\*\*Project Domain:\*\* Logistics and Supply Chain Management



\### Purpose



This machine learning model predicts the probability of a shipment arriving later than its planned transit time.



It is designed to help logistics teams identify potentially delayed shipments and take proactive measures.



\*\*Important:\*\* This is an educational project using synthetic data. It is not a production-validated logistics prediction system.



\## 2. Intended Use



\### Primary Use Cases

\- Predict potential late deliveries.

\- Identify shipments requiring additional monitoring.

\- Support proactive logistics planning.

\- Demonstrate an end-to-end machine learning workflow.



\### Out-of-Scope Uses

\- Autonomous shipment cancellation or rerouting.

\- Making financial or contractual decisions without human review.

\- Real-world deployment without validation using actual operational data.



\## 3. Dataset Description



The project uses a synthetically generated logistics dataset.



| Attribute | Description |

|---|---|

| Unique shipments | 50,000 |

| Original shipment rows | 50,100 |

| Duplicate records | 100 |

| Warehouse scan events | 200,000 |

| Distribution hubs | 12 |

| Data period | January–December 2025 |

| Target variable | late\_delivery |



\### Data Characteristics



The synthetic dataset includes:

\- Missing values.

\- Duplicate shipment records.

\- Different warehouse naming conventions.

\- Multiple timestamp time zones.

\- Out-of-order warehouse scan events.

\- Different vehicle types, weather conditions and traffic levels.



No real customer or company shipment records were used.



\## 4. Target Definition



The target variable is `late\_delivery`.



\- \*\*0:\*\* Shipment delivered on or before the planned delivery time.

\- \*\*1:\*\* Shipment delivered after the planned delivery time.



The actual delivery timestamp is used only to generate the target label. It is excluded from model input features to prevent target leakage.



\## 5. Input Features



The model uses information available at dispatch time.



\### Numerical Features

\- Route distance.

\- Ambient temperature.

\- Humidity percentage.

\- Package temperature.

\- Planned transit hours.

\- Dispatch year.

\- Dispatch month.

\- Dispatch day.

\- Dispatch hour.

\- Day of the week.



\### Categorical Features

\- Origin hub.

\- Destination hub.

\- Vehicle type.

\- Traffic level.

\- Weather condition.



Post-dispatch information is not used as a prediction input.



\## 6. Data Preparation



The following preprocessing steps were performed:



1\. Duplicate shipment removal.

2\. Hub-name standardization.

3\. Timestamp conversion to UTC.

4\. Missing-value handling.

5\. Dispatch-time feature engineering.

6\. Numerical feature scaling.

7\. Categorical feature encoding.



\### Preprocessing Pipeline



\*\*Numerical features:\*\*

\- Median imputation.

\- StandardScaler.



\*\*Categorical features:\*\*

\- Most-frequent imputation.

\- OneHotEncoder with unknown-category handling.



\## 7. Training Methodology



Four machine learning algorithms were evaluated:



1\. Logistic Regression.

2\. Decision Tree.

3\. Random Forest.

4\. XGBoost.



\### Model Selection



\- Hyperparameter tuning: RandomizedSearchCV.

\- Cross-validation: TimeSeriesSplit.

\- Class imbalance: Class weights or positive-class weighting.

\- Model selection metric: Average Precision (PR-AUC).



\*\*Selected Model:\*\* Logistic Regression.



\### Data Split



A chronological 80:20 split was used.



| Dataset | Records |

|---|---:|

| Training | 40,000 |

| Testing | 10,000 |



This approach helps simulate predictions on future shipments using historical data.



\## 8. Model Performance



\### Test Results at Default Threshold (0.50)



| Metric | Result |

|---|---:|

| Accuracy | 72.03% |

| Precision | 48.21% |

| Recall | 72.61% |

| F1-Score | 57.95% |

| ROC-AUC | 79.87% |

| PR-AUC | 62.06% |



\### Baseline Comparison



The dummy classifier achieved a PR-AUC of 26.54%.



The selected model achieved a relative PR-AUC improvement of approximately 133.82% over this baseline.



\## 9. Business-Based Threshold Optimization



The default classification threshold of 0.50 was compared with an optimized threshold of 0.33.



\### Illustrative Business Cost Assumptions



| Error Type | Assumed Cost |

|---|---:|

| False Negative | ₹1,000 |

| False Positive | ₹200 |



A false negative represents a late shipment incorrectly classified as on-time.



A false positive represents an on-time shipment incorrectly flagged as potentially late.



The threshold was selected using out-of-fold training predictions, without using the test set for threshold selection.



\### Test Set Comparison



| Metric | Threshold 0.50 | Threshold 0.33 |

|---|---:|---:|

| Accuracy | 72.03% | 64.12% |

| Precision | 48.21% | 41.47% |

| Recall | 72.61% | 85.49% |

| F1-Score | 57.95% | 55.85% |

| Business Cost | ₹11,41,000 | ₹10,25,600 |



\### Business Impact



\- Estimated test-set cost reduction: ₹1,15,400.

\- Relative cost reduction: approximately 10.11%.

\- Additional late shipments detected: 342.

\- False negatives reduced from 727 to 385.



\*\*Selected Decision Threshold: 0.33\*\*



This threshold prioritizes detecting more potentially late shipments, accepting a higher number of false positives.



All financial costs are illustrative assumptions, not actual company expenses or verified savings.



\## 10. Explainability



SHAP (SHapley Additive exPlanations) was used to understand model predictions.



The analysis identified important predictive features, including:



1\. Clear weather category.

2\. High traffic category.

3\. Low traffic category.

4\. Storm weather category.

5\. Refrigerated Van vehicle category.

6\. Package temperature.



SHAP explanations help understand how features contribute to model predictions.



These associations do not establish causation.



\## 11. API and Deployment



The trained model is integrated into an application using:



\- FastAPI for prediction services.

\- Pydantic for input validation.

\- Streamlit for the interactive dashboard.

\- Docker and Docker Compose for containerization.



\### Available API Endpoints



| Endpoint | Method | Purpose |

|---|---|---|

| `/` | GET | API information |

| `/health` | GET | Health check |

| `/predict` | POST | Shipment prediction |



Swagger documentation is available at `/docs` when the API is running.



\## 12. Monitoring



A Population Stability Index (PSI)-based drift monitoring system was implemented.



The system compares training and test feature distributions.



The analysis identified significant distribution drift in `dispatch\_month`, which is expected to be influenced by the chronological split.



This result indicates a distribution difference, not proof of model performance deterioration.



Production monitoring would require ongoing evaluation against actual shipment outcomes.



\## 13. Limitations



\- The dataset is synthetic rather than collected from real logistics operations.

\- The simulated delivery-delay mechanism may not represent real-world complexity.

\- The dataset contains 50,000 unique shipments, not the full 1.5 million records described in the original project scenario.

\- Business cost assumptions are illustrative.

\- Seasonal and operational changes may affect prediction quality.

\- Model performance must be validated using real-world data before operational use.

\- Predictions represent probabilities, not guarantees.



\## 14. Responsible Use



This model should be used as a decision-support demonstration.



Human review is recommended before taking operational action.



Although protected personal attributes are not included, operational variables such as hubs, traffic and vehicle types may reflect differences in logistics conditions.



Future evaluation should examine prediction errors across hubs, vehicle types and other operational groups.



\## 15. Future Improvements



\- Train and validate using real-world logistics datasets.

\- Scale the dataset towards the original 1.5 million-record requirement.

\- Incorporate real-time shipment tracking information.

\- Improve seasonal drift monitoring.

\- Evaluate additional boosting algorithms.

\- Optimize API latency.

\- Conduct continuous model-performance monitoring.



\## 16. Conclusion



This project demonstrates an end-to-end machine learning workflow for predicting late shipment deliveries.



It covers synthetic data generation, data cleaning, exploratory analysis, feature engineering, model comparison, hyperparameter tuning, threshold optimization, SHAP explainability, FastAPI development, Streamlit visualization, automated testing, drift monitoring and Docker deployment.



The project demonstrates how machine learning can support proactive logistics decision-making while highlighting the importance of transparency, validation and responsible deployment.



\---



\*\*Author:\*\* SAKTHI S  

\*\*Project:\*\* Logistics ML Model with API  

\*\*Documentation Version:\*\* 1.0  

\*\*Data Disclosure:\*\* Synthetic data generated for educational and demonstration purposes.

