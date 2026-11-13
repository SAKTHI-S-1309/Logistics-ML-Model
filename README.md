\# 🚚 Logistics ML Model with API



An end-to-end Machine Learning application that predicts whether a logistics shipment is likely to be delivered late. The project combines predictive analytics, explainable AI, a REST API, an interactive dashboard and Docker containerization.



\## 📌 Project Overview



Late deliveries can affect customer satisfaction, operational efficiency and supply-chain performance.



This project develops a Machine Learning model to identify shipments at risk of late delivery using shipment, route, vehicle, traffic and weather information.



The system provides:

\- Late delivery probability predictions.

\- Business-cost-based decision threshold optimization.

\- Explainable AI using SHAP.

\- REST API using FastAPI.

\- Interactive dashboard using Streamlit.

\- Data drift monitoring.

\- Docker-based deployment.



\*\*Dataset note:\*\* This project uses synthetically generated logistics data representing a fictional cold-chain distribution company. It does not contain real company shipment records.



\## 🎯 Objectives



\- Generate and preprocess logistics shipment data.

\- Handle missing values, duplicate records and inconsistent hub names.

\- Normalize timestamps from different time zones.

\- Perform exploratory data analysis.

\- Engineer relevant shipment features.

\- Train and compare multiple Machine Learning models.

\- Optimize the prediction threshold based on business costs.

\- Explain model predictions using SHAP.

\- Deploy predictions through FastAPI and Streamlit.

\- Monitor changes in input data distributions.



\## 🛠️ Technology Stack



| Category | Technologies |

|---|---|

| Programming | Python |

| Data Processing | Pandas, NumPy |

| Visualization | Matplotlib, Seaborn |

| Machine Learning | Scikit-learn, XGBoost |

| Hyperparameter Tuning | RandomizedSearchCV |

| Explainable AI | SHAP |

| API Development | FastAPI, Pydantic |

| Frontend | Streamlit |

| Testing | Pytest |

| Deployment | Docker, Docker Compose |

| Model Storage | Joblib |



\## 📂 Project Structure



```text

Logistics-ML-Model/

│

├── app/

│   ├── main.py

│   └── streamlit\_app.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── models/

│   ├── logistic\_regression.joblib

│   ├── decision\_tree.joblib

│   ├── random\_forest.joblib

│   └── best\_model.joblib

│

├── notebooks/

│

├── reports/

│   ├── figures/

│   ├── evaluation\_results.json

│   ├── model\_comparison.csv

│   ├── optimal\_threshold.json

│   └── drift\_report.json

│

├── src/

│   ├── generate\_data.py

│   ├── clean\_data.py

│   ├── eda.py

│   ├── prepare\_data.py

│   ├── train\_models.py

│   ├── evaluate\_model.py

│   ├── optimize\_threshold.py

│   ├── explain\_model.py

│   └── monitor\_drift.py

│

├── tests/

│   └── test\_api.py

│

├── Dockerfile

├── docker-compose.yml

├── .dockerignore

├── requirements.txt

└── README.md

```



\## 📊 Dataset



The synthetic dataset contains:



\- 50,000 unique shipments.

\- 12 logistics hubs.

\- Multiple vehicle types.

\- Traffic and weather conditions.

\- Shipment temperature and humidity information.

\- Dispatch and delivery timestamps.

\- Warehouse scan events.



The generated raw shipment file initially contains 50,100 records, including 100 intentional duplicate records. These are removed during data cleaning.



\### Data preprocessing



The preprocessing workflow includes:



1\. Duplicate shipment removal.

2\. Hub name standardization.

3\. Timestamp conversion to UTC.

4\. Chronological sorting of warehouse scan events.

5\. Missing-value handling through ML pipelines.

6\. Dispatch-time feature engineering.

7\. Chronological train-test splitting.



The final dataset contains 50,000 unique shipments.



\## 🤖 Machine Learning Models



Four classification algorithms were trained and compared:



1\. Logistic Regression

2\. Decision Tree

3\. Random Forest

4\. XGBoost



The training pipeline uses:

\- Numerical imputation.

\- Categorical imputation.

\- One-hot encoding.

\- Feature scaling.

\- Class imbalance handling.

\- RandomizedSearchCV for hyperparameter tuning.

\- TimeSeriesSplit for chronological cross-validation.



\*\*Best-performing model:\*\* Logistic Regression.



It achieved the highest cross-validation average precision score among the four evaluated models.



\## 📈 Model Evaluation



The final model was evaluated on a separate chronological test dataset containing 10,000 shipments.



| Metric | Result |

|---|---:|

| Accuracy | 72.03% |

| Precision | 48.21% |

| Recall | 72.61% |

| F1 Score | 57.95% |

| ROC-AUC | 79.87% |

| PR-AUC | 62.06% |



The dummy baseline achieved a PR-AUC of 26.54%.



The trained model achieved approximately \*\*133.82% relative improvement in PR-AUC\*\* over the dummy baseline.



\## 💰 Business-Cost-Based Threshold Optimization



The default classification threshold of 0.50 was compared with an optimized threshold of 0.33.



Illustrative business costs:

\- False Negative: ₹1,000

\- False Positive: ₹200



A false negative represents a late shipment that the model fails to flag. A false positive represents an on-time shipment incorrectly flagged as late.



| Metric | Default 0.50 | Optimized 0.33 |

|---|---:|---:|

| Accuracy | 72.03% | 64.12% |

| Precision | 48.21% | 41.47% |

| Recall | 72.61% | 85.49% |

| F1 Score | 57.95% | 55.85% |

| Estimated Test Cost | ₹11,41,000 | ₹10,25,600 |



The optimized threshold reduces estimated business cost by approximately 10.11% on the test dataset.



It also identifies 342 additional late shipments compared with the default threshold.



The lower threshold prioritizes identifying more potentially late shipments, accepting more false alarms in return.



\*\*Note:\*\* These cost values are illustrative assumptions, not actual company financial data.



\## 🔍 Explainable AI with SHAP



SHAP (SHapley Additive exPlanations) is used to understand the factors influencing model predictions.



The analysis includes:

\- Global feature importance.

\- SHAP summary visualization.

\- Individual shipment explanations.

\- SHAP waterfall visualization.



Important model features include traffic conditions, weather conditions, vehicle type, package temperature and dispatch time.



\## 🌐 FastAPI



The project exposes a REST API using FastAPI.



\### Available endpoints



| Method | Endpoint | Description |

|---|---|---|

| GET | `/` | API information |

| GET | `/health` | Health status |

| POST | `/predict` | Predict late delivery risk |



Interactive API documentation is available through Swagger UI.



\## 🖥️ Streamlit Dashboard



The Streamlit dashboard allows users to enter shipment information and receive:



\- Late delivery probability.

\- Decision threshold.

\- Predicted delivery status.

\- Risk classification.

\- Business recommendation.



The dashboard communicates with the FastAPI backend.



\## 📉 Data Drift Monitoring



A Population Stability Index (PSI)-based monitoring system compares training and test feature distributions.



Drift categories:

\- PSI below 0.10: No significant drift.

\- PSI from 0.10 to 0.20: Moderate drift.

\- PSI of 0.20 or above: Significant drift.



The chronological test period produces a significant shift in the dispatch-month distribution. This is expected because the training and testing periods cover different months.



The monitoring process generates a drift report and log file.



\## 🐳 Docker Deployment



Docker Compose runs both application services together.



\### Prerequisites



\- Docker Desktop

\- WSL 2 on Windows

\- Git (optional)



\### Run the application



Clone the repository and navigate into the project directory.



Build the Docker images:



```bash

docker compose build

```



Start the services:



```bash

docker compose up -d

```



Check running containers:



```bash

docker compose ps

```



\### Application URLs



\*\*Streamlit Dashboard\*\*



http://localhost:8501



\*\*FastAPI Swagger Documentation\*\*



http://localhost:8000/docs



Stop the application:



```bash

docker compose down

```



\## 🧪 Testing



The API is tested using Pytest and FastAPI TestClient.



Test cases include:

\- Root endpoint.

\- Health endpoint.

\- Valid shipment prediction.

\- Invalid humidity validation.

\- Invalid dispatch timestamp validation.



\*\*Latest test result: 5 passed, 0 failed.\*\*



\## ⚠️ Limitations



\- The dataset is synthetic and may not represent real-world logistics complexity.

\- The business costs are illustrative.

\- Model predictions are probabilistic, not guarantees of delivery performance.

\- Drift detection identifies changes in input distributions; it does not independently establish model performance deterioration.

\- Real-world deployment would require validation using actual operational data.



\## 🚀 Future Improvements



\- Scale the synthetic dataset to approximately 1.5 million records.

\- Add real-time shipment tracking integration.

\- Introduce automated model retraining.

\- Improve drift monitoring with seasonal baselines.

\- Evaluate production API latency.

\- Deploy the application to a cloud platform.

\- Add continuous integration and automated testing.



\## 👩‍💻 Author



\*\*SAKTHI S\*\*



B.Tech – Artificial Intelligence and Data Science



\## 📄 License



This project is intended for educational and portfolio purposes.

