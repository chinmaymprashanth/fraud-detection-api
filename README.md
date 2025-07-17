

# 💳 Credit Card Fraud Detection API

A machine learning API that detects fraudulent credit card transactions using a trained Random Forest classifier. Built with FastAPI, the API also provides **SHAP explanations** for model transparency.

---

## 📂 Dataset

**Source**: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

* **Description**: This dataset contains transactions made by credit cards in September 2013 by European cardholders.
* **Features**:

  * 28 anonymized PCA components (`V1` to `V28`)
  * `Amount`: Transaction amount
  * `Class`: Target variable (1 = Fraud, 0 = Not Fraud)

---

## 🎯 Project Overview

* Train a fraud detection model using **RandomForestClassifier**
* Use a **custom threshold (0.22)** for fraud decisioning: The fraud prediction threshold was set to 0.22 based on exploratory data analysis to optimize recall and reduce false negatives. This means any transaction with a predicted fraud probability above 0.22 is flagged as fraud.
* Deploy as a **REST API** with `FastAPI`
* Return **SHAP values** to explain each prediction
* Dockerized for easy deployment

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/your-username/creditcard-fraud-api.git
cd creditcard-fraud-api
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### ▶️ Run locally

```bash
uvicorn api:app --reload
```

Access documentation:

* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 🐳 Run with Docker

Build the Docker image:

```bash
docker build -t fraud-api .
```

Run the container:

```bash
docker run -p 8000:8000 fraud-api
```

---

## 🧠 API Details

### `POST /predict`

Make a fraud prediction with SHAP explanations.

#### 🔸 Request Format

```json
{
  "features": [0.5, -0.3, 0.1, ..., 0.5]  // Total 28 numerical features
}
```

#### 🔹 Response Format

```json
{
  "fraud_probability": 0.89,
  "is_fraud": 1,
  "shap_values": [0.0039, -0.0016, 0.0286, ..., 0.0023]
}
```

* `fraud_probability`: Probability (0 to 1) of fraud.
* `is_fraud`: 1 = Fraud, 0 = Not Fraud.
* `shap_values`: How much each feature contributed to the prediction.

---

## 🔍 SHAP Explanation

This API uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

* SHAP values quantify how each input feature pushed the model towards classifying a transaction as fraud or not.
* Each SHAP value corresponds to one input feature.

---

## 📁 File Structure

```
├── api.py                  # FastAPI app
├── rf_model.pkl            # Trained Random Forest model
├── requirements.txt        # Python dependencies
├── Dockerfile              # For containerization
├── README.md               # Project documentation
```

---

## 📈 Future Enhancements

* Include feature names in SHAP output
* Add visualization for SHAP explanations
* Support batch predictions
* Integrate frontend UI or Streamlit dashboard

---

## 🙋‍♂️ Author

Chinmay Prashanth
*Data Analyst & ML Enthusiast*

---

## 📄 License

This project is licensed under the MIT License.


