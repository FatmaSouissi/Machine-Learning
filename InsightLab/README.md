# 🌿 PharmAI Intelligence Dashboard

A professional **Angular-architecture ML Dashboard** with Python Flask backend serving **6 ML models** from the parapharmacie data warehouse.

---

## 📁 Project Structure

```
ml-dashboard/
├── backend/
│   ├── app.py                  # Flask REST API — 6 model endpoints + AI chatbot
│   └── requirements.txt        # Python dependencies
├── frontend/
│   └── index.html              # Complete Angular-style dashboard (single-file SPA)
└── README.md
```

---

## 🤖 ML Models Integrated

| Goal | Objective | Model A | Model B |
|------|-----------|---------|---------|
| **Goal 1** | Sales Volume Forecasting | SARIMA | XGBoost Time Series |
| **Goal 2** | Customer Segmentation | K-Means | DBSCAN |
| **Goal 3** | Creditworthiness | Logistic Regression | Random Forest |
| **Goal 4** | Margin Prediction | Ridge Regression | XGBoost Regressor |
| **Advanced A** | Product Recommendations | Item-Based CF | SVD Factorization |
| **Advanced B** | Anomaly Detection | Isolation Forest | Autoencoder |

---

## 🚀 Running the Application

### 1. Backend (Python Flask)

```bash
cd backend
pip install -r requirements.txt

# Run development server
python app.py
# → API available at http://localhost:5000
```

### 2. Frontend (Angular-style SPA)

Simply open `frontend/index.html` in a browser, or serve with any HTTP server:

```bash
# Using Python
cd frontend
python -m http.server 4200
# → Dashboard at http://localhost:4200

# Using Node
npx serve frontend -p 4200
```

**Or for a full Angular project**, scaffold with Angular CLI:
```bash
npm install -g @angular/cli
ng new pharmai-dashboard --routing --style=scss
# Then integrate the component code from index.html
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard/kpis` | GET | Dashboard KPI summary |
| `/api/goal1/timeseries` | GET | Sales forecasting data (SARIMA + XGBoost) |
| `/api/goal2/segmentation` | GET | Customer clustering (K-Means + DBSCAN) |
| `/api/goal3/credit` | GET | Credit risk classification (LR + RF) |
| `/api/goal4/margin` | GET | Margin prediction (Ridge + XGBoost) |
| `/api/advanced/recommendations` | GET | Product recommendations (CF + SVD) |
| `/api/advanced/anomalies` | GET | Anomaly detection (IF + Autoencoder) |
| `/api/chatbot` | POST | AI assistant chatbot |

---

## 🎨 Dashboard Features

- **Green-themed** professional dark dashboard
- **6 dedicated pages** — one per ML model group
- **Real-time KPI cards** with trend indicators
- **Interactive charts** (Chart.js) — line, bar, scatter, doughnut
- **AI chatbot** (bottom-right) with ML-aware responses
- **Model comparison tables** with metrics
- **Feature importance bars** for tree-based models
- **Confusion matrix** rendering for classifiers
- **ROC curves** for credit risk model
- **Anomaly timeline** with consensus overlay
- **Recommendation lists** — Item-Based vs SVD side-by-side

---

## 🔧 Connecting to Real Database

In `backend/app.py`, replace mock data generators with your `db_connection.py`:

```python
from db_connection import get_connection, run_query, QUERY_GOAL1_TIMESERIES

conn = get_connection()
df_raw = run_query(QUERY_GOAL1_TIMESERIES, conn)
```

Each route already has the same variable names as in your Jupyter notebook.

---

## 📦 Production Deployment

```bash
# Backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend (Angular build)
ng build --prod
# Deploy dist/ to Nginx / Apache / Azure Static
```

---

*Built for dwh_parapharmacie · Sep 2024 – Apr 2025 · 30,573 rows · Fact_Revenus*
