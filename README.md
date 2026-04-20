
# Flight Price Prediction (TripSmooth AI)

This project builds a machine learning system to predict flight ticket prices using real-world aviation data.

It demonstrates how AI can be used to power intelligent pricing systems for travel platforms like **TripSmooth**.

---

## Project Highlights

- Built on real flight pricing dataset (300k+ records)
- Advanced feature engineering (time, duration, routes)
- Reduced feature space from **1000+ → ~60 features**
- Compared multiple models:
  - Random Forest
  - XGBoost
  - Tuned XGBoost
- Implemented **segmented modeling (Economy vs Business)**
- Extracted real-world pricing insights

---

## Model Performance

| Model | MAE | R² |
|------|-----|----|
| **Random Forest** | **1917** | **0.979** |
| Tuned XGBoost | 2179 | 0.970 |
| XGBoost | 2804 | 0.964 |

Random Forest performed best on this dataset.

---

## Key Insights

### 1. Travel class dominates pricing
Initial model showed **~90% importance from class alone**, indicating strong price separation between economy and business.

### 2. Segmented modeling improved insight
By training separate models:

#### Economy pricing
- Driven mainly by **duration (~50%)**
- Time of travel and routes matter
- Airline impact is smaller

#### Business pricing
- More influenced by **time (day/week/month)**
- Airline plays a larger role
- More variability and complexity

---

## Feature Importance Example

![Feature Importance](results/figures/xgb_tuned_feature_importance.png)

---

## Model Comparison

![Model Comparison](results/figures/model_comparison_mae.png)

---

## Actual vs Predicted

![Actual vs Predicted](results/figures/actual_vs_predicted.png)

---

## Application to TripSmooth

This project demonstrates how machine learning can enable:

- Real-time flight price prediction
- Smarter search ranking
- Personalized travel recommendations
- AI-powered booking assistants
- Fare intelligence dashboards

---

## Project Structure

notebooks/
01_eda_model.ipynb

results/
figures/

data/
raw/ (not included)
sample/


---

## Dataset

Dataset used:
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Raw data is not included due to size limits.

This project uses the Kaggle dataset:

`shubhambathwal/flight-price-prediction`

The raw CSV files are not stored in this repository. Download them with:

```bash
pip install -r requirements.txt
python scripts/download_data.py
---


## How to Run

```bash
pip install -r requirements.txt
jupyter notebook

Open:
notebooks/01_eda_model.ipynb

## Future Improvements
Deploy as API (FastAPI)
Build Streamlit UI
Add real-time pricing engine
Integrate with flight search APIs
Add demand prediction layer

## Author
Abiola Olaleye



