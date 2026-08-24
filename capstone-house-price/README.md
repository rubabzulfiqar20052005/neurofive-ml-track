# Pakistan House Price Prediction

## Problem Statement
Real estate pricing in Pakistan is often inconsistent and driven by informal negotiation rather than data. Buyers, sellers, and agents frequently lack a reliable way to estimate a fair market price for a property based on its actual features. This project builds a machine learning model that predicts house prices using property listings scraped from Zameen.com, helping stakeholders make more informed pricing decisions.

## Dataset
- Source: Zameen.com property listings (Kaggle)
- 168,446 raw listings across 5 major Pakistani cities (Islamabad, Lahore, Karachi, Rawalpindi, Faisalabad)
- Filtered to 120,655 "for sale" listings, cleaned down to 118,536 valid rows after removing data entry errors

## Approach
1. **Data Cleaning:** Removed zero-price/zero-area listings and capped extreme outliers (99th percentile) in price, area, bathrooms, and bedrooms
2. **EDA:** Analyzed price distribution by city and property type; found `Total_Area` correlates only weakly with price (0.32), while location and bedrooms matter more
3. **Feature Engineering:** Created `price_per_sqft`, grouped 1,444 unique locations into top-20 + "Other" to manage high cardinality
4. **Modeling:** Trained and compared Linear Regression, Random Forest, and XGBoost
5. **Deployment:** Built and deployed a Streamlit app for live price prediction

## Results
| Model              | RMSE (PKR)  | R² Score |
|---------------------|-------------|----------|
| Linear Regression    | 18,334,088  | 0.451    |
| Random Forest         | 9,864,214   | 0.841    |
| **XGBoost (Best)**    | **9,718,241** | **0.846** |

**Key Insight:** Location (especially DHA Defence) and city (especially Karachi) are the strongest price drivers — even more than property size — confirming the real estate principle "location, location, location."

## How to Run
```bash
git clone https://github.com/rubabzulfiqar20052005/neurofive-ml-track.git
cd neurofive-ml-track/capstone-house-price
pip install -r requirements.txt
streamlit run application.py
```

## Live App
🔗 [Try the live predictor here](https://neurofive-ml-track-dofkrpamv9qxaqk2bmf8ua.streamlit.app/)

## Tools Used
Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, Streamlit