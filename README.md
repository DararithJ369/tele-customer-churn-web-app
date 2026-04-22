
# Telco Customer Churn Prediction

A Streamlit dashboard that predicts customer churn using a trained logistic regression model and visualizes key dataset insights.

## Project Structure
- `app.py`: Streamlit web app (prediction + EDA dashboard)
- `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset
- `models/artifacts/logistic_churn_model.joblib`: saved model used by the app
- `preprocessing.ipynb`: data exploration, preprocessing, and model training

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App
Make sure the dataset and saved model exist:

- Dataset: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Model: `models/artifacts/logistic_churn_model.joblib`

Then run:

```bash
streamlit run app.py
```

## Notes
- The app loads the saved model and **does not retrain** at runtime.
- If you retrain the model in `preprocessing.ipynb`, copy or save it to `models/artifacts/logistic_churn_model.joblib`.

