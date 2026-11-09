# American Airlines – AI-Driven Brand Reputation Intelligence

IIM Ranchi · *AI-Driven Business Analytics: Text Intelligence and Operational Insights* · Prof. Sasadhar Bera · Group Assignment 06

An interactive Streamlit dashboard that analyses airline tweets to measure sentiment, surface recurring complaints, build word networks and topic models, train/evaluate sentiment classifiers (including a genuinely trained Keras LSTM), and translate findings into Pareto priorities and a Six Sigma DMAIC plan.

## Repository layout (flat – no subfolders required)

```text
American-Airlines-Brand-Analytics/
├── app.py
├── requirements.txt
├── Airline_Tweets.csv
└── README.md
```
(`.streamlit/config.toml` is an optional theme file.)

## Dataset requirements
`Airline_Tweets.csv` must sit next to `app.py`. Required columns: `tweet_id`, `airline_sentiment`, `airline`, `text`. The app also uses (when present) `negativereason`, `retweet_count`, `tweet_date`/`tweet_created`, `record_origin`, `record_is_observed` and the `scenario_*` columns. The file is loaded automatically; no upload is needed.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Use Python 3.11 or 3.12 (TensorFlow wheels).

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload the files above to the repository **root**.
2. Go to share.streamlit.io → *New app* → select the repository and branch.
3. Set **Main file path** to `app.py`; under *Advanced settings* choose Python 3.11 or 3.12.
4. Click **Deploy**. First start installs TensorFlow and can take several minutes.

## Features (10 tabs)
Executive Overview · Data & Quality · Sentiment · Text & Word Networks · Complaints & Risk (Pareto, priority table) · Topic Modeling (NMF/LDA) · AI Model Lab · Reputation & Scenarios · Insights & Six Sigma DMAIC · Reports & Exports.

## Model training
Models never train automatically. Set the training configuration in the sidebar and press **Train / evaluate models**. Models: majority-class baseline, VADER, TF-IDF + Logistic Regression, Embedding + LSTM. Training uses **original observed tweets only**, removes duplicate texts, fits all text transformations on training data only, and evaluates on a fixed, seeded American Airlines hold-out. Descriptive filters do not influence training or the test set. If TensorFlow cannot start, the app shows an explicit error and all other tabs keep working – no substitute results are produced.

## Computational limitations
CPU only. The LSTM is small (Embedding 64 → LSTM 48 → Dense 32) with early stopping; defaults (8 epochs, batch 64, length 40) are chosen for Community Cloud. Topic modelling is capped at a configurable number of tweets.

## Data provenance and interpretation
- Observed tweets are authentic; generated records and `scenario_*` business metrics are hypothetical demonstrations and are excluded by default (Demonstration mode includes generated rows in descriptive views only).
- Scenario metrics are stored per airline-month and repeated on tweet rows; the app aggregates at airline × month.
- Observed tweets cover a short window, so long-term forecasting is disabled unless data meet a minimum (≥30 days observed; ≥12 months for scenario series).
- Twitter is a biased sample; results do not represent all passengers. Severity scores in the priority table are management assumptions.
