# Names
Haneen gado 
Noura Ahmed Mahmoud 

# Email Spam Detector

Simple Streamlit application for classifying email text as **Spam** or **Ham** using a trained LSTM model.

## What this app does

- Loads a trained Keras model from `spam_detector.keras`
- Loads a tokenizer from `tokenizer.pkl`
- Cleans and preprocesses text using SpaCy
- Predicts a spam probability for email content
- Displays the result in a Streamlit web interface

## Files used by `app.py`

- `app.py` - main Streamlit application and prediction logic
- `spam_detector.keras` - trained model weights
- `tokenizer.pkl` - serialized tokenizer used at training time
- `requirements.txt` - dependency list

## Requirements

- Python 3.8+
- `streamlit`
- `tensorflow`
- `spacy`
- `en_core_web_sm` language model
- `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
pip install streamlit
python -m spacy download en_core_web_sm
```

3. Place `spam_detector.keras` and `tokenizer.pkl` in the project root.

## Run the app

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Notes

- `app.py` caches the loaded model and SpaCy pipeline for faster repeated use.
- If the model or tokenizer files are missing, the app will stop and show an error message.
- Enter the full email body text before clicking the analyse button.
