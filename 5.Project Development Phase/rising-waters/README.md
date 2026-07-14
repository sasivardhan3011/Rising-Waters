# Rising Waters

A Flask web application for flood-risk prediction using a trained machine learning model.

## Project Overview

This project trains a classification model on synthetic flood-related data and exposes it through a simple web interface. Users can enter rainfall and visibility values to receive a flood-risk prediction, probability, and advisory message.

## Features

- Train a flood prediction model from generated data
- Serve the model through a Flask web app
- Display prediction results and risk guidance in the browser

## Project Structure

- app.py: Flask application entry point
- train_model.py: Model training script
- generate_data.py: Synthetic flood dataset generator
- templates/: HTML templates for the web UI
- static/: CSS styling
- data/: Generated dataset location
- models/: Trained model artifacts

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the model

Run:

```bash
python train_model.py
```

This will generate the dataset if needed and create the model artifacts in the models folder.

## Run the app

Start the Flask server:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- The app expects the trained model files to exist in the models folder.
- If the model is missing, run train_model.py first.
