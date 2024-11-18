# power Output prediction
Power Output prediction using sklearn, FastAPI and streamlitApp
## Table of contents
- [Description](#description)
- [Requirement](#requirements)
- [Getting Started](#getting started)
  -[1. Train and save the model](# 1-train-and-save-the-model)
  - [2. Deploy FastAPI](#2-deploy-fastapi)
  - [3. RUN Streamlit](#3-streamlit)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Examples Input and Output](#example-input-and-output)
- [File Structure](#file-structure)
- [License](#license)

## Description
This project provides an API and a Streamlit application for predicting power output 
(PE) based on environmental factors. 
The model uses Linear Regression from Scikit-Learn, 
trained on features including:

- Ambient Temperature (AT)
- Exhaust Vacuum (V)
- Ambient Pressure (AP)
- Relative Humidity (RH)


# The API is deployed using FastAPI, and a Streamlit app provides an interactive interface for users to input values and get predictions.

## Requirements
To set up and run this project, you’ll need the following Python packages:

- 'fastapi'
- 'uvicorn'
- 'scikit-learn'
- 'pandas'
- 'joblib'
- 'numpy'
- 'streamlit'

You can install these dependencies by running:

'''bash
pip install -r requirements.txt
'''

## Getting Started
Follow these steps to set up and run the project.

1. Train and Save Model

    Train a Linear Regression modeling using scikit-learn, and save the trained model to a file for deployment:
  '''bash
  python linear_regression_model.py
  '''