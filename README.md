# CineAI Streamlit App

## Overview
CineAI is a Streamlit application designed for film recommendation and analysis. This README provides comprehensive documentation on how to run the `working_app.py` Streamlit app and its supporting files.

## Setup Instructions
To set up the project and run the Streamlit app, follow the instructions below:

### 1. Prerequisites
Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package manager)

### 2. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/Rajwill321/cineai.git
cd cineai
```

### 3. Install Required Packages
Use pip to install the required dependencies:
```bash
pip install -r requirements.txt
```

## Required Data and Model Files
To run the application, you will need the following files in your project directory:
1. `movies.csv` — Contains movie metadata.
2. `features_movie.csv` — Feature set for movies.
3. `features_user.csv` — Feature set for user preferences.
4. `xgboost_model.pkl` — Trained XGBoost model for recommendations.
5. `llm_client.py` — Script for managing language model interactions.
6. `features_movie_svd.csv` — SVD-generated features for movies.
7. `features_user_svd.csv` — SVD-generated features for user preferences.

Ensure you have these files in the root directory of the project to avoid errors while running the app.

## How to Run the App
To run the Streamlit app, use the following command in the terminal:
```bash
streamlit run working_app.py
```
Once the app is running, it will automatically open in your default web browser.

## Troubleshooting
- **Module Not Found Error**: Ensure all required packages are installed as specified in `requirements.txt`.
- **File Not Found Error**: Ensure all necessary data files are present in the project root directory.
- **Streamlit Not Starting**: Verify that Streamlit is installed correctly. You can check by running `streamlit --version`.

## Supporting Utilities
The `llm_client.py` file provides utilities for interacting with a language model for enhanced recommendations. It is crucial for the app's functionality and should remain in the project.

### Notes
- Always keep your data files up-to-date for the best performance of the Streamlit app.
- Consider version control for your model files for consistency and reproducibility.

Feel free to contribute to this project or reach out if you have any questions or suggestions!