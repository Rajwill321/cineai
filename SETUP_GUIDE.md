# Setup Guide for CineAI Streamlit App

## Environment Configuration
1. Ensure you have Python 3.7 or higher installed on your machine.
2. It is recommended to create a virtual environment for the project. You can do this by running:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the necessary packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## API Key Setup
1. Sign up for an API key from the appropriate service that the app uses (e.g., OpenAI, etc.).
2. Store your API key in a secure location.
3. Create a `.env` file in the root of your project with the following content:
   ```plaintext
   API_KEY=your_api_key_here
   ```
   Make sure to replace `your_api_key_here` with your actual API key.

## Data File Placement
1. Download the required dataset files and place them in a folder named `data` in the root of your repository.
   - Ensure the structure looks like this:
     ```plaintext
     /your-repo
     ├── data/
     │   ├── dataset1.csv
     │   └── dataset2.csv
     └── SETUP_GUIDE.md
     ```

## Step-by-Step Running Instructions
1. Open a terminal in the project directory.
2. If you have not activated the virtual environment, do so:
   ```bash
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Run the Streamlit app with the following command:
   ```bash
   streamlit run app.py
   ```
4. Your default web browser should open and display the app. If it doesn't, you can manually go to `http://localhost:8501` in your web browser.

5. Follow any additional on-screen instructions in the app.