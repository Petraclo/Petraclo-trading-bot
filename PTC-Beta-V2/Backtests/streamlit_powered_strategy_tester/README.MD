## About
The `streamlit_powered_strategy_tester` is a full-stack backtester that simulates the execution of the PeTraClo trading strategy on past data of users choice with a given set of strategy parameters

## How to run
1. ```
    cd strategy_tester
    ```

2. **Install dependencies**  
   Make sure you have Python 3.10+ installed.


   **Optional** 
   + Create and activate a virtual environment
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   Install required packages using pip:
   ```
   pip install -r requirements.txt
   ```
   *(Or try manually)*

3. **Configure trade account and api key**
   - In `config.py` enter your mt5 account details
   - In `.streamlit/secrets.example.toml` replace boilerplate with your groq api key from https://console.groq.com/keys .

4. **Start the Streamlit app**  
   In your terminal, run:
   ```
   streamlit run app.py
   ```

4. **Access the app**  
   Open your browser and go to the URL shown in the terminal (usually http://localhost:8501).

6. **Usage**  
   - Use the sidebar to set backtest parameters and run simulations.
   - Use the "AI Consultant" tab to ask questions about past runs.

---

**Note:**  
- MT5 must be installed and running if you want to fetch live data.
- Database and index files will be created automatically in this
