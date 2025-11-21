from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import plotly.express as px
from typing import List, Dict, Any

# --- FastAPI App Initialization ---
app = FastAPI(title="Master Dashboard Generator API", version="1.0")

# --- 2. Input Data Structure Definition ---
# This defines the data the UI will send (the core request).
class DashboardRequest(BaseModel):
    disease_topic: str = Field(..., example="Heart Disease")
    record_count: int = Field(..., example=20000)
    intent: str = Field(..., example="risk_factor_analysis")

# --- 3. Mock Agent Output Functions (Simulation) ---

def generate_synthetic_data(disease: str, count: int) -> pd.DataFrame:
    """Simulates the Synthetic Generator Agent output."""
    np.random.seed(42) # For reproducible data

    # Create synthetic data based on the requested count
    data = {
        'Age': np.random.randint(30, 75, count),
        'Cholesterol': np.random.randint(150, 350, count),
        'Blood_Pressure': np.random.randint(110, 180, count),
        'Smoking': np.random.choice([0, 1], size=count, p=[0.7, 0.3]),
    }
    # Calculate a synthetic Risk Score (0 or 1) based on features
    risk_prob = (0.05 + data['Age'] / 150 + data['Cholesterol'] / 600 + data['Smoking'] * 0.2)
    data['Risk_Outcome'] = (np.random.rand(count) < risk_prob).astype(int)
    
    return pd.DataFrame(data)

def simulate_research_output(disease: str) -> dict:
    """Simulates Research and Data Agent output links and text."""
    return {
        "data_agent_link": "https://kaggle.com/datasets/synthetic_heart_disease_v1",
        "data_description": f"Synthetic cohort of {disease} patients generated using GAN-based techniques to mimic statistical properties of NHANES data.",
        "paper_link": "https://pubmed.gov/34567890",
        "paper_description": "A meta-analysis identifying primary non-modifiable risk factors (Age, Gender) in cardiovascular mortality, heavily influencing our feature weighting."
    }

# --- 4. AutoViz Logic: Chart Generator (Adapted from Dashboard Agent) ---

def generate_chart_json(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """
    Generates a Plotly chart (JSON string) based on two columns.
    We assume the last column is usually the target ('Risk_Outcome').
    """
    import plotly.express as px
    
    try:
        if df[x_col].nunique() < 10 and x_col != y_col: # Treat as categorical
            # Chart 1: Bar Chart comparing average risk across categories
            fig = px.bar(df.groupby(x_col)[y_col].mean().reset_index(), 
                         x=x_col, y=y_col, 
                         title=f'Average Risk by {x_col}',
                         template="plotly_dark")
        else:
            # Chart 2/3: Scatter Plot for correlation
            fig = px.scatter(df, x=x_col, y=y_col, 
                             color=y_col, # Color by the final Risk Outcome
                             title=f'Feature Correlation: {x_col} vs {y_col}',
                             template="plotly_dark")
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error generating chart for {x_col} vs {y_col}: {e}")
        return None

# --- 5. The Master Agent Endpoint ---
@app.post("/generate_master_dashboard")
def generate_master_dashboard(request: DashboardRequest):
    """
    Runs the full simulation pipeline and returns all data needed for Page 3.
    """
    
    # --- PHASE 1: DATA AGENT (Generate Data) ---
    df = generate_synthetic_data(request.disease_topic, request.record_count)
    
    # --- PHASE 2: ML TRAINER AGENT (Simulate Accuracy) ---
    # We simulate a high, but realistic, accuracy score based on the data size
    simulated_accuracy = round(0.75 + (request.record_count / 200000.0), 4) # Small bonus for more data
    
    # --- PHASE 3: RESEARCH & DATA AGENT (Get Info) ---
    info = simulate_research_output(request.disease_topic)

    # --- PHASE 4: DASHBOARD AGENT (Generate Charts) ---
    
    charts_json_list = []
    
    # Chart 1: Primary Risk Factor Correlation (e.g., Age vs Risk)
    chart1_json = generate_chart_json(df, x_col='Age', y_col='Risk_Outcome')
    if chart1_json: charts_json_list.append(chart1_json)

    # Chart 2: Secondary Risk Factor Correlation (e.g., Cholesterol vs Risk)
    chart2_json = generate_chart_json(df, x_col='Cholesterol', y_col='Risk_Outcome')
    if chart2_json: charts_json_list.append(chart2_json)
    
    # Chart 3: Categorical Factor (e.g., Smoking vs Risk)
    chart3_json = generate_chart_json(df, x_col='Smoking', y_col='Risk_Outcome')
    if chart3_json: charts_json_list.append(chart3_json)

    # --- FINAL MASTER RESPONSE (The complete package for the web page) ---
    return {
        "status": "success",
        "topic": request.disease_topic,
        "record_count": request.record_count,
        "model_accuracy": simulated_accuracy,
        
        # All required links and descriptions for display
        "data_agent_link": info["data_agent_link"],
        "data_description": info["data_description"],
        "paper_link": info["paper_link"],
        "paper_description": info["paper_description"],
        
        # The visual dashboard data
        "charts": charts_json_list
    }

# ----------------------------------------------------------------------
# END OF final_master_dashboard.py CODE
# ----------------------------------------------------------------------