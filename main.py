


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import uvicorn
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

app = FastAPI(title="AI Engine Anomaly Detector & Mechanic")

# --- CONFIGURE YOUR AI ---
genai.configure(api_key="AIzaSyBgm7c9-4FMzSPrFyluuXYNja9Z7pbfV6I")
llm_model = genai.GenerativeModel('gemini-3.1-flash-lite')

# --- DATA MODELS ---
class TelemetryData(BaseModel):
    rpm: float
    speed: float
    load: float
    coolant: float
    intake: float
    throttle: float

class DtcChatRequest(BaseModel):
    car_year: int
    car_make: str
    car_model: str
    dtc_codes: list[str]
    user_message: str = ""

# NEW: Data model for the Home Screen Chat
class GeneralChatRequest(BaseModel):
    prompt: str
    context: str

# --- ENDPOINTS ---

# 1. Your Existing DTC Mechanic Endpoint
@app.post("/mechanic/chat")
def ai_mechanic_chat(req: DtcChatRequest):
    if req.user_message == "":
        codes_str = ", ".join(req.dtc_codes)
        prompt = f"""
        You are an expert car mechanic. 
        The user owns a {req.car_year} {req.car_make} {req.car_model}.
        Diagnostic Trouble Codes (DTCs) found: {codes_str}.
        
        TASK: Explain ONLY what these codes mean in simple terms.
        Keep the answer extremely short and to the point (maximum 3 sentences).
        Do NOT explain causes or fixes yet.
        
        CRITICAL RULE: You MUST write your entire response in the Urdu language (using Urdu script).
        """
    else:
        prompt = f"""
        You are an expert car mechanic talking to the owner of a {req.car_year} {req.car_make} {req.car_model} with active codes {req.dtc_codes}.
        The user is asking: "{req.user_message}".
        
        TASK: Answer their question directly and concisely. Use bullet points to keep it easy to read.
        CRITICAL RULE: You MUST answer entirely in the Urdu language (using Urdu script).
        """

    try:
        response = llm_model.generate_content(prompt)
        return {"status": "success", "reply": response.text}
    except Exception as e:
        return {"status": "error", "reply": f"Sorry, the AI mechanic is currently unavailable. Error: {str(e)}"}
    




# ... [Your existing imports, AI setup, and Chat endpoints] ...

# 1. Simulate a "Healthy" Engine Baseline for FYP Training
# In a real commercial app, you would load a .pkl file of a pre-trained model.
# For your FYP defense, we train it on the fly with "normal" driving data.
healthy_baseline = pd.DataFrame({
    'rpm': [800, 2000, 2500, 3000, 850, 1500, 2200],
    'speed': [0, 40, 60, 80, 0, 30, 50],
    'load': [20, 40, 50, 60, 25, 35, 45],
    'coolant': [90, 92, 95, 93, 91, 90, 94],
    'intake': [30, 35, 40, 38, 32, 33, 36],
    'throttle': [10, 20, 25, 30, 12, 18, 22]
})

# Initialize and train the Isolation Forest
ml_model = IsolationForest(contamination=0.1, random_state=42)
ml_model.fit(healthy_baseline)

# 2. THE ML ANALYSIS ENDPOINT
@app.post("/analyze")
def analyze_telemetry(data: TelemetryData):
    try:
        # Convert incoming Flutter data to a DataFrame
        current_data = pd.DataFrame([data.model_dump()]) # Use .dict() if using older Pydantic
        
        # --- A. ANOMALY DETECTION (Machine Learning) ---
        # Predict returns 1 for normal, -1 for anomaly
        prediction = ml_model.predict(current_data)[0]
        is_anomaly = bool(prediction == -1)
        
        # --- B. BEHAVIOR ANALYSIS & FAILURE PREDICTION (Heuristics) ---
        warnings = []
        behavior_score = 100 
        
        if data.coolant > 105:
            warnings.append("Critical: Cooling system failure predicted. High risk of engine overheating.")
            behavior_score -= 40
        if data.rpm > 5500 and data.speed < 40:
            warnings.append("Behavior: Aggressive revving detected. Unnecessary engine wear.")
            behavior_score -= 20
        if data.load > 85:
            warnings.append("Warning: Extreme engine load. Possible transmission strain.")
            behavior_score -= 15
        if data.speed > 5:
            warnings.append("Warning: High-speed driving detected. Possible transmission strain.")
            behavior_score -= 15    

        # --- C. AI DIAGNOSTIC SYNTHESIS ---
        if is_anomaly and not warnings:
            # The ML model caught something the hardcoded rules missed!
            ai_diagnosis = "Invisible Anomaly Detected: Sensor patterns deviate from healthy baseline. Recommend early inspection."
        elif is_anomaly and warnings:
            ai_diagnosis = f"Active Threat: {warnings[0]}"
        else:
            ai_diagnosis = "Engine operating within optimal ML parameters."

        return {
            "status": "success",
            "is_anomaly": is_anomaly,
            "behavior_score": behavior_score,
            "warnings": warnings,
            "ai_diagnosis": ai_diagnosis
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 2. NEW: Home Screen General Chat Endpoint
@app.post("/chat")
def general_ai_chat(req: GeneralChatRequest):
    prompt = f"""
    You are DriveAI, an expert and friendly automotive mechanic assistant.
    You are having a casual conversation with the car owner.
    
    Context about their car: {req.context}
    
    User Question: {req.prompt}
    
    TASK: Answer their question accurately and concisely. Be helpful and professional.
    CRITICAL RULE: You MUST provide your explanation primarily in English, followed by a brief summary in Urdu (using Urdu script).
    """

    try:
        response = llm_model.generate_content(prompt)
        return {"status": "success", "response": response.text}
    except Exception as e:
        return {"status": "error", "response": f"Sorry, the AI mechanic is currently unavailable. Error: {str(e)}"}


# --- RUNNER ---
if __name__ == "__main__":
    # The reload=True flag forces the server to update immediately when you hit Save!
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)