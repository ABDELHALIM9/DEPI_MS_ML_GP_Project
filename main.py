from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from Lstm_Model import predict as pr  # Import predict function
import text_preprocessing as ps       # Import text preprocessing module

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class TextInput(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/predict/")
async def predict_sentiment(input: TextInput):
    # Preprocess the text
    preprocessed_text = ps.preprocess_text(input.text)
    
    # Predict sentiment using the predict function
    prediction = pr([preprocessed_text])  
    val = prediction["score"]
    
    # Determine sentiment based on threshold
    sentiment = 1 if val > 0.5 else 0
    #sentiment=0
    return {"sentiment": sentiment}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn app:app --reload