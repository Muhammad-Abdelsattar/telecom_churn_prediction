from fastapi import FastAPI
from inference import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline(pipeline_path="models/pipeline.pkl")

@app.get("/")
def index():
    return "Hello"

@app.post("/predict")
def predict(data: list[dict]):
    return pipeline(data)
