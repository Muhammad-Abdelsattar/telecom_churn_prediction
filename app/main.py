from fastapi import FastAPI
from inference import InferencePipeline

app = FastAPI()
pipeline = InferencePipeline(pipeline_path="models/pipeline.pkl")

@app.get("/")
def index():
    return "Hello"

@app.post("/predict")
def predict(data: list[dict]|dict):
    if(type(data) == dict):
        data = [data]
    return pipeline(data)
