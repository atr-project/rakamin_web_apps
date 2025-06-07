from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (so Streamlit can call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or limit to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train the model once
iris = load_iris()
X = iris.data
y = iris.target
model = RandomForestClassifier().fit(X, y)

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(input: IrisInput):
    data = [[
        input.sepal_length,
        input.sepal_width,
        input.petal_length,
        input.petal_width
    ]]
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0]
    return {
        "prediction": int(pred),
        "label": iris.target_names[pred],
        "probabilities": dict(zip(iris.target_names, proba.round(3).tolist()))
    }
