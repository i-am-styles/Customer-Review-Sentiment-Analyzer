from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import joblib

app = FastAPI()

model = joblib.load("model.joblib")
accuracy = joblib.load("accuracy.joblib")


class ReviewFeatures(BaseModel):
    review: str = Field(min_length=3, max_length=500)


@app.get("/")
def home():
    return {
        "message": "Customer Review Sentiment Analyzer",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "MultinomialNB",
        "accuracy": round(float(accuracy), 2)
    }


@app.post("/predict")
def predict(data: ReviewFeatures):
    try:
        prediction = model.predict([data.review])[0]
        return {"predicted_sentiment": prediction,
                "confidence": accuracy
                }

    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file")

    try:
        df = pd.read_csv(io.BytesIO(await file.read()))

        if "review" not in df.columns:
            raise HTTPException(400, "CSV must contain a 'review' column")

        if df.empty:
            raise HTTPException(400, "CSV contains no data rows")

        df["predicted_sentiment"] = model.predict(df["review"])

        return StreamingResponse(
            io.StringIO(df.to_csv(index=False)),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")