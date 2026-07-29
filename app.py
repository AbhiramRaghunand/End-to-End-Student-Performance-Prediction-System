from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import uvicorn


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
    request,
    "index.html",
    {
        "request": request,
        "prediction": None,
        "error": None
    }
)

@app.get("/predict", response_class=HTMLResponse)
def predict_get(request: Request):
    return templates.TemplateResponse(
    request,
    "home.html",
    {
        "request": request,
        "prediction": None,
        "error": None
    }
)

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    try:
        form_data = await request.form()
        data = CustomData(
            gender=form_data.get("gender"),
            race_ethnicity=form_data.get("race_ethnicity"),
            parental_level_of_education=form_data.get("parental_level_of_education"),
            lunch=form_data.get("lunch"),
            test_preparation_course=form_data.get("test_preparation_course"),
            reading_score=int(form_data.get("reading_score")),
            writing_score=int(form_data.get("writing_score"))
        )

        prediction_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(prediction_data)
        # return templates.TemplateResponse("home.html", {"request": request, "prediction": round(float(prediction[0]), 2)})
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "request": request,
                "prediction":  round(float(prediction[0]), 2),
                "error": None
            }
            )
    except Exception as e:
        # return templates.TemplateResponse("home.html", {"request": request, "error": str(e)})
        return templates.TemplateResponse(
            request,
            "home.html",
            {
                "request": request,
                "prediction": None,
                "error": str(e)
            }
        )

    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



    