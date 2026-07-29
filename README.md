# Student Performance Prediction

An end-to-end machine learning project that predicts a student's math score from demographic and academic features. The project includes data ingestion, preprocessing, model training, a FastAPI web app, Docker support, and Render deployment configuration.

## Demo

This project is deployed as a web app. Open the landing page, click the prediction button, fill in the student details form, and submit it to get a predicted math score.

## Features

- Trains multiple regression models and compares their performance.
- Saves the fitted preprocessor and model artifacts for inference.
- Serves a browser-based prediction form with FastAPI and Jinja2 templates.
- Supports local development, Dockerized runs, and Render deployment.

## How To Use

### 1. Train the model

Run the ingestion script to build the training and test datasets, fit the preprocessing pipeline, and save the trained model artifacts:

```bash
python src/components/data_ingestion.py
```

### 2. Start the web app

Launch the FastAPI application locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Open the app in your browser

- Landing page: `http://127.0.0.1:8000`
- Prediction form: `http://127.0.0.1:8000/predict`

### 4. Submit student details

Fill in the form and submit it to receive the predicted math score.

## Project Structure

```text
MLProject/
├── app.py
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── data.csv
│   ├── test.csv
│   └── train.csv
├── Dockerfile
├── render.yaml
├── requirements.txt
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── params.yaml
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── home.html
│   └── index.html
└── notebook/
	└── data/
		└── stud.csv
```

## Requirements

- Python 3.10 or 3.11 recommended
- pip
- Docker, if you want to run the app in a container

## Installation

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Local Setup

Run the application locally with Uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

- `http://127.0.0.1:8000` for the landing page
- `http://127.0.0.1:8000/predict` for the prediction form

## Training the Model

The training workflow reads the dataset from `notebook/data/stud.csv`, splits it into train and test sets, preprocesses the features, evaluates several models, and saves the best model plus preprocessor into `artifacts/`.

After training completes, make sure these runtime files exist:

- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

The web app uses those files during prediction.

## Prediction Workflow

1. Fill out the student information form in the browser.
2. FastAPI receives the form data and builds a dataframe.
3. The prediction pipeline loads the saved preprocessor and model.
4. The model returns the predicted math score.

## Docker

Build and run the container locally:

```bash
docker build -t mlproject .
docker run -p 8000:8000 mlproject
```

The container uses the `PORT` variable provided by Render when deployed there.

## Render Deployment

The project includes a Docker-based Render configuration.

Deployment steps:

1. Push the repository to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repository.
4. Render will use the Dockerfile and `render.yaml`.
5. Deploy the service.

No extra environment variables are required for the current codebase.

## Notes

- Keep `artifacts/model.pkl` and `artifacts/preprocessor.pkl` available for inference.
- Do not commit large generated training outputs such as `catboost_info/`.
- If you retrain the model, rebuild the Docker image or redeploy so the updated artifacts are included.
- Screenshots can be added later once the UI is finalized.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
