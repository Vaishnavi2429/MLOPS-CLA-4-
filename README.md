# Titanic Survival Prediction – Machine Learning Deployment

A beginner-friendly machine learning project that predicts whether a Titanic passenger survived using passenger details. The notebook trains a scikit-learn classification pipeline, saves the model, runs predictions on unseen records, and exposes predictions through a FastAPI REST API.

## Features

- Loads the Titanic training dataset from a ZIP file.
- Preprocesses numeric and categorical features.
- Trains a `RandomForestClassifier`.
- Evaluates the model with accuracy and a classification report.
- Saves the trained model as `model.pkl`.
- Predicts outcomes for one passenger and for batches of 20 unseen passengers.
- Demonstrates sequential (“streaming-style”) predictions.
- Provides a FastAPI API with health-check and prediction endpoints.
- Includes a Python client for calling the API.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- joblib
- FastAPI and Uvicorn
- Pydantic
- requests
- pyngrok (for temporary public access from Google Colab)

## Dataset

The notebook expects a ZIP file at `/content/titanic.zip` containing a Titanic `train.csv` file. Upload the dataset ZIP to the Colab session and ensure the path matches the notebook.

The training data must include the target column `Survived` and the features used by the notebook, including `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`, along with any additional columns present in the original dataset.

- `Survived = 1`: passenger survived
- `Survived = 0`: passenger did not survive

The notebook reserves a portion of the data as unseen records, then splits the remaining development data into training and testing sets. This is intended to keep the unseen records separate from model training and evaluation.

## Run in Google Colab

1. Open `CLA_4 (1).ipynb` in Google Colab.
2. Upload the Titanic ZIP dataset and make sure it is available as `/content/titanic.zip`.
3. Run the notebook cells in order.
4. Check the printed dataset information and evaluation metrics.
5. Confirm that the output directory `/content/titanic-ml-deployment` contains the saved model and generated files.
6. Run the prediction cells to test a single passenger, batch predictions, and sequential predictions.
7. Run the FastAPI setup cells to start the API and check its health endpoint.
8. If using ngrok to expose the API publicly, configure your own ngrok authentication token securely. Do not commit or share tokens in notebooks or repositories.
9. Run `client.py` and enter the active API base URL when prompted.

> Note: Colab runtimes are temporary. If the runtime disconnects or resets, generated files and running API processes may need to be recreated by rerunning the notebook.

## Generated Project Files

The notebook is designed to create files in `/content/titanic-ml-deployment`, including:

- `model.pkl` – saved trained preprocessing/model pipeline.
- `unseen_records.csv` – held-out passenger records for demonstration.
- `batch_predictions.csv` – predictions for up to 20 unseen passengers.
- `app.py` – FastAPI application.
- `client.py` – Python client for calling the API.
- `requirements.txt` – Python package list.

The exact files available depend on which notebook cells have run successfully.

## API Endpoints

### `GET /health`

Checks whether the API is running and the model loaded.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /predict`

Accepts passenger features and returns the predicted class and label.

Example request:

```json
{
  "Pclass": 3,
  "Sex": "female",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Example response format:

```json
{
  "prediction": 1,
  "label": "Survived"
}
```

The actual prediction depends on the trained model. The interactive API documentation is available at `/docs` while the API is running.

## Running the API Locally

After generating `model.pkl`, ensure `app.py` and `model.pkl` are in the same directory. Install the requirements and start Uvicorn from that directory:

```bash
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs` in a browser. These commands assume the model file has already been created by running the training portion of the notebook.

## Limitations

- This is an educational prediction project, not a system for making real-world safety or life-critical decisions.
- Predictions are estimates from historical Titanic data and do not establish causation.
- Model performance depends on the dataset, preprocessing, train/test split, and random seed.
- A public ngrok URL is temporary and should not be treated as a secured production deployment.
- Do not publish authentication tokens, API secrets, or private data.

## Security Note

The notebook includes an ngrok authentication-token assignment. If that token is real or has been shared, revoke it in your ngrok account and create a new one. Store secrets in Colab Secrets or environment variables rather than hard-coding them into a notebook.

## Future Improvements

- Add cross-validation and compare multiple classification algorithms.
- Record evaluation metrics and a confusion matrix.
- Add automated tests for the API.
- Add Docker support and deploy to a managed hosting platform.
- Add input validation and monitoring appropriate for a production environment.

## License

Add a license before redistributing this project. If the dataset is from Kaggle, follow the dataset's terms and attribution requirements.
