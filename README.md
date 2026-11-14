# House Price Prediction API

FastAPI coursework service exposing a recovered regression model through validated prediction and health endpoints.

## Endpoints

- `GET /health` — model-loading health.
- `GET /model/info` — model metadata.
- `POST /predict` — prediction from 13 validated property features.
- `GET /docs` — interactive API documentation.

## Run

Install `requirements.txt`, then run `uvicorn main:app --reload` from this directory.

## Source and model

The three Python source files, model metadata, and model artifact come from the recovered approved coursework archive. The service wraps a supplied/recovered model; this repository does not claim to include model training. No dependency lockfile accompanied the recovered submission; this is not evidence of a missing course deliverable. The included requirements list is a publication convenience inferred from imports; loading a scikit-learn artifact may require its original version.

The metadata’s training date and RMSE are preserved as artifact metadata, not independently reproduced results. No new runtime test or production deployment is claimed. Only load pickle/joblib model files from sources you trust.

## Background

Completed as part of TripleTen’s AI/ML Engineering curriculum. This repository presents my coursework and analysis; course-provided scaffolding, models, and data are identified where applicable. Repository publication date is not the original project completion date.
