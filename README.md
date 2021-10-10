# {{ Project name}}

This project is an example of a fully packaged machine learning model that can be readily deployed in any computing environment:
- Gradient boosted trees as the core model using Microsoft's LightGBM implementation
- Model training anf finetuning using bayesian optimization
- REST model exposition using FastAPI framework
- Docker to package the whole thing in a containerized run environment


The API exposes three end-points:
- `/train` to train and finetune the model on X, y datasets passed as payload
- `/validation` to train and evaluate the model performance using cross-validation and report metrics
- `/prediction` for inference on X dataset passed as payload