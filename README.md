# Hugging Face x Azure ML

Azure ML Endpoints is a service provided by Microsoft as part of its Azure Machine Learning platform. It allows users to deploy and manage machine learning models as scalable and secure cloud services. This service simplifies the deployment process, provides built-in monitoring tools, and ensures that models are accessible via HTTPS APIs, making it easier to integrate machine learning capabilities into various applications.

This repository serves as a How-To example on how to deploy a custom model from Hugging Face Hub (or basically any other model) to Azure ML.

In order to deploy a custom model to Azure ML and create an endpoint for inferences, please refer to the ```deploy_model_sdk_v2.ipynb``` notebook. To be able to run everything found in the notebook, please install all packages listed in ```requirements.txt```.

In early 2024 Microsoft released the newest version of Azure ML Python SDK v2. The new version differs significantely in most cases from the previous v1. Microsoft recommends to use version 2 from now on. Version 1 will be deprecated on September 30, 2025.

Please find the old How-To examples in `deploy_model_sdk_v1.ipynb`.

Have fun deploying models to Azure ML.