# SOFTWARE DEFECT PREDCITION
## Datatalks ML Zoomcamp Mid term project

## Objectives of Midterm Project
* Find a dataset
* Explore and prepare the data
* Train several models and select the best model
* Train full test dataset with final model, save model
* Load model into a web service
* Deploy model locally with Docker
* Deploy model to the cloud

The Dataset used in the project was gotten from a [kaggle competetion](https://www.kaggle.com/competitions/playground-series-s3e23/data). 
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Software Defect Dataset. Feature distributions are close to, but not exactly the same, as the original. Feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.

## Following up on Objectives of the project
* Check the data types for the features
* Checked for missing values
* Checked for outliers and duplicates
* Trained data using 3 different models, xgboost presented the best accuracy score
* The model got turned into a webservice using Flask

## Deploying Model Locally with Docker
After ensuring the model runs fine locally with flask, proceed to deploy the model on docker and build image as follows:
* Install and run docker on local machine
* Ensure virtual environment is running:
  - pipenv install
  - pipenv shell
* Build docker image:
  - docker build -t project1 .
* Run docker image:
  - docker run -it --rm -p 9696:9696 project1 

Having successfully run the above, activate another virtual environment for running the test file as follows:
  - python test.py

## Deploy model to the cloud
Render was used for cloud deployment.
- Create a Render account
- Copy github repository link to render webservice creation 
- Deploy service
- Copy render deploymnet link and update test_render.py scrips as "host"
- activate virtual environment
- Run test file:
  - pipenv shell
  - python test_render.py
