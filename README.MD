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
Goal: Predict defects in C programs given various various attributes about the code.
The dataset for this competition (both train and test) was generated from a deep learning model trained on the Software Defect Dataset.

## Following up on Objectives of the project
* Check the data types for the features
* Checked for missing values
* Checked for outliers and duplicates
* Trained data using 3 different models, xgboost presented the best accuracy score
* The model got turned into a webservice using Flask

##  To deploy Model Locally with Flask (You do not need this step to test the deployed service on render)
* create virtual environment from pipfile and pipfilelock folders running command below:
  - ``` pipenv install ```
  * Activate shell  
  - ``` pipenv shell ``` 
The above line of code is meant to install all required dependencies listed in the pipfiles in a virtual environment, these can also be installed directly by running:
    - - ``` pip install name_of_package ```

* Run service app (predict.py) in virtual environment
 - ``` python predict.py ``` OR ``` pipenv run python predict.py ``` (if using virtual environment)
* Run test file in a sepearate virtual environment terminal (test.py)
  - ``` python test.py ```

## Deploying Model Locally with Docker
* Install and run docker on local machine
* Build docker image in a terminal (ensuring you're in project folder):
  -  ``` docker build -t project1 . ```

![alt text](/docker_build_image.png)

* Run docker image:
  - ``` docker run -it --rm -p 9696:9696 project1 ```

![alt text](/docker.png)

Having successfully run the above, open another terminal and run the test file as follows:
  - ``` python test.py ```

## Test model from already deployed cloud service via render
* Run test file (test_render.py) for render containing 'code-defect-classifier.onrender.com' as host:
  - ``` python test_render.py ```

## Deploy model to the cloud
Render was used for cloud deployment.
- Create a Render account
- Copy github repository link to render webservice creation 
- Deploy service
- Copy render deploymnet link and update test_render.py scrips as "host"
- activate virtual environment
- Run test file:
  - ``` pipenv shell ```
  - ``` python test_render.py ```


![alt text](/render_deploy.png)
