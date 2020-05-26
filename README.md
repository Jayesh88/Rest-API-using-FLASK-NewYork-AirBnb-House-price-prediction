# Rest API using FLASK: NewYork AirBnb House price prediction


Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.
Dataset can be found at  https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data

### Project division:
This project has 3 major parts:
1.	model.py - This contains code for our Machine Learning model to predict employee house prices for NewYork AirBnb based on training data in ‘airbnb.csv’ file.
2.	app.py - This contains Flask APIs that receives house details through GUI or API calls, computes the precited value based on our model and returns it.
3.	templates - This folder contains the HTML template to allow user to enter house detail, area, number of nights and number of people displays the predicted house prices.

### To Execute:
You can run the model,py which will store the regression algorithm used into a file names model.pkl. It is a file which is used to store the algorithm and can be used when executing flask API.
Lastly, You can open Anaconda command prompt where you can type 
##           Python app.py

Which will basically run your model and generate a host id for example: http://127.0.0.1:5000/. You can then directly paste this Id in your browser and you will be able to see fields to enter some values to predict the house prices of AirBnb.


Note: There is a screenshot in this repository which shows how the layout of page looks like. 
