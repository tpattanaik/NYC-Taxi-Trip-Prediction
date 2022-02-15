# NYC-Taxi-Trip-Prediction

# Abstract

A typical taxi company faces a common problem of efficiently assigning the cabs to passengers so that the service is smooth and hassle free. One of main issue is determining the duration of the current trip so it can predict when the cab will be free for the next trip.
For this ML Project, I have used NYC Taxi Dataset.
The primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers and several other variables. The dataset is based on the 2016 NYC yellow cab trip record data.
We will use different EDA concepts using Python and their packages, Matplotlib and seaborn to come up with various graphs and charts by using the dataset and get some insights of data which will help us in modelling. We will use different regression methods to develop a model and see how our model behaves.

# Problem statement

Our task is to build a model that predicts the total ride duration of taxi trips in New York City. Based on
the individual trip attributes, we should predict the duration of each trip in the test set.

# Packages Required

1. Numpy
2. Pandas
3. Seaborn
4. Matplotlib
5. Datetime
6. XGBRegressor
7. Haversine 
8. Linear Regression
9. Metricc
10. Train test split
11. Gridsearch CV

# Approaches

After loading and reading the dataset in pandas Data Frame, we know that our dataset has 1458644 rows
and 11 columns.

We first imported all necessary libraries in our notebook which required for the analysis.

During Data exploration, we got to know that there are no Null values in the dataset. We also converted
timestamp to datetime format to fetch other details which will help us to gain more insights from the
data. We used Haversine formula to calculate distance between pickup and dropoff coordinates. Also we
dummified all the categorical features. We made the dataset ready for further analysis by doing these
data exploration techniques.

We did Univariate and Bivariate analysis of the dataset to get more insights and clear picture before
modelling the data.

In **Univariate Analysis**, we got to know some insights of data as given below:

• Passengers- There were some outliers as in few trips the number of passengers mentioned are
7,8 or 9. So we took the mean, median and mode as 1 and replace it to 0 passengers in the list.
From the analysis, it is evident that most of the trips was taken by single passenger.
• Vendors- We analyze the data only for 2 vendors which are listed as 1 and 2 in the dataset.
Vendor 2 is more famous among the population.
• Distance- From the analysis, it is evident that most of the rides are completed between 1-10 Kms
with some rides with distance between 10-30 Kms.
• Trip duration- We can observe that most of the trips took 0-30 mins to complete.
• Speed- Most of the trips were done at a speed range of 10-20 Km/Hr.
• Total trips per hour- Here we can see an increasing trend of taxi pickups starting from Monday till
Friday. The trend starts declining from Saturday till Monday. Taxi pickups seems to be consistent
across the week at 15 Hours i.e., at 3 PM.
• Total trips per month- It was quite a balance across the months here.






