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

--> After loading and reading the dataset in pandas Data Frame, we know that our dataset has 1458644 rows
and 11 columns.

--> We first imported all necessary libraries in our notebook which required for the analysis.

--> During Data exploration, we got to know that there are no Null values in the dataset. We also converted
timestamp to datetime format to fetch other details which will help us to gain more insights from the
data. We used Haversine formula to calculate distance between pickup and dropoff coordinates. Also we
dummified all the categorical features. We made the dataset ready for further analysis by doing these
data exploration techniques.

--> We did Univariate and Bivariate analysis of the dataset to get more insights and clear picture before
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

In **Bivariate Analysis**, we got to know some insights of data as given below:

• Trip duration per hour- Average trip duration is lowest at 6 AM and highest at 3 PM. It is similar
during early morning hours.

• Trip duration per weekday- The trip duration is almost equally distributed across the week on a
scale of 0-1000 minutes with minimal difference in the duration times.

• Trip duration per month- There is an increasing trend in the average trip duration along the
subsequent month.

• Trip duration per vendor- Average trip duration for vendor 2 is higher than vendor 1 by approx.
200 seconds i.e., 3 minutes per trip.

• Distance per hour/weekday/month- Trip distance is higher during early morning hours and equal
from morning till the evening varying around 3-3.5 Kms. Its fairly equal distribution with average
distance metric varying around 3.5 Km/hr with Sunday being the top. Month wise also the
distribution is almost equivalent with 5th month being the highest and 2nd month being the
lowest.

• Passenger count per vendor- It seems that vendor 2 trips generally consist of 2 passengers as
compared to the vendor 1 with 1 passenger.

--> **Feature selection-** 

We used backward elimination technique to select best features to train our model.
For now we looked only at the P and adjusted R squared value to decide the features to keep and which
needed to be removed. Here we took the level of significance as 5%.

--> **Feature extraction-**

We used PCA for feature extraction i.e., Principal component analysis.

--> We split our dataset randomnly with a ratio of 80/20 where the training set consists of more than 1
million records and test dataset with more than 0.35 million records.

--> **Correlation analysis-**

All of the features shows NO correlation at all because feature extraction removes
all collinearity.

# Models

We used Multiple linear regression and XGBoost Regressor for modelling.

**Linear Regression**- 

Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. A linear regression line has an equation of the form 
Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0). The most common method for fitting a regression line is the method of least-squares. This method calculates the best-fitting line for the observed data by minimizing the sum of the squares of the vertical deviations from each data point to the line (if a point lies on the fitted line exactly, then its vertical deviation is 0). Because the deviations are first squared, then summed, there are no cancellations between positive and negative values.

In multiple linear regression, we observed that it has very poor root mean squared value and the low
variance score which is also bad. So both the models i.e., from the feature selection and extraction group
resulted quite bad in prediction.

**XGBoost**-

To understand XGBoost we have to know gradient boosting beforehand. Gradient boosted trees consider the special case where the simple model is a decision tree. In this case, there are going to be 2 kinds of parameters P: the weights at each leaf, w, and the number of leaves T in each tree (so that in the above example, T=3 and w=[2, 0.1, -1]).
When building a decision tree, a challenge is to decide how to split a current leaf. For instance, in the above image, how could I add another layer to the (age > 15) leaf? A ‘greedy’ way to do this is to consider every possible split on the remaining features (so, gender and occupation), and calculate the new loss for each split; you could then pick the tree which most reduces your loss.
XGBoost is one of the fastest implementations of gradient boosting. trees. It does this by tackling one of the major inefficiencies of gradient boosted trees: considering the potential loss for all possible splits to create a new branch (especially if you consider the case where there are thousands of features, and therefore thousands of possible splits). XGBoost tackles this inefficiency by looking at the distribution of features across all data points in a leaf and using this information to reduce the search space of possible feature splits.


In XGBoost regressor, there is a significant improvement in the RMSE score for the tuned XGBoost
regressor when trained on the feature selection group. Also, the RMSE score on the raw data and feature
selected data are same.

# Few things to improve our model:

• Add more training instances to improve validation curve in the XGBoost model.

• Increase the regularization for the learning algorithm.

• Reduce the numbers of features in the training data that we currently use.

# End notes:

In this project we covered various aspects of the Machine learning development cycle. We observed that
the data exploration and variable analysis is a very important aspect of the whole cycle and should be
done for thorough understanding of the data. We also cleaned the data while exploring as there were
some outliers which should be treated before feature engineering. Further we did feature engineering to
filter and gather only the optimal features which are more significant and covered most of the variance in
the dataset. Then finally we trained the models on the optimum featureset to get the results.







