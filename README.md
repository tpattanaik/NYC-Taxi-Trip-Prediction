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

# Steps Involved

**1. Exploratory Data Analysis**

After loading the dataset we performed this method by comparing our target variable that is Trip duration with other independent variables. This process helped us figuring out various aspects and relationships among the target and the independent variables. It gave us a better idea of which feature behaves in which manner compared to the target variable.

**2. Null values Treatment**

There is no NaN/NULL record in the dataset, So we dont have to impute any record.

**3. Datetime format**

A date in Python is not a data type of its own, but we can import a module named datetime to work with dates as date objects. We can convert the timestamp to datetime format in our dataset to fetch details like weekday, month, pickup hour etc. 

**4. Haversine Formula**

The Haversine formula calculates the shortest distance between two points on a sphere using their latitudes and longitudes measured along the surface. It is important for use in navigation. We used a function called calc_distance to calculate distance between pickup and dropoff coordinates using Haversine formula.

**5. Encoding of categorical columns**

A dataset may contain various type of values, sometimes it consists of categorical values. Categorical features that are in string format cannot be understood by the machine and needs to be converted to numerical format. So, in-order to use those categorical value for programming efficiently we create dummy variables. A dummy variable is a binary variable that indicates whether a separate categorical variable takes on a specific value. We dummify all the categorical variables like “store_and_fwd_flag”. 

**6. Univariate Analysis**

Univariate analysis is the simplest form of analysing data. “Uni” means “one”, so in other words your data has only one variable. It doesn't deal with causes or relationships (unlike regression) and it's major purpose is to describe; It takes data, summarizes that data and finds patterns in the data. There are many univariate columns in our dataset where we can analyse and find some insights. Some columns where we did univariate analysis are passengers, vendor, Distance, trip duration, speed, Trips per hour/weekday/month.

**7.Bivariate Analysis**

Bivariate analysis is used to find out if there is a relationship between two sets of values. It usually involves the variables X and Y. Bivariate analysis is one of the simplest forms of quantitative analysis. It is one of the simplest forms of statistical analysis, used to find out if there is a relationship between two sets of values. There are many bivariate columns in our dataset where we can analyse and find some insights. Some columns where we did bivariate analysis are
Trip duration per hour/weekday/month/vendor, distance per hour/weekday/month/vendor, speed per hour/weekday, Passengers per vendor.

**8. Split data**

Before training our model on the dataset, we need to split the dataset into training and testing datasets. This is required to train our model on the major part of our dataset and test the accuracy of the model on the minor part. This will divide our dataset randomly with a ratio of 80/20 where training set consists of more than 1 million records and test dataset with more than .35 million records. Let's train our model on the training set now.

**9. Feature Selection**

Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modelling and, in some cases, to improve the performance of the model. We select a subset of the original feature set based on the statistical significance of different parameters. We used backward elimination technique to select the best features to train our model. It displays some statistical metrics with their significance value. Like, It shows the p values for each feature as per its significance in the whole dataset. It also shows the adjusted R squared values to identify whether removing or selecting the feature is beneficial or not. We only look at the P and adjusted R squared value to decide which features to keep and which needed to be removed. Duration variable assigned to Y because that is the dependent variable. Features such as id, timestamp and weekday were not assigned to X array because they are of type object. And we need an array of float data type. Fit stats model on the X array to figure out an optimal set of features by recursively checking for the highest p value and removing the feature of that index. Here we took the level of significance as 0.05 i.e., 5% which means that we will reject feature from the list of array and re-run the model till p value for all the features goes below .05 to find out the optimal combination for our model.

**10. Feature Extraction**

Here, we build a new set of features from the original feature set. We used PCA for feature extraction i.e. Principal Component Analysis. It is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components. Here we see that almost 40 variables are needed for capturing atleast 99% of the variance in the training dataset. Hence, we will use the same set of variables.

**11. Correlation Analysis**

Correlation analysis is a method of statistical evaluation used to study the strength of a relationship between two or more, numerically measured, continuous variables. This analysis is useful when we need to check if there are possible connections between variables. We will utilize Heatmap for our analysis. A heatmap is a graphical representation of data that uses a system of color-coding to represent statistical relationship between different values.








