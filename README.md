likely.js
=========
A javascript library for collaborative filtering and recommendation engines designed for node.js

Description
---------
Likely.js is a library used for collaborative filtering and making recommendations. It takes an input
training matrix where the rows represent users and the columns represent items. Each entry in the matrix
is the rating the user has given to that item. After training, you can retrieve a list of recommended
items and their estimated ratings for any given user. 

For example, you may provide as input a training matrix where each row is a customer and the columns are the ratings they give 
to various movies. Then you might use this model to get recommendations for movies a given customer might like that they have
not yet seen.

Installation
---------

    npm install likely

Usage 
---------
    
To use Likely, require the likely module and follow the 3 steps below.
<!-- language: lang-js -->
    var Recommender = require('likely');
    
#### STEP 1. Assemble the input
Create an input matrix where rows are users and the columns are items
A cell represents the rating of that item by the entity. inputMatrix[0][1] is the 
rating of item 1 by user 0
<!-- language: lang-js -->
    var inputMatrix = [ [ 1, 2, 3, 0 ],
                        [ 4, 0, 5, 6 ],
                        [ 7, 8, 0, 9 ]
                      ];
                      
Labels to provide more context to the input. Row 0 of the input matrix corresponds to label rowLabels[0]
<!-- language: lang-js -->
    var rowLabels = ['John', 'Sue', 'Joe'];
    var colLabels = ['Red', 'Blue', 'Green', 'Purple'];

Using these values, John rates Red 1 while Joe rates Red 7. Sue has no rating for Blue.

#### STEP 2. Train the model
Using the inputMatrix you build a model, which estimates the ratings for all entities for all users. 
<!-- language: lang-js -->
    var Model = Recommender.buildModel(inputMatrix, rowLabels, colLabels);
    
or, if you don't have or care about labels
<!-- language: lang-js -->
    var Model = Recommender.buildModel(inputMatrix);
    
The Model object now contains a matrix of the same size as inputMatrix but with estimates ratings for
all items for all entities. 
    
#### STEP 3. Extract recommendations
There are a few ways to retrieve recommendations from the model you have built.

**Example 1**: Retrieve a list of all items not already rated by a user, sorted by estimated ratings using labels.
<!-- language: lang-js -->
    var recommendations = model.recommendations('John');
	
    // recommendations = [['Purple', 1.34]];
    

**Example 2**: Retrieve a list of all items not already rated by a user, sorted by estimated ratings without labels.
<!-- language: lang-js -->
    var recommendations = model.recommendations(0);
	
    // recommendations = [[3, 1.34]];

**Example 3**: Retrieve a list of all items, sorted by the ratings for a given user (both estimated and actual), using labels.
<!-- language: lang-js -->
    var allItems = model.rankAllItems('John');
	
    // allItems = [['Green', 3.00], ['Blue', 2.00], ['Purple', 1.34], ['Red', 1.00]];

**Example 4**: Retrieve a list of all items, sorted by the ratings for a given user (both estimated and actual) without labels.
<!-- language: lang-js -->	
    var allItems = model.rankAllItems(0);
	
    // allItems = [[2, 3.00], [1, 2.00], [3, 1.34], [0, 1.00]];


Handling Bias 
---------
In a lot of input data there is inherent bias. For example, a given user might tend to rate a great movie as 10 stars which another user never gives above 8 stars. 
These inherent biases can skew the estimations by providing false signals to the model. To handle these, you can adjust for biases by including them in your model 
building. 
<!-- language: lang-js -->	
	var bias = Recommender.calculateBias(input);
	
    // Build the model using the training set and considering the bias
    var model = Recommender.buildModelWithBias(trainingSet, bias);
	
The resulting model should be very similar to the one obtained without providing bias, but more accurate. 

FAQ
---------
#### I used Likely.js to generate recommendations based on my data. How do I know if it's working?

As with any machine learning, it's important to test to make sure it works with your data. The best way to do this is to 
take your input data (known ratings) and divide it into two groups: training and cross-validation (CV). To see if the model is working you 
should train the model using your training set and check the error using the CV set. If it looks good then you can apply
the model built using the training set to your entire data and have confidence in the estimates. 

In practice, this would look as follows:
<!-- language: lang-js -->
    var Recommender = require('likely.js');

    // Build the model using the training set
    var model = Recommender.buildModel(trainingSet);
	
    // Calculate the error from the produced model against the CV set of known values
    var totalError = Recommender.calculateTotalError(model.estimate, crossValidationSet);


#### I trained a model using my training set but the error is too high against my CV set. What do I do?

This is common for any machine learning application. In this case it will be necessary to tune the parameters of the 
Likely.js learning algorithm to better fit your data. The available options that you can adjust are as follows:
<!-- language: lang-js -->	
	var Recommender = require('likely.js');
	
	// Number of iterations it will use to try and learn the model, the larger the better. 
	// 	However, the larger the steps the longer it will take to train the model.
	Recommender.DESCENT_STEPS;  // DEFAULT = 5000 
	
	// The rate of learning. If your error is very large you can try making this larger.
	// 	However, if the learning rate is too large the error will increase.
	Recommender.ALPHA;          // DEFAULT = 0.0005
	
	// The regularization factor, this prevents over-fitting. If your error is not affected by changing
	// 	the steps or learning rate try adjusting this. It should never be a large value.
	Recommender.BETA;           // DEFAULT = 0.0007
	
	// The number of features to learn. In theory the more the better but a larger value will slow down
	// 	the training. 
	Recommender.k; 		    // DEFAULT = 5


#### How is bias calculated?

When you call Recommender.calculateBias(input) the Recommender computes the overall mean score and average deviation from the mean for each row and column. 
The overall mean and row/column deviations can then be used to normalize all the data.

This is a fairly simple model of bias but should improve performance in most cases. 

Tests 
---------
Likely has both unit and regression tests. To make sure your version is working you should run both.

Running the unit tests:

    ./scripts/run_unit_tests.sh

Running the regression test:

    node regression_test.js
    
Note that the unit test requires you have installed the expresso testing framework which can be done easily:

    npm install expresso

Or install the npm with the -dev flag

    npm install -dev likely

How it works
---------
Likely.js uses matrix factorization to estimate the rating values not provided. MF attempts to find two matrices, P and Q 
such that the product is equal to the input matrix. 

    P x Q = inputMatrix
    
P and Q are initialized as random matrices and Gradient Descent is used to minimize the error over a number of iterations.
Regularization is used to make sure that there is no overfitting, which means the actual result of P x Q is not exactly
the inputMatrix but something that closely approximates it. The effect of this approximation is estimates for the 
values not provided in the inputMatrix. 

These estimated values are what are used to provide the recommendations.

Likely.js uses Stochastic Gradient Descent to estimate P and Q which, while simple, can be slow for very large input matrices.
This is the most basic but effective method of recommendations that was highlighted in the Netflix prize: 
http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf 
