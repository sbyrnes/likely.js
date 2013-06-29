likely.js
=========

***NOTE: Likely.js is a work in progress. You probably already figured that out from the fact that it is just a README right now 
but it will be more soon!***

A javascript library for collaborative filtering and recommendation engines designed for node.js

Usage 
---------
    
    Recommender = require('likely.js');
    
    // Input matrix where rows are input entities (such as users) and the columns are items
    //    A cell represents the rating of that item by the entity. inputMatrix[0][1] is the 
    //    rating of item 1 by user 0
    var inputMatrix = [ [ 1, 2, 3, 0 ],
                        [ 4, 0, 5, 6 ],
                        [ 7, 8, 0, 9 ]
                      ];

    var Prediction = Recommender.train(trainingMatrix);
    
    // Prediction is now a matrix of the same size as inputMatrix but with estimates ratings for
    // all items for all entities. 
    
    var recommendedList = Prediction.getRecommendations(0);
    
    // recommendedList is now a sorted list of all items not rated by user 0, sorted by the estimated rating.

Description
---------
Likely.js is a library of utilities used for collaborative filtering and recommendation engines. It takes an input
training matrix comprised of the object IDs and values for the set of features chosen. After training, you provide
an input vector comprised of various values to produce a list of recommended object IDs. 

For example, you may provide as input a training matrix where each row is a customer and the columns are the ratings they give 
to various movies. Then you might submit a single users rating history as a vector to produce a list of movies they might like.

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
