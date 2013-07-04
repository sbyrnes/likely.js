likely.js
=========
A javascript library for collaborative filtering and recommendation engines designed for node.js

Usage 
---------
    
    Recommender = require('likely.js');
    
    // STEP 1. Assemble the input
    // Input matrix where rows are input entities (such as users) and the columns are items
    //    A cell represents the rating of that item by the entity. inputMatrix[0][1] is the 
    //    rating of item 1 by user 0
    var inputMatrix = [ [ 1, 2, 3, 0 ],
                        [ 4, 0, 5, 6 ],
                        [ 7, 8, 0, 9 ]
                      ];
                      
    // Labels to provide more context to the input. Row 0 of the input matrix corresponds to label rowLabels[0]
    var rowLabels = ['John', 'Sue', 'Joe'];
    var colLabels = ['Red', 'Blue', 'Green', 'Purple'];
    // Using these values, John rates Red 1 while Joe rates Red 7. Sue has no rating for Blue.

	// STEP 2. Train the model
	// Using the inputMatrix you build a model, which estimates the ratings for all entities for all users. 
	var Model = Recommender.buildModel(inputMatrix, rowLabels, colLabels);
    
    // or, if you don't have or care about labels
    
    var Model = Recommender.buildModel(inputMatrix);
    
    // Prediction is now a matrix of the same size as inputMatrix but with estimates ratings for
    // all items for all entities. 
    
    // STEP 3. Extract recommendations
	// Retrieve a list of all items not already rated by a user, sorted by estimated ratings
	var recommendations = model.recommendations('John');
	
	// recommendations = [['Purple', 1.34]];

	// Retrieve a list of all items, sorted by the ratings for a given user (both estimated and actual)
	var allItems = model.rankAllItems('John');
	
	// allItems = [['Green', 3.00], ['Blue', 2.00], ['Purple', 1.34], ['Red', 1.00]];

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

Likely.js uses Stochastic Gradient Descent to estimate P and Q which, while simple, can be slow for very large input matrices.
This is the most basic but effective method of recommendations that was highlighted in the Netflix prize: 
http://www2.research.att.com/~volinsky/papers/ieeecomputer.pdf 
