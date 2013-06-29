likely.js
=========

***NOTE: Likely.js is a work in progress. You probably already figured that out from the fact that it is just a README right now 
but it will be more soon!***

A javascript library for collaborative filtering and recommendation engines designed for node.js

Usage 
---------
    
    Recommender = require('likely.js');

    Recommender.train(trainingMatrix);
    var recommendedList = Recommender.recommend(inputVector);

Description
---------
Likely.js is a library of utilities used for collaborative filtering and recommendation engines. It takes an input
training matrix comprised of the object IDs and values for the set of features chosen. After training, you provide
an input vector comprised of various values to produce a list of recommended object IDs. 

For example, you may provide as input a training matrix where each row is a customer and the columns are the ratings they give 
to various movies. Then you might submit a single users rating history as a vector to produce a list of movies they might like.

