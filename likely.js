/**
 * Likely.js 
 * 
 * Javascript library for collaborative filtering / recommendation engine. 
 */
var sylvester = require('sylvester');
 
// Learning parameters
var DESCENT_STEPS = 5000; // number of iterations to execute gradient descent 
var ALPHA = 0.0005;       // learning rate, should be small
var BETA = 0.0007;        // regularization factor, should be small
var k = 5; 				  // number of features to simulate
var MAX_ERROR = 0.0005;	  // threshold which, if reached, will stop descent automatically

// Builds a complete model from the input array 
function buildModel(inputArray, rowLabels, colLabels)
{
	var model = new Model($M(inputArray), rowLabels, colLabels);
	model.estimated = train(sylvester.Matrix.create(inputArray));
	
	return model
}

// Trains the model on the given input by composing two matrices P and Q which
// approximate the input through their product. 
// RETURNS: A model entity, with estimated values based on the input
function train(inputMatrix)
{
  // Generate random P and Q based on the dimensions of inputMatrix
  N = inputMatrix.rows();    // number of rows 
  M = inputMatrix.cols(); // number of columns
  
  var P_model = generateRandomMatrix(N, k);
  var Q_model = generateRandomMatrix(k, M);
  
  for(var i = 0; i < DESCENT_STEPS; i++)
  {
  	//console.log('------------------ Iteration --------------------');
    // determine error
    var error = calculateError(P_model.x(Q_model), inputMatrix);

	P_prime = P_model.elements;
	Q_prime = Q_model.elements;
	
	//console.log('P: ' + JSON.stringify(P_prime));
	//console.log('Q: ' + JSON.stringify(Q_prime));
	
	// update P and Q accordingly
    for (var row = 0; row < N; row++)
    {
    	for (var col = 0; col < M; col++)
    	{
    		for(var feature = 0; feature < k; feature++)
    		{
    			// update formulas will change values in the opposite direction of the gradient.
    			
    			// P Update Formula
    			// p_ik = p_ik + alpha * (e_ij * q_kj - beta * p_ik)
    			// Reverse Gradient: alpha * e_ij * q_kj   -- Note that we omit the 2* factor since it's not necessary for convergence.
    			// Regularization factor: alpha * beta * p_ik 
    			var p_prev = P_prime[row][feature];
    			P_prime[row][feature] = P_prime[row][feature] + 
    									  ALPHA*(error.e(row+1, col+1)*Q_prime[feature][col] -
    									  	    BETA * P_prime[row][feature]);
    			//console.log('P['+row+']['+feature+'] ('+p_prev+') <- ('+P_prime[row][feature]+')');
    									  	   
    			// Q Update Formula
    			// q_kj = q_kj + alpha x (e_ij x p_ik - beta x q_kj)
    			// Reverse Gradient: alpha * e_ij * p_ik   -- Note that we omit the 2* factor since it's not necessary for convergence.
    			// Regularization factor: alpha * beta * q_kj  
    			var q_prev = Q_prime[feature][col];
  				Q_prime[feature][col] = Q_prime[feature][col] +
  											 ALPHA *(error.e(row+1, col+1)*P_prime[row][feature] -
  											 		BETA * Q_prime[feature][col]);  	
    			//console.log('Q['+feature+']['+col+'] ('+q_prev+') <- ('+Q_prime[feature][col]+')');								  	    
    									  	
            }
    	}
    }
    
    // if we've already reached the error threshold, no need to descend further
    var totError = calculateTotalError(error.dup());
    //console.log('total error: ' + totError);
    if(totError < MAX_ERROR)
    {
    	console.log('short circuiting');
	    break;
    }
  }
  
  // produce the final estimation by multiplying P and Q
  return P_model.x(Q_model); 
}

// Generates a random Matrix of size rows x columns
// TODO: Adjust the range of generated random values
function generateRandomMatrix(rows, columns)
{
	return sylvester.Matrix.Random(rows, columns);
}

// Computes the error from model matrices P and Q against the given input. 
// Result is a matrix of size input.rows by input.columns
function calculateError(estimated, input)
{ 	
	// Error is (R - R')
	return input.subtract(estimated);
}

// Computes the total error based on a matrix of error values
function calculateTotalError(estimated, input)
{
	var errorMatrix = calculateError(estimated, input);
	var totError = 0.0;
	for(var i = 1; i <= errorMatrix.rows(); i++)
	{
		for(var j = 1; j <= errorMatrix.cols(); j++)
		{
			totError += Math.pow(errorMatrix.e(i, j), 2);
		}
	}
	
	return totError;
}

// Model representation object
function Model(inputMatrix, rowLabels, colLabels) {
	this.rowLabels = rowLabels;	// labels for the rows
	this.colLabels = colLabels; // labels for the columns
	this.input = inputMatrix;	// input data
	this.estimated = sylvester.Matrix.Zeros(this.input.rows(),this.input.cols());
}
Model.prototype = {
	// Returns all items for a given row, sorted by rating.
	rankAllItems: function(row)
	{
		var rowIndex = row; // assume row is a number
		// If we're using labels we have to look up the row index
		if(this.rowLabels)
		{
			rowIndex = findInArray(this.rowLabels, row);
		}
		
		// estimates for this user
		var ratingElements = this.estimated.row(rowIndex+1).elements;
		
		// build a two dimensional array from the ratings and indexes
		//     [[index, rating], [index, rating]]
		var outputArray = new Array();
		for(var i=0; i<ratingElements.length; i++)
		{
			outputArray[i] = [i, ratingElements[i]];
			
			// if we have column labels, use those
			if(this.colLabels)
			{
				outputArray[i][0] = this.colLabels[i];
			}
		}
		
		// Sort the array by index
		return outputArray.sort(function(a, b) {return a[1] < b[1]})
	},
	
	// Returns all items for the given row where there was no input value, sorted by rating.
	recommendations: function(row)
	{
		var recommendedItems = new Array();
		var allItems = this.rankAllItems(row);
			
		var rowIndex = row; // assume row is a number
		// If we're using labels we have to look up the row index
		if(this.rowLabels)
		{
			rowIndex = findInArray(this.rowLabels, row);
		}
		
		for(var i=0; i< allItems.length; i++)
		{
			// look up the value in the input
			var colIndex = allItems[i][0];
			// see if we're using column labels or not
			if(this.colLabels) 
			{
				colIndex = findInArray(this.colLabels, allItems[i][0]);
			}
			
			var inputRating = this.input.e(rowIndex+1, colIndex+1);
			
			// if there was no rating its a recommendation so add it
			if(inputRating == 0)
			{
				recommendedItems.push(allItems[i]);
			}
		}
		
		return recommendedItems;
	}	
}

// Finds the specified value in the array and returns the index. Returns -1 if not found.
function findInArray(array, value)
{
	var index = -1;
	for(var i=0;i<array.length;i++)
	{
		if(array[i] == value) 
		{
			index = i;
			break;
		}
	}
	
	return index;
}

module.exports.train = train;
module.exports.generateRandomMatrix = generateRandomMatrix;
module.exports.calculateError = calculateError;
module.exports.calculateTotalError = calculateTotalError;
module.exports.Model = Model;

