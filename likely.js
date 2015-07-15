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
var K = 5; 				  // number of features to simulate
var MAX_ERROR = 0.0005;	  // threshold which, if reached, will stop descent automatically

/** Builds a complete model from the input array 
 *
 * @param inputArray A two dimensional array of the input, where each cell is the rating given to Col by Row
 * @param [optional] rowLabels A one dimensional array of string labels for each row of inputArray
 * @param [optional] colLabels A one dimensional array of string labels for each column of inputArray
 * @returns Model An instance of a Model object
 */
function buildModel(inputArray, rowLabels, colLabels)
{
	return buildModelWithBias(inputArray, undefined, rowLabels, colLabels);
}

/** Builds a complete model from the input array with bias
 *
 * @param inputArray A two dimensional array of the input, where each cell is the rating given to Col by Row
 * @param bias A two dimensional matrix of the bias of the inputArray
 * @param [optional] rowLabels A one dimensional array of string labels for each row of inputArray
 * @param [optional] colLabels A one dimensional array of string labels for each column of inputArray
 * @returns Model An instance of a Model object
 */
function buildModelWithBias(inputArray, bias, rowLabels, colLabels)
{
	var model = new Model($M(inputArray), rowLabels, colLabels);
	model.estimated = train(sylvester.Matrix.create(inputArray), bias);
	
	return model
}

/** 
 * Trains the model on the given input by composing two matrices P and Q which
 * approximate the input through their product. 
 * @param inputMatrix A two dimensional array representing input values
 * @returns Model A model entity, with estimated values based on the input
 */
function train(inputMatrix, bias)
{
  N = inputMatrix.rows();    // number of rows 
  M = inputMatrix.cols(); // number of columns
  
  // Generate random P and Q based on the dimensions of inputMatrix
  var P_model = generateRandomMatrix(N, K);
  var Q_model = generateRandomMatrix(K, M);
  
  var i = 0
  for(i = 0; i < DESCENT_STEPS; i++)
  {
  	//console.log('------------------ Iteration --------------------');
    // Calculate error
    var error = calculateError(P_model.x(Q_model), inputMatrix, bias);

	P_prime = P_model.elements;
	Q_prime = Q_model.elements;
	
	// For debugging
	//console.log('P: ' + JSON.stringify(P_prime));
	//console.log('Q: ' + JSON.stringify(Q_prime));
	
	// Update P and Q to reduce error
    for (var row = 0; row < N; row++)
    {
    	for (var col = 0; col < M; col++)
    	{
    		for(var feature = 0; feature < K; feature++)
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
    var totError = calculateTotalError(error);
    if(totError < MAX_ERROR)
    {
    	//console.log('Reached error threshold early, no more descent needed.');
	    break;
    }
  }
  
  //console.log('Descent steps used: ' + i);
  
  // produce the final estimation by multiplying P and Q
  var finalModel = P_model.x(Q_model); 
  
  // if we were considering bias, we have to add it back in
  if(bias)
  {
  		// add back the overall average
  		finalModel = finalModel.map(function(x) { return x + bias.average; });
  	
		var finalElements = finalModel.elements;
		
		// add back the row bias from each row
		for(var i = 1; i <= finalModel.rows(); i++)
		{
			for(var j = 1; j <= finalModel.cols(); j++)
			{
				finalElements[i-1][j-1] += bias.rowBiases.e(i);
			}
		}
		
		// add back the column bias from each column
		for(var i = 1; i <= finalModel.rows(); i++)
		{
			for(var j = 1; j <= finalModel.cols(); j++)
			{
				finalElements[i-1][j-1] += bias.colBiases.e(j);
			}
		}
  }
  
  return finalModel;
}

/**
 * Generates a random Matrix of size rows x columns.
 * @param rows Number of rows in the random matrix
 * @param columns Number of columns in the random matrix
 * @returns Matrix A matrix of size rows x columns filled with random values.
 */
function generateRandomMatrix(rows, columns)
{
	return sylvester.Matrix.Random(rows, columns);
}

/**
 * Computes the error from model matrices P and Q against the given input. 
 * @param estimated A matrix of the estimated values.
 * @param input A matrix of the input values
 * @returns A matrix of size input.rows by input.columns where each entry is the difference between the input and estimated values.
 */
function calculateError(estimated, input, bias)
{ 	
	var adjustedInput = input.dup();
	
	var adjustedElements = adjustedInput.elements;
		
	// If bias adjustment is provided, adjust for it
	if(bias)
	{
		// subtract the row and column bias from each row
		for(var i = 0; i <= adjustedInput.rows()-1; i++)
		{
			for(var j = 0; j <= adjustedInput.cols()-1; j++)
			{
				if(adjustedElements[i][j] == 0) continue; // skip zeroes
				
				adjustedElements[i][j] -= bias.average;
				
				adjustedElements[i][j] -= bias.rowBiases.e(i+1);
				
				adjustedElements[i][j] -= bias.colBiases.e(j+1);
			}
		}
	}
	
	
	var estimatedElements = estimated.elements;
	
	// Error is (R - R')
	// (but we ignore error on the zero entries since they are unknown)
	for(var i = 0; i <= adjustedInput.rows()-1; i++)
	{
		for(var j = 0; j <= adjustedInput.cols()-1; j++)
		{
			if(adjustedElements[i][j] == 0) continue; // skip zeroes
			
			adjustedElements[i][j] -= estimatedElements[i][j];
		}
	}				

	// Error is (R - R')
	return adjustedInput;
}

/**
 * Computes the total error based on a matrix of error values.
 * @param estimated A matrix of the estimated values.
 * @param input A matrix of the input values
 * @returns float Total error, defined as the sum of the squared difference between the estimated and input values. 
 */
function calculateTotalError(estimated, input)
{
	return calculateTotalError(calculateError(estimated, input));
}

/**
 * Computes the total error based on a matrix of error values.
 * @param errorMatrix A matrix of the error values.
 * @returns float Total error, defined as the sum of the squared errors. 
 */
function calculateTotalError(errorMatrix)
{
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

/**
 * Computes the biases from a matrix of values. 
 * @param input A matrix of the input values
 * @returns Bias Object containing the average, bias for rows and bias for columns.
 */
function calculateBias(input)
{
	var inputMatrix = $M(input);
	var average = calculateMatrixAverage(inputMatrix);
	var rowAverages = calculateRowAverage(inputMatrix);
	var colAverages = calculateColumnAverage(inputMatrix);
	
	var rowBiases = new Array();
	var colBiases = new Array();
	
	// The row bias is the difference between the row average and the overall average
	for(var i = 1; i <= rowAverages.dimensions().cols; i++)
	{
		rowBiases[i-1] = rowAverages.e(i) - average;
	}
	
	// the column bias is the difference between the column average and the overall average
	for(var i = 1; i <= colAverages.dimensions().cols; i++)
	{
		colBiases[i-1] = colAverages.e(i) - average;
	}
	
	var biases = new Bias(average, $V(rowBiases), $V(colBiases));
	
	return biases;
}

/**
 * Bias representation object. Contains all bias elements.
 */
function Bias(average, rowBiases, colBiases) {
	this.average = average;		// Overall value average
	this.rowBiases = rowBiases; // Bias for each row
	this.colBiases = colBiases;	// Bias for each column
}

/**
 * Computes the overall average value from a matrix of values. 
 * @param input A matrix of the input values
 * @returns float Average value.
 */
function calculateMatrixAverage(inputMatrix)
{
	var cells = inputMatrix.rows() * inputMatrix.cols();
	
	var sum = 0;
	for(var i = 1; i <= inputMatrix.rows(); i++)
	{
		for(var j = 1; j <= inputMatrix.cols(); j++)
		{
			sum += inputMatrix.e(i, j);
		}
	}
	
	return sum/cells;
}

/**
 * Computes the average value for each column of a matrix of values. 
 * @param input A matrix of the input values
 * @returns Vector Average value for each column. Vector[i] is the average for column i.
 */
function calculateColumnAverage(inputMatrix)
{
	var rows = inputMatrix.rows();
	
	var averages = new Array();
	for(var i = 1; i <= inputMatrix.cols(); i++)
	{
		var sum = 0;
		for(var j = 1; j <= inputMatrix.rows(); j++)
		{
			sum += inputMatrix.e(j, i);
		}
		averages[i-1] = sum/rows;
	}
	
	return $V(averages);
}

/**
 * Computes the average value for each row of a matrix of values. 
 * @param input A matrix of the input values
 * @returns Vector Average value for each row. Vector[i] is the average for row i.
 */
function calculateRowAverage(inputMatrix)
{
	var cols = inputMatrix.cols();
	
	var averages = new Array();
	for(var i = 1; i <= inputMatrix.rows(); i++)
	{
		var sum = 0;
		for(var j = 1; j <= inputMatrix.cols(); j++)
		{
			sum += inputMatrix.e(i, j);
		}
		averages[i-1] = sum/cols;
	}
	
	return $V(averages);
}

/**
 * Model representation object. Contains both input and estimated values.
 */
function Model(inputMatrix, rowLabels, colLabels) {
	this.rowLabels = rowLabels;	// labels for the rows
	this.colLabels = colLabels; // labels for the columns
	this.input = inputMatrix;	// input data
	
	// estimated data, initialized to all zeros
	this.estimated = sylvester.Matrix.Zeros(this.input.rows(),this.input.cols());
}
Model.prototype = {
	/**
	 * Returns all items for a given row, sorted by rating.
	 * @param row The row to return values for
	 * @returns Array An array of arrays, each containing two entries. [0] is the index and [1] is the value, sorted by value.
	 */
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
	
	/**
	 * Returns all items for the given row where there was no input value, sorted by rating.
	 * @param row The row to return values for
	 * @returns Array An array of arrays, each containing two entries. [0] is the index and [1] is the value, sorted by value. 
	 */
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

/**
 * Finds the specified value in the array and returns the index. Returns -1 if not found.
 * @param array The array to look for the value within
 * @param value The value to look for
 * @returns int Index of the value in the array. -1 if not found.
 */
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

module.exports.buildModel = buildModel;
module.exports.buildModelWithBias = buildModelWithBias;
module.exports.generateRandomMatrix = generateRandomMatrix;
module.exports.calculateError = calculateError;
module.exports.calculateTotalError = calculateTotalError;
module.exports.calculateBias = calculateBias;
module.exports.calculateMatrixAverage = calculateMatrixAverage;
module.exports.calculateColumnAverage = calculateColumnAverage;
module.exports.calculateRowAverage = calculateRowAverage;

module.exports.DESCENT_STEPS = DESCENT_STEPS;
module.exports.ALPHA = ALPHA;
module.exports.BETA = BETA;
module.exports.K = K;
module.exports.MAX_ERROR = MAX_ERROR;

module.exports.Bias = Bias;
module.exports.Model = Model;

