/**
 * Likely.js 
 * 
 * Javascript library for collaborative filtering / recommendation engine. 
 */
var sylvester = require('sylvester');
 
// Learning parameters
var DESCENT_STEPS = 2000; // number of iterations to execute gradient descent 
var ALPHA = 0.0005;       // learning rate, should be small
var BETA = 0.0007;        // regularization factor, should be small
var k = 2; 				  // number of features to simulate

// Model storage
// TODO: make this stateless
var P_model;
var Q_model;
var estimatedResult;

// Trains the model on the given input by composing two matrices P and Q which
// approximate the input through their product. 
module.exports.train = function(inputMatrix)
{
  // Generate random P and Q based on the dimensions of inputMatrix
  N = inputMatrix.length;    // number of rows 
  M = inputMatrix[0].length; // number of columns
  
  P_model = generateRandomMatrix(N, k);
  Q_model = generateRandomMatrix(M, k);
  
  for(i = 0; i < DESCENT_STEPS; i++)
  {
    // determine error
    var error = calculateError(P_model, Q_model, inputMatrix);
    
    // update P and Q accordingly
    P_prime = updateP(P, Q, error);
    Q_prime = updateQ(Q, P, error);
    
    P_model = P_prime;
    Q_model = Q_prime;
  }
  
  // produce the final estimation by multiplying P and Q
  estimatedResult = P_model.x(Q_model); 
}

// Generates a random Matrix of size rows x columns
// TODO: Adjust the range of generated random values
module.exports.generateRandomMatrix = function(rows, columns)
{
	return sylvester.Matrix.Random(rows, columns);
}

// Computes the error from model matrices P and Q against the given input. 
// Result is a matrix of size input.rows by input.columns
module.exports.calculateError = function(P, Q, input)
{
	// calculate R' as the result of P x Q
	var R_prime = P.x(Q); 
	
	// Error is (R - R')
	return input.subtract(R_prime);
}

// Computes the total error based on a matrix of error values
module.exports.calculateTotalError(error)
{
	var totError = 0;
	for(int i = 0; i < error.rows(); i++)
	{
		for(int j = 0; j < error.columns(); j++)
		{
			totError += pow(error.e(i, j), 2);
		}
	}
	
	return totError;
}

function updateP(P, Q, error)
{
	// Start with original
	var P_prime = P.dup();
	
	// Gradient descent factor Alpha*2*Error*Q
	
	P_prime.subtract(Error.x(Q).x(2 * Alpha));
	
	// Regularization, subtract Alpha*Beta*P
	P_prime.subtract(P.x(ALPHA * BETA));
	
	return P_prime;
}

function updateQ(Q, P, error)
{
	
}

// For a given entity index, returns all item ids that had no rating for that item. 
//     Results are sorted in order of descenting estimated value.
module.exports.getRecommendations = function(index)
{
  
}
 
 
