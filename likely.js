/**
 * Likely.js 
 * 
 * Javascript library for collaborative filtering / recommendation engine. 
 */
 
DESCENT_STEPS = 2000; // number of iterations to execute gradient descent 
ALPHA = 0.0005;       // learning rate, should be small
BETA = 0.0007;        // regularization factor, should be small
 
module.exports.train(inputMatrix)
{
  // Generate random P and Q based on the dimensions of inputMatrix
  N = inputMatrix.length;    // number of rows 
  M = inputMatrix[0].length; // number of columns
  k = 2;                     // number of features to simulate
  
  P = generateRandomArray(N, k);
  Q = generateRandomArray(M, k);
  
  for(i = 0; i < DESCENT_STEPS; i++)
  {
    // determine error
    var error = calculateError(P, Q, inputMatrix);
    
    // update P and Q accordingly
    P_prime = update(P, error);
    Q_prime = update(Q, error);
    
    P = P_prime;
    Q = Q_prime;
  }
  
  // produce the final estimation by multiplying P and Q
  var estimatedArray = P x Q; 
}

// Generates a random Matrix of size rows x columns
module.exports.generateRandomArray(rows, columns)
{

}

// For a given entity index, returns all item ids that had no rating for that item. 
//     Results are sorted in order of descenting estimated value.
getRecommendations(index)
{
  
}
 
 
