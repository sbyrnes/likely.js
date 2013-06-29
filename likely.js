/**
 * Likely.js 
 * 
 * Javascript library for collaborative filtering / recommendation engine. 
 */
 
DESCENT_STEPS = 2000; // number of iterations to execute gradient descent. 
ALPHA = 0.0005;       // learning rate
BETA = 0.0007;        // regularization factor
 
train(inputMatrix)
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
    
  }
}

generateRandomArray(rows, columns)
{

}

getRecommendations(index)
{

}
 
 
