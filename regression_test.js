/**
 * Regression test for the Likely.js recommendation library.
 */
var sylvester = require('sylvester');
var Recommender = require('./likely.js');

// TODO: lower the maximum error threshold
// Note that this is the squared error
var MAXIMUM_ERROR = 1;

var input = $M([
				[1, 0, 3, 1, 0],
				[2, 3, 0, 0, 5],
				[3, 1, 3, 4, 1],
				[0, 1, 1, 1, 1]
				]);
				
console.log('Input Matrix: ');
prettyPrint(input);

console.log('\nEstimated Matrix: ');
var model = Recommender.buildModel(input);
var estimate = model.estimated;
prettyPrint(estimate);

console.log('\nError Matrix: ');
var errorMatrix = Recommender.calculateError(estimate, input);
prettyPrint(errorMatrix);

console.log('\nTotal Error: ');
var error = Recommender.calculateTotalError(errorMatrix);
console.log(error);

console.log('\n');

if(error > MAXIMUM_ERROR)
	console.log('FAIL - Error too large');
else 
	console.log('SUCCESS');
	
console.log('\n\n');

/** Utility to pretty print matrix contents */
function prettyPrint(matrix)
{	
	console.log(matrix.inspect());
}