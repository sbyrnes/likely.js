/**
 * Tests for the Likely.js recommendation library.
 */
var sylvester = require('sylvester');
var Recommender = require('../likely.js');

exports['test generateRandomMatrix#size'] = function(beforeExit, assert){
	var M1 = Recommender.generateRandomMatrix(4,3);

    assert.equal(4, M1.rows());
    assert.equal(3, M1.cols());
    
	var M2 = Recommender.generateRandomMatrix(1,1);

    assert.equal(1, M2.rows());
    assert.equal(1, M2.cols());
};

exports['test generateRandomMatrix#random'] = function(beforeExit, assert){
	var M1 = Recommender.generateRandomMatrix(4,3);
	var M2 = Recommender.generateRandomMatrix(4,3);
	
	assert.equal(false, M1.eql(M2));
};

exports['test calculateError#values'] = function(beforeExit, assert){
	var P = $M([[2, 2],[2, 2],[2, 2]]);
	var Q = $M([[2, 2, 2],[2, 2, 2]]);
	var input = $M([[8, 8, 8],[8, 8, 8],[8, 8, 8]]);
	
	var error = Recommender.calculateError(P, Q, input);
	
	var expectedError = sylvester.Matrix.Zeros(3, 3);
	
	assert.equal(true, expectedError.eql(error));
};
