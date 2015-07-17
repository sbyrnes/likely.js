/**
 * Tests for the Likely.js recommendation library.
 */
var sylvester = require('sylvester');
var Recommender = require('../likely.js');

// Test random matrix generation
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

// Test the calculation of error values
exports['test calculateError#values'] = function(beforeExit, assert){
	var P = $M([[2, 2],[2, 2],[2, 2]]);
	var Q = $M([[2, 2, 2],[2, 2, 2]]);
	var input = $M([[8, 8, 8],[8, 8, 8],[8, 8, 8]]);
	
	var error = Recommender.calculateError(P.x(Q), input);
	
	var expectedError = sylvester.Matrix.Zeros(3, 3);
	
	assert.equal(true, expectedError.eql(error));
};

// Test the calculation of total error 
exports['test calculateTotalError#values'] = function(beforeExit, assert){
	var error = sylvester.Matrix.Zeros(3, 3);
	
	assert.equal(0, Recommender.calculateTotalError(error));

	error = $M([[1, -2, 0.5],[1, 0, -2]]);
	
	assert.equal(10.25, Recommender.calculateTotalError(error));
};

// Test calculating matrix average
exports['test calculateAverage#matrix'] = function(beforeExit, assert){
	var input = $M([[2, 2],[2, 2],[2, 2]]);
	
	assert.equal(2, Recommender.calculateMatrixAverage(input));
	
	input = $M([ [ 1, 2, 3, 0 ],
			     [ 4, 0, 5, 6 ],
			     [ 7, 8, 0, 9 ]
		       ]);
	
	assert.equal(3.75, Recommender.calculateMatrixAverage(input));
};

// Test calculating row average
exports['test calculateAverage#row'] = function(beforeExit, assert){
	var input = $M([[2, 2],[2, 2],[2, 2]]);
	
	var result = Recommender.calculateRowAverage(input);
	assert.equal(3, result.dimensions().cols);
	assert.equal(2, result.e(1));
	assert.equal(2, result.e(2));
	assert.equal(2, result.e(2));
	
	input = $M([ [ 1, 2, 3, 0 ],
			     [ 4, 0, 5, 6 ],
			     [ 7, 8, 0, 9 ]
		       ]);
	result = Recommender.calculateRowAverage(input);
	assert.equal(3, result.dimensions().cols);
	assert.equal(1.5, result.e(1));
	assert.equal(3.75, result.e(2));
	assert.equal(6, result.e(3));
};

// Test calculating column average
exports['test calculateAverage#column'] = function(beforeExit, assert){
	var input = $M([[2, 2],[2, 2],[2, 2]]);
	var result = Recommender.calculateColumnAverage(input);
	assert.equal(2, result.dimensions().cols);
	assert.equal(2, result.e(1));
	assert.equal(2, result.e(2));
	
	input = $M([ [ 1, 2, 3, 0 ],
			     [ 4, 0, 5, 6 ],
			     [ 7, 8, 0, 9 ]
		       ]);
	result = Recommender.calculateColumnAverage(input);
	assert.equal(4, result.dimensions().cols);
	assert.equal(4, result.e(1));
	assert.equal(3.33, result.e(2).toFixed(2));
	assert.equal(2.67, result.e(3).toFixed(2));
	assert.equal(5, result.e(4));
};

// Test calculating matrix bias
exports['test calculateBias#matrix'] = function(beforeExit, assert){
	var input = [[2, 2],[2, 2],[2, 2]];
	
	var bias = Recommender.calculateBias(input)
	assert.equal(2, bias.average);
	assert.equal(3, bias.rowBiases.dimensions().cols);
	assert.equal(0, bias.rowBiases.e(1));
	assert.equal(0, bias.rowBiases.e(2));
	assert.equal(0, bias.rowBiases.e(3));
	assert.equal(2, bias.colBiases.dimensions().cols);
	assert.equal(0, bias.colBiases.e(1));
	assert.equal(0, bias.colBiases.e(2));
	
	input = [ [ 1, 2, 3, 0 ],
			  [ 4, 0, 5, 6 ],
			  [ 7, 8, 0, 9 ]
		    ];
	bias = Recommender.calculateBias(input)
	assert.equal(3.75, bias.average);
	assert.equal(3, bias.rowBiases.dimensions().cols);
	assert.equal(-2.25, bias.rowBiases.e(1));
	assert.equal(0, bias.rowBiases.e(2));
	assert.equal(2.25, bias.rowBiases.e(3));
	assert.equal(4, bias.colBiases.dimensions().cols);
	assert.equal(0.25, bias.colBiases.e(1));
	assert.equal(-0.42, bias.colBiases.e(2).toFixed(2));
	assert.equal(-1.08, bias.colBiases.e(3).toFixed(2));
	assert.equal(1.25, bias.colBiases.e(4));
};

// Test the Model object's ability to return all items sorted by rating, with labels provided
exports['test Model#rankAllItems|withLabels'] = function(beforeExit, assert){
    var rowLabels = ['John', 'Sue', 'Joe'];
    var colLabels = ['Red', 'Blue', 'Green', 'Purple'];
    var inputMatrix = [ [ 1, 2, 3, 0 ],
                        [ 2, 0, 5, 6 ],
                        [ 7, 8, 0, 9 ]
                      ];
                      	
    var model = Recommender.buildModel($M(inputMatrix), rowLabels, colLabels);
    
    var sueArray = model.rankAllItems('Sue');
    
    assert.equal(4, sueArray.length);     
    assert.equal('Purple', sueArray[0][0]);  
    assert.equal('Green', sueArray[1][0]);       
    assert.equal('Blue', sueArray[2][0]);       
    assert.equal('Red', sueArray[3][0]);            
};

// Test the Model object's ability to return all items sorted by estimated rating, without labels 
exports['test Model#rankAllItems|withoutLabels'] = function(beforeExit, assert){
    var inputMatrix = [ [ 1, 2, 1, 0 ],
                        [ 2, 0, 1, 6 ],
                        [ 7, 8, 0, 9 ]
                      ];
                      	
    var model = Recommender.buildModel($M(inputMatrix));
    
    var rowTwoArray = model.rankAllItems(2);
    
    assert.equal(4, rowTwoArray.length);     
    assert.equal(3, rowTwoArray[0][0]);  
    assert.equal(1, rowTwoArray[1][0]);       
    assert.equal(0, rowTwoArray[2][0]);       
    assert.equal(2, rowTwoArray[3][0]);            
};

// Test the Model object's ability to return only recommended items with labels provided
exports['test Model#recommendations|withLabels'] = function(beforeExit, assert){
    var rowLabels = ['John', 'Sue', 'Joe'];
    var colLabels = ['Red', 'Blue', 'Green', 'Purple', 'Brown', 'Black', 'White', 'Gray'];
    var inputMatrix = [ [ 1, 9, 3, 0, 9, 5, 0, 1 ],
                        [ 4, 0, 5, 6, 9, 1, 0, 0 ],
                        [ 7, 4, 0, 9, 0, 2, 0, 2 ]
                      ];
                      	
    var bias = Recommender.calculateBias(inputMatrix);
    
    var model = Recommender.buildModelWithBias(inputMatrix, bias, rowLabels, colLabels);
    
    var rowTwoArray = model.recommendations('Joe');

    assert.equal(3, rowTwoArray.length);     
    assert.equal('Brown', rowTwoArray[0][0]);  
    assert.equal('Green', rowTwoArray[1][0]);       
    assert.equal('White', rowTwoArray[2][0]);             
};


// Test the Model object's ability to return only recommended items without labels provided
exports['test Model#recommendations|withoutLabels'] = function(beforeExit, assert){
    var inputMatrix = [ [ 1, 2, 3, 0, 2, 5, 1, 3 ],
                        [ 4, 0, 5, 6, 3, 1, 0, 0 ],
                        [ 7, 8, 0, 9, 0, 2, 0, 5 ]
                      ];
                      	
    var model = Recommender.buildModel($M(inputMatrix));
    
    var rowTwoArray = model.recommendations(1);
    
    assert.equal(3, rowTwoArray.length);    
    assert.equal(1, rowTwoArray[0][0]);  
    assert.equal(7, rowTwoArray[1][0]);       
    assert.equal(6, rowTwoArray[2][0]);             
};



// Test the Model object's ability to return all items sorted by estimated rating, without labels 
exports['test Model#rankAllItems|withBias|withoutLabels'] = function(beforeExit, assert){
    var inputMatrix = [ [ 1, 2, 1, 0 ],
                        [ 4, 0, 5, 6 ],
                        [ 5, 8, 0, 9 ]
                      ];  
                      	
    var bias = Recommender.calculateBias(inputMatrix);
    var model = Recommender.buildModelWithBias(inputMatrix, bias);
    
    var rowTwoArray = model.rankAllItems(2);
    
    assert.equal(4, rowTwoArray.length);     
    assert.equal(3, rowTwoArray[0][0]);  
    assert.equal(1, rowTwoArray[1][0]);       
    assert.equal(2, rowTwoArray[2][0]);       
    assert.equal(0, rowTwoArray[3][0]);            
};

// Test setting constants
exports['test Constants#values'] = function(beforeExit, assert){
	assert.equal(5000, Recommender.DESCENT_STEPS);
	
	Recommender.DESCENT_STEPS = 5500;
	
	assert.equal(5500, Recommender.DESCENT_STEPS);
	
	Recommender.DESCENT_STEPS = 5000;
	
	assert.isDefined(Recommender.DESCENT_STEPS);
	assert.isDefined(Recommender.ALPHA);
	assert.isDefined(Recommender.BETA);
	assert.isDefined(Recommender.K);
	assert.isDefined(Recommender.MAX_ERROR);
}

