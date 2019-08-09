package com.info6205.project;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class NeuralNetTest {
	//initialize how many neural in specific layer 
	int[] net = {3,3,2};
	//give the input of the input layer
	double[] input = {0.3,0.4,0.5};
	//give the result that we expected to compute error
	double[] expected = {0.99, 0.99};
	
	@Test
	public void feedForward0() {
		NeuralNet nt = new NeuralNet(net);
		nt.generateTestWeights();
		nt.generateTestBias();
		nt.calHiddenOutPut(input);
		assertEquals(0.90024, nt.hiddenOutput[0], 1.0E-5);
		assertEquals(0.90024, nt.hiddenOutput[1], 1.0E-5);
	}
	
	@Test
	public void feedForward1() {
		NeuralNet nt = new NeuralNet(net);
		nt.generateTestWeights();
		nt.generateTestBias();
		nt.calHiddenOutPut(input);
		nt.calFinalOutput();
		assertEquals(0.9758, nt.finalOutput[0], 1.0E-4);
		assertEquals(0.9758, nt.finalOutput[1], 1.0E-4);
	}
	
	@Test
	public void backPropagation0() {
		NeuralNet nt = new NeuralNet(net);
		nt.generateTestWeights();
		nt.generateTestBias();
		nt.calHiddenOutPut(input);
		nt.calFinalOutput();
		nt.deltaOutputLayer(expected);
		assertEquals(0.00033, nt.outputError[0], 1.0E-5);
		assertEquals(0.00033, nt.outputError[1], 1.0E-5);
	}
	
	@Test
	public void backPropagation1() {
		NeuralNet nt = new NeuralNet(net);
		nt.generateTestWeights();
		nt.generateTestBias();
		nt.calHiddenOutPut(input);
		nt.calFinalOutput();
		nt.deltaOutputLayer(expected);
		nt.deltaHiddenLayer();
		assertEquals(5.96E-5, nt.hiddenError[0], 1.0E-5);
		assertEquals(5.96E-5, nt.hiddenError[1], 1.0E-5);
	}

}
