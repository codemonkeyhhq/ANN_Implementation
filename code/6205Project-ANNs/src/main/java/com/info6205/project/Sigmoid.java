package com.info6205.project;

public class Sigmoid implements Activation{
	//The main reason why we use sigmoid function is because it exists between (0 to 1).
	//Therefore, it is especially used for models where we have to predict the probability as an output.
	//reference:https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
		public double callActivation(double x) {	
			return (1 / (1 + Math.exp(-x)));
		}
		public double derivatives(double x) {
			double a = x;
			return a * (1-a);
		}
}