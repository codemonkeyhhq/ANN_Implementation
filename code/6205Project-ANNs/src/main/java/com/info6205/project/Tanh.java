package com.info6205.project;

public class Tanh implements Activation{
	//tanh is also like logistic sigmoid but better. 
	//The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s - shaped).
	//reference:https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
	
		public double callActivation(double x) {
			// TODO Auto-generated method stub
			return Math.tanh(x);
		}


		public double derivatives(double x) {
			// TODO Auto-generated method stub
			double a=x;
			return 1-a*a;
		}

}