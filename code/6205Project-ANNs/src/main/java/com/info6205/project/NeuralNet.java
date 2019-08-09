package com.info6205.project;


public class NeuralNet {
	//network structure
	int inputNeural;
	int hiddenNeural;
	int outputNeural;
	//weights and bias 
	double[][] inputWeights, hiddenWeights;
	double[] hiddenBias, outputBias;
	//to save the result calculate by activation function
	double[] hiddenOutput;
	double[] finalOutput;
	//these two matrix is used for write down the differences according to the error --> (a-y)
	double[] outputError;
	double[] hiddenError;
	
	public NeuralNet(int[]sizes) {
		NeuralNet(sizes,"sigmo");
	}
	public NeuralNet(int[]sizes,String input) {
		NeuralNet(sizes,input);
	}
	Activation act;
	//build a three layers network, and generate random weight and bias
	private void NeuralNet(int[] sizes,String s) {
		if(s.equalsIgnoreCase("sigmoid")) {
			act=new Sigmoid();
		}else if(s.equalsIgnoreCase("tanh")) {
			act=new Tanh();
		}
		else {
			act=new Sigmoid();
		}
		//int layers = sizes.length;
		inputNeural = sizes[0];
		hiddenNeural = sizes[1];
		outputNeural = sizes[2];
		
		//
		hiddenOutput =  new double[hiddenNeural];
		finalOutput = new double[outputNeural];
		
		//generate random weights for each connected neural 
		inputWeights = new double[inputNeural][hiddenNeural];
		hiddenWeights = new double[hiddenNeural][outputNeural];
		for(int i = 0; i < hiddenNeural; i++) {
			for(int j = 0; j < inputNeural; j++) {
				inputWeights[j][i]= (2 * Math.random()) - 1;
			}
			for(int k = 0; k < outputNeural; k++) {
				hiddenWeights[i][k] = (2 * Math.random()) - 1;
			}
		}
		
		//generate random bias for each hidden neural and outputNeural
		hiddenBias = new double[hiddenNeural];
		for(int m = 0; m < hiddenNeural ; m++) hiddenBias[m] = (2 * Math.random()) - 1;
		outputBias = new double[outputNeural];
		for(int n = 0; n < outputNeural; n++) outputBias[n] = (2 * Math.random()) - 1;
	}
	
	
    public void train(TrainSet set, int loops, int groupSize) {
        if(set.inputSize != inputNeural || set.outputSize != outputNeural) return;
        for(int i = 0; i < loops; i++) {
            TrainSet group = set.getRandomData(groupSize);
            for(int b = 0; b < groupSize; b++) {
                this.train(group.getInput(b), group.getOutput(b), 0.8);          
            }
            System.out.println(groupAVGMSE(group));
        }
    }
    
    //calculate the cost function 
    public double MSE(double[] input, double[] expected) {
        calHiddenOutPut(input);
		calFinalOutput();
        double v = 0;
        for(int i = 0; i < expected.length; i++) {
            v += (expected[i] - finalOutput[i]) * (expected[i] - finalOutput[i]);
        }
        return v / (2d * expected.length);
    }

    public double groupAVGMSE(TrainSet set) {
        double v = 0;
        for(int i = 0; i< set.size(); i++) {
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }
    
//    public double AVGvariance(TrainSet set) {
//    		double v = 0;
//    		for(int i = 0; i < set.size(); i++) {
//    			v += variance(set.getInput(i), set.getOutput(i));
//    		}
//    		return (v / set.size());
//    }
//    
//    public double variance(double[] input, double[] output) {
//		calHiddenOutPut(input);
//		calFinalOutput();
//		double v = 0;
//		for(int i = 0; i< outputNeural; i++) {
//			v+= (output[i]- finalOutput[i]) * (output[i]- finalOutput[i]) ;
//		}
//		return (v / output.length);
//    }
    
	public void train(double[] input, double[] expected, double learningRate) {
		double n = learningRate;
		//feed forward
		calHiddenOutPut(input);
		calFinalOutput();
		//back-propagation
		deltaOutputLayer(expected);
		deltaHiddenLayer();
		//adjust
		updateOutputLayer(n);
		updateHiddenLayer(input,n);
	}
	
	//after the training process, the code should guess the result correctly
	public void guess(double[] input) {
		calHiddenOutPut(input);
		calFinalOutput();
	}
	
	//calculate the output(activation) for the second layer(hidden layer)
	public void calHiddenOutPut(double[] input) {
		for(int i = 0; i< hiddenNeural; i++) {
			double sum = 0;
			for(int j = 0; j< inputNeural; j++) {
				sum += inputWeights[j][i] * input[j];
			}
			sum = sum + hiddenBias[i];
			hiddenOutput[i] = act.callActivation(sum);
		}
	}
	
	//calculate the output(activation) for the output layer
	public double[] calFinalOutput() {
		for(int i = 0; i < outputNeural; i++) {
			double sum = 0;
			for(int j = 0; j < hiddenNeural; j++) {
				sum += hiddenWeights[j][i] * hiddenOutput[j];
			}
			sum = sum + outputBias[i];
			finalOutput[i] = act.callActivation(sum);
			
		}
		return finalOutput;
		
	}
	
	public double[] feedForward(double[] input) {
		calHiddenOutPut(input);
		return calFinalOutput();
	}
	
	//compute the error of the output layer and get the delta value
	public void deltaOutputLayer(double expected[]) {
		outputError = new double[outputNeural];
		for(int j = 0; j < outputNeural; j++) {
			//the way to calculate error δL=(aL−y)⊙σ′(zL)
			outputError[j] = (expected[j] - finalOutput[j]) * act.derivatives(finalOutput[j]);	
		}
	}
	
	//we calculate the error of the output layer, then we can adjust the weights and bias to make things right
	public void updateOutputLayer(double n) {
		//update weights from hidden to output
		for(int i = 0; i < hiddenNeural; i++) {
			for(int j = 0; j < outputNeural; j++) {
				hiddenWeights[i][j] += (outputError[j] * hiddenOutput[i] * n);
				//ystem.out.println(hiddenWeights[i][j]);
			}
		}
		//update bias form output layer
		for(int m = 0; m < outputNeural; m++) {
			outputBias[m] += (outputError[m] * (n/10));
		}			
	}
	
	//compute the error of the hidden layer and get the delta value
	public void deltaHiddenLayer() {
		hiddenError = new double[hiddenNeural];
		//we have to know the output layer error to compute the hidden layer error
		for(int i = 0; i < hiddenNeural; i++) {
			for(int j= 0; j < outputNeural; j++) {
				hiddenError[i] += (hiddenWeights[i][j] * outputError[j]);
			}
			hiddenError[i] *= act.derivatives(hiddenOutput[i]);
		}
	}
	
	//after we know the error of the hidden layer, we can adjust the input weights and the bias
	public void updateHiddenLayer(double[] input, double n) {
		//update weight form input to hidden
		for(int i = 0; i < inputNeural; i++) {
			for(int j = 0; j< hiddenNeural; j++) {
				inputWeights[i][j] += (hiddenError[j] * input[i] * n);
			}
			//update the bias in the hidden layer
			for(int m = 0; m< hiddenNeural; m++){
				hiddenBias[m] += (hiddenError[m] * (n/10));
			}
		}
	}
	
	//this method is write for the test, to generate identical weight
	public void generateTestWeights() {
		for(int i = 0; i < hiddenNeural; i++) {
			for(int j = 0; j < inputNeural; j++) {
				inputWeights[j][i]= 1;
			}
			for(int k = 0; k < outputNeural; k++) {
				hiddenWeights[i][k] = 1;
			}
		}
	}
	//this method is write for the test, to generate identical weight
	public void generateTestBias() {
		hiddenBias = new double[hiddenNeural];
		for(int m = 0; m < hiddenNeural ; m++) hiddenBias[m] = 1;
		outputBias = new double[outputNeural];
		for(int n = 0; n < outputNeural; n++) outputBias[n] = 1;
	}
	
//	//activation function
//	public double sigmoid(double x) {
//			return  (1 / (1 + Math.exp(-x)));
//	}
//	
//	public double derivativesOfSigmoid(double activation) {
//		double a = activation;
//		return a * (1-a);
//	}
	
}
