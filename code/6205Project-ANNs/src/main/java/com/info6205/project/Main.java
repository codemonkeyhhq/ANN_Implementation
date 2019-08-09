package com.info6205.project;

import java.io.File;

import com.info6205.util.MnistImageFile;
import com.info6205.util.MnistLabelFile;

public class Main {

	public static void main(String[] args) {
		int i = (int) Math.random();
		System.out.print(i);
		int[] netS = {784, 75, 10};
		NeuralNet net = new NeuralNet(netS);
		//net.generateTestBias();
	
		TrainSet ts =createTrainSet(0,10000);
		
		trainData(net, ts, 15, 10, 1000);
		
		TrainSet ts2 = createTrainSet(10001, 11000);
		testData(net, ts2);
		
	}
	

	//create the train set from mnist database
	public static TrainSet createTrainSet(int start, int end) {
		//initialize train set with 28*28 input and 10 output
		TrainSet set = new TrainSet(784, 10);
		try {
			String path = new File("").getAbsolutePath();
			MnistImageFile img = new MnistImageFile(path +"/train-images-idx3-ubyte", "rw");
		    MnistLabelFile label = new MnistLabelFile(path + "/train-labels-idx1-ubyte", "rw");
		    
		    for(int i = start; i <= end; i++) {
		    		if(i % 100 ==  0){
		    			//System.out.println("prepared: " + i);
		    		}
		    		double[] input = new double[28 * 28];
		        double[] output = new double[10];
		        output[label.readLabel()] = 1d;
		        for(int j = 0; j < 28*28; j++){
		        		input[j] = (double)img.read() / (double)256;
		        }
		        set.addData(input, output);
		        img.next();
		        label.next();
		    }
		} catch (Exception e) {
			e.printStackTrace();
		}
		return set;
	}


	public static void trainData(NeuralNet net,TrainSet set, int epochs, int loops, int group_size) {
	    for(int e = 0; e < epochs; e++) {
	        net.train(set, loops, group_size);
	        System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   "+ e+ "   <<<<<<<<<<<<<<<<<<<<<<<<<<");
	    }
	}
	
	public static void testData(NeuralNet net, TrainSet set) {
		int size = set.size();
		int count = 0;
		
		for(int i = 0; i< set.size(); i++) {
			net.calHiddenOutPut(set.getInput(i));
			net.calFinalOutput();
			int a = findMaxIndex(net.finalOutput);
			int b = findMaxIndex(set.getOutput(i));
			
			if(a == b) count++;
			
		}
		
		System.out.println("System gets " + count + " correct prediction from " + size+ " hand-writing images by Sigmoid");

	} 
	
	public static int findMaxIndex(double[] nums) {
		double max = 0;
		int index = 0;
		for(int i = 0; i < nums.length; i++) {
			if(nums[i] > max) {
				max = nums[i];
				index =i;
			}
		}
		return index;
	}
}


		
		
//		int[] net = {3,3,1};
//		double[] input1 = {1.0, 1.0, 0.0};
//		double[] input2 = {0.0, 1.0, 0.0};
//		double[] input3 = {1.0, 0.0, 0.0};
//		double[] input4 = {0.0, 0.0, 0.0};
//		double[] expected1 = {1.0};
//		double[] expected2 = {0.0};
//		double[] expected3 = {1.0};
//		NeuralNet nt = new NeuralNet(net);
//		for(int i = 0; i<10000; i++) {
//			nt.train(input1, expected1, 0.2);
//			nt.train(input2, expected2, 0.2);
//			nt.train(input3, expected3, 0.2);	
//		}
//		nt.guess(input4);
//		for(int m = 0; m < nt.outputNeural; m++) {
//			System.out.println("the predict result is: " + nt.finalOutput[m]);
//		}
		
//		nt.generateTestWeights();
//		for(int i= 0; i< nt.inputNeural; i++) {
//			for ( int j = 0; j< nt.hiddenNeural; j++) {
//				double weight = nt.inputWeights[i][j];
//				System.out.println("the weights are:" + weight);
//			}
//		}
//		nt.generateTestBias();
//		for(int m = 0; m < nt.hiddenNeural ; m++) System.out.println("the bias are:" + nt.hiddenBias[m]); 
//		for(int n = 0; n < nt.outputNeural; n++) System.out.println("the bias are:" + nt.outputBias[n]);
		



