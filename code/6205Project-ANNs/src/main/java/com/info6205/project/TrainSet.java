package com.info6205.project;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class TrainSet {
	
	public final int inputSize;
	public final int outputSize;
	
	List<double[][]> data = new ArrayList<double[][]>();
	
	public TrainSet(int inputSize, int outputSize) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
	}
	
    public void addData(double[] in, double[] expected) {
        if(in.length != inputSize || expected.length != outputSize) return;
        data.add(new double[][]{in, expected});
    }

	public TrainSet getRandomData(int size) {
	if(size > 0 && size <= this.size()) {
		TrainSet groupData = new TrainSet(inputSize, outputSize);
		for(int i = 0; i< size; i++) {
			//the previous one
			//int n = (int) Math.random()* (this.size());
			int n = (int) (Math.random()* (this.size()));
			groupData.addData(this.getInput(n), this.getOutput(n));
		}
			return groupData;
		}else return this;		
	}
    
	public int size() {
        return data.size();
    }

    public double[] getInput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[0];
        else return null;
    }

    public double[] getOutput(int index) {
        if(index >= 0 && index < size())
            return data.get(index)[1];
        else return null;
    }

    public int getINPUT_SIZE() {
        return inputSize;
    }

    public int getOUTPUT_SIZE() {
        return outputSize;
    }
}
