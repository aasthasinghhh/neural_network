package learn.nn.core.examples;

import java.io.IOException;
import java.util.List;

import learn.nn.core.Connection;
import learn.nn.core.InputUnit;
import learn.nn.core.LogisticUnit;
import learn.nn.core.MultiLayerFeedForwardNeuralNetwork;
import learn.nn.core.Unit;
import learn.nn.core.Example;

public class IrisNeuralNetwork extends MultiLayerFeedForwardNeuralNetwork {
    private static final int numLayers = 3;
	private static final int INPUTS = 0;
	private static final int HIDDENS = 1;
	private static final int OUTPUTS = 2;
	
	private static final int NUM_INPUTS = 4;
	private static final int NUM_HIDDENS = 7;
	private static final int NUM_OUTPUTS = 3;
	
	public IrisNeuralNetwork() {
		super(new Unit[numLayers][]);
		// Input units
		this.layers[INPUTS] = new InputUnit[NUM_INPUTS];
		for (int i=0; i < NUM_INPUTS; i++) {
			this.layers[INPUTS][i] = new InputUnit();
		}
		// Hidden units: each connected to all input units
		this.layers[HIDDENS] = new LogisticUnit[NUM_HIDDENS];
		for (int j=0; j < NUM_HIDDENS; j++) {
			this.layers[HIDDENS][j] = new LogisticUnit();
			for (int i=0; i < NUM_INPUTS; i++) {
                new Connection(this.layers[INPUTS][i], this.layers[HIDDENS][j]);
			}
		}
		// Output units
		this.layers[OUTPUTS] = new LogisticUnit[NUM_OUTPUTS];
		for (int j=0; j < NUM_OUTPUTS; j++) {
			this.layers[OUTPUTS][j] = new LogisticUnit();
			for (int i=0; i < NUM_HIDDENS; i++) {
				new Connection(this.layers[HIDDENS][i], this.layers[OUTPUTS][j]);
			}
		}
	}

	public int getOutputValue() {
		double max = Double.NEGATIVE_INFINITY;
		int answer = -1;
		for (int i=0; i < this.getOutputUnits().length; i++) {
			Unit unit = this.getOutputUnits()[i];
			if (unit.getOutput() > max) {
				answer = i;
				max = unit.getOutput();
			}
		}
		return answer;
	}
	
	public double[][] confusionMatrix(List<Example> examples) {
		int numClasses = this.getOutputUnits().length;
		int[] n = new int[numClasses];					
		int[][] counter = new int[numClasses][numClasses];	
		for (Example example : examples) {
			propagate(example);
			int predicted = getOutputValue();
			int actual = -1;
			for (int i=0; i < example.outputs.length; i++) {
				if (example.outputs[i] == 1.0) {
					actual = i;
					break;
				}
			}
			n[actual] += 1;
			counter[actual][predicted] += 1;
		}
		double[][] result = new double[numClasses][numClasses];
		for (int i=0; i < 3; i++) {
			for (int j=0; j < 3; j++) {
				result[i][j] = (double)counter[i][j] / n[i];
			}
		}
		return result;
	}

	public static void main(String[] argv) throws IOException {
		int epochs = 1000;
		double alpha = 0.10;
		if (argv.length > 0) {
			epochs = Integer.parseInt(argv[0]);
		}
		if (argv.length > 1) {
			alpha = Double.parseDouble(argv[1]);
		}
		List<Example> examples = new IrisDatasetReader("/Users/astha.s918/Documents/Java Repos/CSC 242/Project 4/src/learn/nn/core/examples/iris.data.txt").examples();
		IrisNeuralNetwork network = new IrisNeuralNetwork();
		System.out.println("Training for " + epochs + " epochs with alpha=" + alpha);
		network.train(examples, epochs, alpha);
		network.dump();
		double accuracy = network.test(examples);
		System.out.println("Overall accuracy=" + accuracy);
		System.out.println();
		System.out.println("Confusion matrix:");
		double[][] matrix = network.confusionMatrix(examples);
		System.out.println("\tPredicted");
		System.out.print("Actual");
		for (int i=0; i < matrix.length; i++) {
			System.out.format("\t%d", i);
		}
		System.out.println();
		for (int i=0; i < matrix.length; i++) {
			System.out.format("%d", i);
			for (int j=0; j < matrix[i].length; j++) {
				System.out.format("\t%.3f", matrix[i][j]);
			}
			System.out.println();
		}
		System.out.println();

        //k-Fold Cross-Validation
		int numExamples = examples.size();
		int kFolds = 10;
		System.out.println("k-Fold Cross-Validation: n=" + numExamples + ", k=" + kFolds);
		double acc = network.kFoldCrossValidate(examples, kFolds, epochs, alpha);
		System.out.format("average accuracy: %.3f\n", acc);
		System.out.println();
		System.out.println("Learning Curve testing on all training data");
		System.out.println("EPOCHS\tACCURACY");
		for (epochs = 0; epochs <= 4000; epochs+=100) {
			network.train(examples, epochs, alpha);
			accuracy = network.test(examples);
			System.out.format("%d\t%.3f\n",  epochs, accuracy);
		}
	}
}
