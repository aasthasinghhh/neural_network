package learn.nn.core.examples;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import learn.nn.core.Example;

public class IrisDatasetReader {
	
	Scanner sc;
	
	// constructor initializes a scanner to read data from the provided filename
	public IrisDatasetReader(String filename) throws IOException {
		sc = new Scanner(new FileInputStream(filename));
		// delimiter to separate values in the file
		sc.useDelimiter("[,\\n]");
	}
	
	// reads the next data example from the dataset
	// parses the sepal length, sepal width, petal length, petal width, and converts the species label into vector.
	public Example nextData() throws IOException {

		// features of iris flower
		double sepalLength = sc.nextDouble();
		double sepalWidth = sc.nextDouble();
		double petalLength = sc.nextDouble();
		double petalWidth = sc.nextDouble();
		String label = sc.next();

		double[] inputs = {sepalLength, sepalWidth, petalLength, petalWidth};
		double[] outputs = {0.0, 0.0, 0.0};
		
		if (label.startsWith("Iris-setosa")) {
			outputs[0] = 1.0;
		} else if (label.startsWith("Iris-versicolor")) {
			outputs[1] = 1.0;
		} else if (label.startsWith("Iris-virginica")) { 
			outputs[2] = 1.0;
		} else {
			throw new IOException("bad class label " + label);
		}
		return new Example(inputs, outputs);
	}
	
	public List<Example> examples(int length) throws IOException {
		List<Example> result = new ArrayList<Example>(length);
		for (int i=0; i < length; i++) {
			result.add(nextData());
		}
		return result;
	}
	
	public List<Example> examples() throws IOException {
		return examples(150);
	}
}