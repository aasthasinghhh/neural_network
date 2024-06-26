package learn.nn.core;

import java.util.List;

import learn.math.util.VectorOps;

/**
 * A PerceptronUnit is a Unit that uses a hard threshold
 * activation function.
 */
public class PerceptronUnit extends NeuronUnit {

	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold
	 * at z=0. (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		// This must be implemented by you
		if(z < 0) {
			return 0;
		}
		else {
			return 1;
		}
	}

	@Override
	public double activationPrime(double z) {
    	// Return a dummy value as the derivative is not used for Perceptron
    	return 0;
	}
	
	/**
	 * Update this unit's weights using the Perceptron learning
	 * rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		// This must be implemented by you
		List<Connection> incomingConnections = this.incomingConnections;
		double weightVector[]=new double [incomingConnections.size()];
		for(int i=0;i<incomingConnections.size();i++){
			weightVector[i]=incomingConnections.get(i).weight;
		}
		double xNew[]=new double[x.length+1];
		xNew[0]=1;
		for(int i=0; i<x.length;i++){
			xNew[i+1]=x[i];
		}
		double hw = this.activation(VectorOps.dot(weightVector, xNew));
		for(int i=0; i<xNew.length; i++){
			incomingConnections.get(i).weight = incomingConnections.get(i).weight + (alpha * (y-hw))*xNew[i];
		}
	}
}
