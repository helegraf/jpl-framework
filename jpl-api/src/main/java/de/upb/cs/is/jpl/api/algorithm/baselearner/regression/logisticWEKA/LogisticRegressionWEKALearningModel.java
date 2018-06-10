package de.upb.cs.is.jpl.api.algorithm.baselearner.regression.logisticWEKA;

import de.upb.cs.is.jpl.api.algorithm.baselearner.ABaseLearningModel;
import de.upb.cs.is.jpl.api.algorithm.baselearner.dataset.BaselearnerInstance;
import de.upb.cs.is.jpl.api.dataset.IInstance;
import de.upb.cs.is.jpl.api.exception.UnsupportedOperationException;
import de.upb.cs.is.jpl.api.exception.algorithm.PredictionFailedException;
import de.upb.cs.is.jpl.api.math.linearalgebra.IVector;
import weka.classifiers.functions.Logistic;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Encapsulates a trained WEKA {@link Logistic} model.
 * 
 * @author Helena Graf
 *
 */
public class LogisticRegressionWEKALearningModel extends ABaseLearningModel<Double> {

	private Logistic learner;
	private Instances trainingData;

	/**
	 * Creates a new WEKA-based logistic regression model which can make predictions
	 * based on the given leaner that has been trained on the given training data.
	 * 
	 * @param learner
	 *            the learner used to make predictions
	 * @param trainingData
	 *            the data the learner was trained on
	 */
	public LogisticRegressionWEKALearningModel(Logistic learner, Instances trainingData) {
		super(trainingData.numAttributes() - 1);
		this.learner = learner;
		this.trainingData = trainingData;
	}

	@Override
	public Double predict(IInstance<?, ?, ?> instance) throws PredictionFailedException {
		// Convert the instance to a WEKA instance
		BaselearnerInstance convertedInstance = (BaselearnerInstance) instance;
		Instance wekaInstance = new DenseInstance(convertedInstance.getWeight(), convertedInstance.getFeatureVector());

		// Assign to instances training data
		wekaInstance.setDataset(trainingData);

		// Make prediction
		Double prediction;
		try {
			prediction = learner.classifyInstance(wekaInstance);
		} catch (Exception e) {
			throw new PredictionFailedException(e);
		}
		return prediction;
	}

	@Override
	public double getBias() throws UnsupportedOperationException {
		throw new UnsupportedOperationException("Cannot get bias for WEKA implementatino of logistic regression.");
	}

	@Override
	public IVector getWeightVector() throws UnsupportedOperationException {
		throw new UnsupportedOperationException(
				"Cannot get weight vector for WEKA implementation of logistic regression.");
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + ((learner == null) ? 0 : learner.hashCode());
		result = prime * result + ((trainingData != null) ? 0 : trainingData.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (!super.equals(obj))
			return false;
		if (getClass() != obj.getClass())
			return false;
		LogisticRegressionWEKALearningModel other = (LogisticRegressionWEKALearningModel) obj;
		if (learner == null) {
			if (other.learner != null)
				return false;
		} else if (!trainingData.equals(other.trainingData))
			return false;
		return true;
	}

}
