package de.upb.cs.is.jpl.api.algorithm.baselearner.regression.logisticWEKA;

import java.util.ArrayList;

import de.upb.cs.is.jpl.api.algorithm.AAlgorithmConfiguration;
import de.upb.cs.is.jpl.api.algorithm.baselearner.ABaselearnerAlgorithm;
import de.upb.cs.is.jpl.api.algorithm.baselearner.EBaseLearner;
import de.upb.cs.is.jpl.api.algorithm.baselearner.dataset.BaselearnerDataset;
import de.upb.cs.is.jpl.api.algorithm.learningalgorithm.ILearningModel;
import de.upb.cs.is.jpl.api.dataset.IDataset;
import de.upb.cs.is.jpl.api.exception.algorithm.TrainModelsFailedException;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.AddValues;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * This class encapsulates logistic regression as implemented by WEKA in the
 * class @link {@link Logistic}. For training, there must be examples for
 * <b>every</b> desired class value (not only for one).
 * 
 * @author Helena Graf
 *
 */
public class LogisticRegressionWEKA extends ABaselearnerAlgorithm<LogisticRegressionWEKAConfiguration> {

	/**
	 * Creates a LogisticRegressionWEKA object with default parameters (same as WEKA
	 * default parameters).
	 */
	public LogisticRegressionWEKA() {
		super(EBaseLearner.LOGISTIC_REGRESSION_WEKA.getBaseLearnerIdentifier());
	}

	/**
	 * Creates a LogisticRegressionWEKA object with the given parameters.
	 * 
	 * @param debug
	 *            whether debugging output is enabled
	 * @param useConjugateGradientDescent
	 *            whether conjugate gradient descent should be used
	 * @param ridge
	 *            the ridge in the log-likelihood
	 * @param maxIts
	 *            the maximum number of iterations
	 * @param batchSize
	 *            the preferred batch size for batch prediction
	 * @param numDecimalPlaces
	 *            the number of decimal places shown in the output
	 * @param doNotCheckCapabilities
	 *            whether the capabilities should not be checked
	 */
	public LogisticRegressionWEKA(boolean debug, boolean useConjugateGradientDescent, double ridge, int maxIts,
			int batchSize, int numDecimalPlaces, boolean doNotCheckCapabilities) {
		this();
		this.configuration.setDebug(debug);
		this.configuration.setUseConjugateGradientDescent(useConjugateGradientDescent);
		this.configuration.setRidge(ridge);
		this.configuration.setMaxIts(maxIts);
		this.configuration.setBatchSize(batchSize);
		this.configuration.setNumDecimalPlaces(numDecimalPlaces);
		this.configuration.setDoNotCheckCapabilities(doNotCheckCapabilities);
	}

	@Override
	protected ILearningModel<?> performTraining(IDataset<?, ?, ?> dataset) throws TrainModelsFailedException {
		// Transform data set into a WEKA instances object
		BaselearnerDataset baseLearnerDataSet = (BaselearnerDataset) dataset;
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for (int i = 0; i < baseLearnerDataSet.getNumberOfFeatures(); i++) {
			attInfo.add(new Attribute("a" + i));
		}
		attInfo.add(new Attribute("target"));
		Instances train = new Instances(dataset.getUniqueStringIdentifyingDataset(), attInfo, 0);
		double[][] featureVectors = baseLearnerDataSet.getFeatureVectors();
		double[] correctResults = baseLearnerDataSet.getCorrectResults();
		double[] weights = baseLearnerDataSet.getInstanceWeights();
		for (int i = 0; i < featureVectors.length; i++) {
			double[] featureVector = new double[featureVectors[i].length + 1];
			double correctResult = correctResults[i];
			double weight = weights == null ? 1 : weights[i];
			for (int j = 0; j < featureVectors[i].length; j++) {
				featureVector[j] = featureVectors[i][j];
			}
			featureVector[featureVector.length - 1] = correctResult;
			train.add(new DenseInstance(weight, featureVector));
		}
		train.setClassIndex(train.numAttributes() - 1);

		// Transform the target into a nominal target
		Instances output;
		NumericToNominal filter;
		try {
			// Transform numeric attribute to nominal
			filter = new NumericToNominal();
			filter.setAttributeIndices(String.valueOf(train.numAttributes()));
			filter.setInputFormat(train);
			for (Instance instance : train) {
				filter.input(instance);
			}
			filter.batchFinished();
			output = filter.getOutputFormat();
			Instance processed;
			while ((processed = filter.output()) != null) {
				output.add(processed);
			}
		} catch (Exception e1) {
			throw new TrainModelsFailedException(e1);
		}

		// Set up logistic
		Logistic logistic = new Logistic();
		logistic.setDebug(this.configuration.getDebug());
		logistic.setUseConjugateGradientDescent(this.configuration.getUseConjugateGradientDescent());
		logistic.setRidge(this.configuration.getRidge());
		logistic.setMaxIts(this.configuration.getMaxIts());
		logistic.setBatchSize(String.valueOf(this.configuration.getBatchSize()));
		logistic.setNumDecimalPlaces(this.configuration.getNumDecimalPlaces());
		logistic.setDoNotCheckCapabilities(this.configuration.getDoNotCheckCapabilities());

		// Train logistic
		try {
			logistic.buildClassifier(output);
		} catch (Exception e) {
			throw new TrainModelsFailedException(e);
		}

		return new LogisticRegressionWEKALearningModel(logistic, output, filter);
	}

	@Override
	protected AAlgorithmConfiguration createDefaultAlgorithmConfiguration() {
		LogisticRegressionWEKAConfiguration config = new LogisticRegressionWEKAConfiguration();
		config.initializeDefaultConfiguration();
		return config;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = result * prime;
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
		return true;
	}

}
