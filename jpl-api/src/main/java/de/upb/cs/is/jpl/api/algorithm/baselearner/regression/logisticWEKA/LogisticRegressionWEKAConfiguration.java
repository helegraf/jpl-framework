package de.upb.cs.is.jpl.api.algorithm.baselearner.regression.logisticWEKA;

import de.upb.cs.is.jpl.api.algorithm.AAlgorithmConfiguration;
import de.upb.cs.is.jpl.api.configuration.json.IJsonConfiguration;
import de.upb.cs.is.jpl.api.exception.configuration.json.ParameterValidationFailedException;
import de.upb.cs.is.jpl.api.util.StringUtils;
import weka.classifiers.functions.Logistic;

/**
 * This configuration stores all parameters used by
 * {@link LogisticRegressionWEKA}. It encapsulates the WEKA implementation of
 * logistic regression with all parameters as found in the {@link Logistic}
 * class.
 * 
 * @author Helena Graf
 *
 */
public class LogisticRegressionWEKAConfiguration extends AAlgorithmConfiguration {

	/**
	 * The name of the default configuration file for this configuration.
	 */
	private static final String DEFAULT_CONFIGURATION_FILE_NAME = "baselearner" + StringUtils.FORWARD_SLASH
			+ "regression" + StringUtils.FORWARD_SLASH + "logistic" + StringUtils.FORWARD_SLASH
			+ "logistic_regression_WEKA";

	private static final String PARAMETER_NAME_DEBUGGING_FLAG = "debug";
	private static final String PARAMETER_NAME_USECONJUGATEGRADIENTDESCENT = "useConjugateGradientDescent";
	private static final String PARAMETER_NAME_RIDGE = "ridge";
	private static final String PARAMETER_NAME_NUM_ITERATIONS = "maxIts";
	private static final String PARAMETER_NAME_BATCHSIZE = "batchSize";
	private static final String PARAMETER_NAME_NUMDECIMALPLACES = "numDecimalPlaces";
	private static final String PARAMETER_NAME_DONOTCHECKCAPABILITIES = "doNotCheckCapabilities";

	private boolean debug;
	private boolean useConjugateGradientDescent;
	private double ridge;
	private int maxIts;
	private int batchSize;
	private int numDecimalPlaces;
	private boolean doNotCheckCapabilities;

	/**
	 * Creates a default LogisticRegressionWEKAConfiguration.
	 */
	public LogisticRegressionWEKAConfiguration() {
		super(DEFAULT_CONFIGURATION_FILE_NAME);
	}

	@Override
	protected void validateParameters() throws ParameterValidationFailedException {
		if (ridge <= 0)
			throw new ParameterValidationFailedException("Ridge must be >= 0.");

		if (batchSize < 1)
			throw new ParameterValidationFailedException("Batch size must be a least 1. Current value: " + batchSize);

		if (maxIts < -1)
			throw new ParameterValidationFailedException("Max Iterations must be >= -1. Current value: " + maxIts);
		if (numDecimalPlaces < 0)
			throw new ParameterValidationFailedException(
					"Number of shown decimal places must be >= 0. Current value: " + numDecimalPlaces);
	}

	@Override
	protected void copyValues(IJsonConfiguration configuration) {
		LogisticRegressionWEKAConfiguration castedConfiguration = (LogisticRegressionWEKAConfiguration) configuration;
		this.debug = castedConfiguration.debug;
		this.useConjugateGradientDescent = castedConfiguration.useConjugateGradientDescent;
		if (castedConfiguration.ridge < Double.MAX_VALUE)
			this.ridge = castedConfiguration.ridge;
		if (castedConfiguration.maxIts < Integer.MAX_VALUE)
			this.maxIts = castedConfiguration.maxIts;
		if (castedConfiguration.batchSize < Integer.MAX_VALUE)
			this.batchSize = castedConfiguration.batchSize;
		if (castedConfiguration.numDecimalPlaces < Integer.MAX_VALUE)
			this.batchSize = castedConfiguration.batchSize;
		this.doNotCheckCapabilities = castedConfiguration.doNotCheckCapabilities;
	}

	/**
	 * @return the preferred batch size for batch prediction.
	 */
	public int getBatchSize() {
		return batchSize;
	}

	/**
	 * @param batchSize
	 *            the preferred batch size for batch prediction.
	 */
	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	/**
	 * @return the number of decimal places shown in the output.
	 */
	public int getNumDecimalPlaces() {
		return numDecimalPlaces;
	}

	/**
	 * @param numDecimalPlaces
	 *            the number of decimal places shown in the output.
	 */
	public void setNumDecimalPlaces(int numDecimalPlaces) {
		this.numDecimalPlaces = numDecimalPlaces;
	}

	/**
	 * @return whether capabilities are not checked (true if not checked).
	 */
	public boolean getDoNotCheckCapabilities() {
		return doNotCheckCapabilities;
	}

	/**
	 * @param doNotCheckCapabilities
	 *            whether not to check capabilities (true if not to check).
	 */
	public void setDoNotCheckCapabilities(boolean doNotCheckCapabilities) {
		this.doNotCheckCapabilities = doNotCheckCapabilities;
	}

	/**
	 * @return whether the debugging output is enabled (default false)
	 */
	public boolean getDebug() {
		return debug;
	}

	/**
	 * @param debug
	 *            whether the debugging output should be enabled (default false)
	 */
	public void setDebug(boolean debug) {
		this.debug = debug;
	}

	/**
	 * @return whether conjugate gradient descent is used (default false)
	 */
	public boolean getUseConjugateGradientDescent() {
		return useConjugateGradientDescent;
	}

	/**
	 * @param useConjugateGradientDescent
	 *            whether conjugate gradient descent should be used (default false)
	 */
	public void setUseConjugateGradientDescent(boolean useConjugateGradientDescent) {
		this.useConjugateGradientDescent = useConjugateGradientDescent;
	}

	/**
	 * @return the ridge in the log-likelihood
	 */
	public double getRidge() {
		return ridge;
	}

	/**
	 * @param ridge
	 *            the ridge in the log-likelihood
	 */
	public void setRidge(double ridge) {
		this.ridge = ridge;
	}

	/**
	 * @return the maximum number of iterations (default -1, until convergence)
	 */
	public int getMaxIts() {
		return maxIts;
	}

	/**
	 * @param maxIts
	 *            the maximum number of iterations (default -1, until convergence)
	 */
	public void setMaxIts(int maxIts) {
		this.maxIts = maxIts;
	}

	@Override
	public String toString() {
		return PARAMETER_NAME_DEBUGGING_FLAG + StringUtils.COLON + debug
				+ StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND + PARAMETER_NAME_USECONJUGATEGRADIENTDESCENT
				+ StringUtils.COLON + useConjugateGradientDescent + StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND
				+ PARAMETER_NAME_RIDGE + StringUtils.COLON + ridge + StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND
				+ PARAMETER_NAME_NUM_ITERATIONS + StringUtils.COLON + maxIts
				+ StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND + PARAMETER_NAME_BATCHSIZE + StringUtils.COLON
				+ batchSize + StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND + PARAMETER_NAME_NUMDECIMALPLACES
				+ StringUtils.COLON + numDecimalPlaces + StringUtils.COMMA_WITH_SINGLE_WHITESPACE_BEHIND
				+ PARAMETER_NAME_DONOTCHECKCAPABILITIES + StringUtils.COLON + doNotCheckCapabilities;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + ((debug == false) ? 0 : 1);
		result = prime * result + ((useConjugateGradientDescent == false) ? 0 : 1);
		result = prime * result + (int) Math.round(ridge);
		result = prime * result + maxIts;
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
		LogisticRegressionWEKAConfiguration other = (LogisticRegressionWEKAConfiguration) obj;
		if (other.debug != debug || other.useConjugateGradientDescent != useConjugateGradientDescent
				|| other.ridge != ridge || other.maxIts != maxIts)
			return false;
		return true;
	}

}
