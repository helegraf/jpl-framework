package de.upb.cs.is.jpl.api.math.distribution.special;


import de.upb.cs.is.jpl.api.math.distribution.ADistributionSampleSet;
import de.upb.cs.is.jpl.api.math.distribution.IDistributionSampleSet;
import de.upb.cs.is.jpl.api.math.distribution.numerical.integer.IIntegerDistribution;


/**
 * This abstract class represents an abstract {@link IDistributionSampleSet} for
 * {@link ISpecialDistribution}s, which can contain common functionality and variables among all
 * sample sets.
 * 
 * @author Alexander Hetzer
 *
 * @param <SPACE> the space (domain) of the distribution whose samples should be contained inside
 *           this sample set, e.g. Integer for {@link IIntegerDistribution}s
 */
public abstract class ASpecialDistributionSampleSet<SPACE> extends ADistributionSampleSet<SPACE> {

}
