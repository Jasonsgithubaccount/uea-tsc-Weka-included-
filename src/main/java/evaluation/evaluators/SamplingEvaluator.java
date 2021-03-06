/*
 * Copyright (C) 2019 xmw13bzu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package evaluation.evaluators;

import evaluation.storage.ClassifierResults;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

/**
 * Base class for evaluators that will evaluate over many resamples (e.g stratified random resamples)
 * or folds (e.g cross validation) of the data. In api methods, I have simply referred to these 
 * as folds as a semi-arbitrary choice.
 * 
 * Provides functionality for cloning and saving the trained models on each fold 
 * 
 * @author James Large (james.large@uea.ac.uk)
 */
public abstract class SamplingEvaluator extends Evaluator {

    int numFolds;
    
    /**
     * If true, the classifiers shall be cloned when building and predicting on each fold. 
     * 
     * This is achieved via AbstractClassifier.makeCopy(...), and therefore the classifier
     * and all relevant/wanted info/hyperparamters that may have been set up prior to giving 
     * the classifier to the evaluator must be properly (de-)serialisable.
     * 
     * Useful if a particular classifier maintains information after one buildclassifier that 
     * might not be replaced or effect the next call to buildclassifier. Ideally, this 
     * should not be the case, but this option will make sure either way
     * 
     * If maintainClassifiers == true, clone classifiers is forced to true
     */
    protected boolean cloneClassifiers = false;
    
    /**
     * If true, will keep the classifiers trained on each fold in memory
     * 
     * When set to true, will force clone classifier to also be true. Note - this will naturally 
     * come with a large cost to required memory, (size of trained classifier) * numFolds
     */
    protected boolean maintainClassifiers = false;
    
    /** 
     * If maintainClassifiers is true, this will become populated with the classifiers 
     * trained on each fold, [classifier][fold], otherwise will be null
     */
    protected Classifier[][] foldClassifiers = null;
    
    /**
     * Populated with the classifierresults object for each fold, such that each
     * object effectively represents a single hold-out validation set. 
     * [classifier][fold] 
     */
    protected ClassifierResults[][] resultsPerFold = null;
    
    public SamplingEvaluator() {
        super(0,false,false);
    }
    
    public SamplingEvaluator(int seed, boolean cloneData, boolean setClassMissing, boolean cloneClassifiers, boolean maintainClassifiers) {
        super(seed, cloneData, setClassMissing);
        
        this.cloneClassifiers = cloneClassifiers;
        setMaintainClassifiers(maintainClassifiers);
    }

    public int getNumFolds() {
        return numFolds;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }
    
    public ClassifierResults[] getFoldResults() {
        return getFoldResults(0);
    }
    
    public ClassifierResults[] getFoldResults(int classifierIndex) {
        if (resultsPerFold != null)
            return resultsPerFold[0];
        else
            return null;
    }
    
    public ClassifierResults[][] getFoldResultsAll() {
        return resultsPerFold;
    }
    
    public Classifier[] getFoldClassifiers() {
        return getFoldClassifiers(0);
    }
    
    public Classifier[] getFoldClassifiers(int classifierIndex) {
        if (foldClassifiers != null)
            return foldClassifiers[0];
        else
            return null;
    }
    
    public Classifier[][] getFoldClassifiersAll() {
        return foldClassifiers;
    }
    
    public void setMaintainClassifiers(boolean maintainClassifiers) { 
        this.maintainClassifiers = maintainClassifiers;
        if (maintainClassifiers)
            this.cloneClassifiers = true;
    }
    
    protected void cloneClassifier(Classifier classifier) throws Exception {
        // clone them all here in one go for efficiency of serialisation
        foldClassifiers = new Classifier[1][];

        foldClassifiers[0] = AbstractClassifier.makeCopies(classifier, numFolds);
    }
    
    protected void cloneClassifiers(Classifier[] classifiers) throws Exception {
        // clone them all here in one go for efficiency of serialisation
        foldClassifiers = new Classifier[classifiers.length][];

        for (int c = 0; c < classifiers.length; ++c)
            foldClassifiers[c] = AbstractClassifier.makeCopies(classifiers[c], numFolds);
    }
}
