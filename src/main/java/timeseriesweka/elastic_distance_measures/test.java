package timeseriesweka.elastic_distance_measures;


import java.io.*;
import java.io.File;
import java.io.FileWriter;
import java.io.FileNotFoundException;  
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
import utilities.ClassifierTools;
import weka.core.DenseInstance;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.kNN;
import weka.core.Capabilities;
import timeseriesweka.elastic_distance_measures.DTW;
import timeseriesweka.elastic_distance_measures.WeightedDTW;
import timeseriesweka.elastic_distance_measures.DTW_DistanceBasic;
import timeseriesweka.elastic_distance_measures.LCSSDistance;
import timeseriesweka.elastic_distance_measures.MSMDistance;
import timeseriesweka.elastic_distance_measures.TWEDistance;
import timeseriesweka.elastic_distance_measures.ERPDistance;
import timeseriesweka.elastic_distance_measures.test;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.DTW1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ED1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.ERP1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.Efficient1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.LCSS1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.MSM1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.TWE1NN;
import timeseriesweka.classifiers.ensembles.elastic_ensemble.WDTW1NN;
import timeseriesweka.filters.DerivativeFilter;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


public class test {

    
    private String datasetName;
    private String outputFileDirStr;
    private File outputFileDir;
    private Instances train;
    private Instances test;
    private Instances derTrain;
    private Instances derTest;
    

    public static void makeFileDir(File fileDir)throws FileNotFoundException, IOException, Exception  {
    
        
        if  (!fileDir.exists()  && !fileDir.isDirectory())      
        {          
              fileDir.mkdir();    
        } 
    
    }
    
    public enum OptionalMeasures{ 
        DTWR1,
        DTWR02,
        DTWR05,
        DTWR08,
        DDTWR1,
        DDTWR02,
        DDTWR05,
        DDTWR08,
        WDTWG0,
        WDTWG02,
        WDTWG05,
        WDTWG08,
        WDDTWG0,
        WDDTWG02,
        WDDTWG05,
        WDDTWG08,
        LCSS,
        ERP,
        MSM,
        TWE,
        ED
    };
    
    public static Efficient1NN getClassifier(OptionalMeasures measure) throws Exception{
        Efficient1NN knn = null;
        switch(measure){
            case DTWR1:{
                return new DTW1NN(1);
                }
            case DTWR02:{
                return new DTW1NN(0.2);
                }
            case DTWR05:{
                return new DTW1NN(0.5);
                }
            case DTWR08:{
                return new DTW1NN(0.8);
                }
            case DDTWR1:{
                return new DTW1NN(1);
                }
            case DDTWR02:{
                return new DTW1NN(0.2);
                }
            case DDTWR05:{
                return new DTW1NN(0.5);
                }
            case DDTWR08:{
                return new DTW1NN(0.8);
                }
            case WDTWG0:{
                return new WDTW1NN( );
                }
            case WDTWG02:{
                return new WDTW1NN(0.2);
                }
            case WDTWG05:{
                return new WDTW1NN(0.5);
                }
            case WDTWG08:{
                return new WDTW1NN(0.8);
                }
            case WDDTWG0:{
                return new WDTW1NN( );
                }
            case WDDTWG02:{
                return new WDTW1NN(0.2);
                }
            case WDDTWG05:{
                return new WDTW1NN(0.5);
                }
            case WDDTWG08:{
                return new WDTW1NN(0.8);
                }
            case LCSS:{
                return new LCSS1NN();
                }
            case ERP:{
                return new ERP1NN();
                }
            case MSM:{
                return new MSM1NN();
                }
            case TWE:{
                return new TWE1NN();
                }
            case ED:{
                return new ED1NN();
                }
            default:{
                throw new Exception("Unsupported classifier type");
                }
        }
            
    }
    
    public final OptionalMeasures[] measuresToUse ;
    
    public static boolean isDerivative(OptionalMeasures measure){
        return (measure==OptionalMeasures.DDTWR1 || measure==OptionalMeasures.DDTWR02 || measure==OptionalMeasures.DDTWR05 || measure==OptionalMeasures.DDTWR08 || measure==OptionalMeasures.WDDTWG0 || measure==OptionalMeasures.WDDTWG02 || measure==OptionalMeasures.WDDTWG05 || measure==OptionalMeasures.WDDTWG08);
    }
    
    /**
     * Constructor for ; includes all optional measures 
     */
    
    public test(String dataDirStr, String outputFileDirStr, String datasetName)throws FileNotFoundException, IOException, Exception  {
        this.datasetName = datasetName;
        this.outputFileDirStr = outputFileDirStr;
        this.outputFileDir =new File(this.outputFileDirStr + "/" + this.datasetName);
        this.makeFileDir(outputFileDir);
        this.measuresToUse = OptionalMeasures.values();
        this.train = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TRAIN");
        this.test = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TEST");
        DerivativeFilter df = new DerivativeFilter();
        this.derTrain = df.process(train);
        this.derTest = df.process(test);
    }
    
    public test(String dataDirStr, String outputFileDirStr, String datasetName, String[] measuresToUseStr)throws FileNotFoundException, IOException, Exception  {
        this.datasetName = datasetName;
        this.outputFileDirStr = outputFileDirStr;
        this.outputFileDir =new File(this.outputFileDirStr + "/" + this.datasetName);
        this.makeFileDir(outputFileDir);
        this.measuresToUse = new OptionalMeasures[measuresToUseStr.length];
        for(int d = 0; d < measuresToUseStr.length; d++){ 
            this.measuresToUse[d] = OptionalMeasures.valueOf(measuresToUseStr[d]);
        }
        this.train = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TRAIN");
        this.test = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TEST");
        DerivativeFilter df = new DerivativeFilter();
        this.derTrain = df.process(train);
        this.derTest = df.process(test);
    }
 
    public void distanceTrainToTrainFileOutput()throws FileNotFoundException, IOException, Exception {
        for(int c = 0; c < this.measuresToUse.length; c++){
                Efficient1NN ms =  this.getClassifier(this.measuresToUse[c]);
                String filePathTrainToTrainDistance = this.outputFileDirStr + "/" + this.datasetName + "/" + this.datasetName + "_" + this.measuresToUse[c].toString() + "_" + "train" + ".dsv";
                File fileTrainToTrainDistance = new File(filePathTrainToTrainDistance);
                fileTrainToTrainDistance.createNewFile();  
                FileWriter writerTrainToTrainDistance = new FileWriter(fileTrainToTrainDistance);
                if(this.isDerivative(this.measuresToUse[c])){
                    for(int i = 0; i < this.derTrain.numInstances(); i++){
                        writerTrainToTrainDistance.write(String.valueOf((int)this.derTrain.instance(i).classValue()));
                        for(int j = 0; j < this.derTrain.numInstances(); j++){
                        writerTrainToTrainDistance.write(","+ String.valueOf(ms.distance(this.derTrain.instance(i),this.derTrain.instance(j),Double.POSITIVE_INFINITY)));
                        }
                        writerTrainToTrainDistance.write("\n");
                    }
                    writerTrainToTrainDistance.flush();
                    writerTrainToTrainDistance.close();  
                }else {
                    for(int i = 0; i < this.train.numInstances(); i++){
                        writerTrainToTrainDistance.write(String.valueOf((int)this.train.instance(i).classValue()));
                        for(int j = 0; j < this.train.numInstances(); j++){
                        writerTrainToTrainDistance.write(","+ String.valueOf(ms.distance(this.train.instance(i),this.train.instance(j),Double.POSITIVE_INFINITY)));
                        }
                        writerTrainToTrainDistance.write("\n");
                    }
                    writerTrainToTrainDistance.flush();
                    writerTrainToTrainDistance.close(); 
                }
        }
    
    }
    
    
    public void distanceTestToTrainFileOutput()throws FileNotFoundException, IOException, Exception {
        for(int c = 0; c < this.measuresToUse.length; c++){
                Efficient1NN ms =  this.getClassifier(this.measuresToUse[c]);
                String filePathTestToTrainDistance = this.outputFileDirStr + "/" + this.datasetName + "/" + this.datasetName + "_" + this.measuresToUse[c].toString() + "_" + "test" + ".dsv";
                File fileTestToTrainDistance = new File(filePathTestToTrainDistance);
                fileTestToTrainDistance.createNewFile();  
                FileWriter writerTestToTrainDistance = new FileWriter(fileTestToTrainDistance);
                if(this.isDerivative(this.measuresToUse[c])){
                    for(int i = 0; i < this.derTest.numInstances(); i++){
                        writerTestToTrainDistance.write(String.valueOf((int)this.derTest.instance(i).classValue()));
                        for(int j = 0; j < this.derTrain.numInstances(); j++){
                        writerTestToTrainDistance.write(","+ String.valueOf(ms.distance(this.derTest.instance(i),this.derTrain.instance(j),Double.POSITIVE_INFINITY)));
                        }
                        writerTestToTrainDistance.write("\n");
                    }
                    writerTestToTrainDistance.flush();
                    writerTestToTrainDistance.close();  
                }else {
                    for(int i = 0; i < this.test.numInstances(); i++){
                        writerTestToTrainDistance.write(String.valueOf((int)this.test.instance(i).classValue()));
                        for(int j = 0; j < this.train.numInstances(); j++){
                        writerTestToTrainDistance.write(","+ String.valueOf(ms.distance(this.test.instance(i),this.train.instance(j),Double.POSITIVE_INFINITY)));
                        }
                        writerTestToTrainDistance.write("\n");
                    }
                    writerTestToTrainDistance.flush();
                    writerTestToTrainDistance.close(); 
                }
        }
    
    }    
    

    public static void  main(String[] args)throws FileNotFoundException, IOException, Exception {
        
        String outputFileDirStr = "/scratch/dy1n16/TSCResults";
        String dataDirStr = "/scratch/dy1n16/NewTSCProblems";
        String[] datasetNames_4 = {"HandOutlines"};
        String[] measuresToUseStr_4 = {"DTWR1","DTWR02","DTWR05","DTWR08","DDTWR1","DDTWR02","DDTWR05","DDTWR08","LCSS","ERP","MSM","TWE","WDTWG0","WDTWG02","WDTWG05","WDTWG08","WDDTWG0","WDDTWG02","WDDTWG05","WDDTWG08","ED"};
        
        for(int d = 0; d < datasetNames_4.length; d++){  
            String datasetName =  datasetNames_4[d];
            test dfo = new test(dataDirStr, outputFileDirStr, datasetName, measuresToUseStr_4);
            dfo.distanceTestToTrainFileOutput(); 
        }
     
    }
    
    

}