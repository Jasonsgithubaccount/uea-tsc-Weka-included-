package timeseriesweka.elastic_distance_measures;


import java.io.*;
import java.io.File;
import java.io.FileWriter;
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
import timeseriesweka.filters.DerivativeFilter;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;


public class DistancesFileOutput {

    
    private String datasetName;
    private String outputFileDirStr;
    private File outputFileDir;
    private boolean usesDer = false;
    private Instances train;
    private Instances test;
    private Instances derTrain;
    private Instances derTest;
    
    private static DerivativeFilter df = new DerivativeFilter();
    
    public void makeFileDir(File fileDir) {
    
        if  (!fileDir.exists()  && !fileDir.isDirectory())      
        {          
              fileDir.mkdir();    
        } 
    
    }
    
    public enum OptionalMeasures{ 
        Euclidean, 
        DTW_R1,
        DTW_R02, 
        DTW_R05, 
        DTW_R08,
        DDTW_R1,
        DDTW_R02, 
        DDTW_R05, 
        DDTW_R08,
        WDTW_R1,
        WDTW_R02,  
        WDTW_R05,  
        WDTW_R08,    
        WDDTW_R1,
        WDDTW_R02, 
        WDDTW_R05, 
        WDDTW_R08,  
        LCSS_3_1, 
        MSM_01, 
        TWE_0005_05, 
        ERP_05_5
    }
    private final OptionalMeasures[] measuresToUse;
    
    
    public static boolean isDerivative(OptionalMeasures measure){
        return (measure==OptionalMeasures.DDTW_R1 || measure==OptionalMeasures.DDTW_R02 || measure==OptionalMeasures.DDTW_R05 || measure==OptionalMeasures.DDTW_R08 || measure==OptionalMeasures.WDDTW_R1 || measure==OptionalMeasures.WDDTW_R02 || measure==OptionalMeasures.WDDTW_R05 || measure==OptionalMeasures.WDDTW_R08);
    }
    
  
    
    
    
    /**
     * Constructor for ; includes all optional measures 
     */
    public DistancesFileOutput(String outputFileDirStr, String datasetName){
        this.datasetName = datasetName;
        this.outputFileDirStr = outputFileDirStr;
        this.measuresToUse = OptionalMeasures.values();
        this.outputFileDir =new File(this.outputFileDirStr + "/" + this.datasetName + "Distances");
        this.makeFileDir(this.outputFileDir);
    }
    /**
     * Constructor allowing specific optional measure types to be passed
     * @param measuresToUse OptionalMeasures[] list of measures to use as enums
     */
   public DistancesFileOutput(String outputFileDirStr, String datasetName, OptionalMeasures[] measuresToUse){
       this.datasetName = datasetName;
       this.outputFileDirStr = outputFileDirStr;
       this.measuresToUse = measuresToUse;
       this.outputFileDir =new File(this.outputFileDirStr + "/" + this.datasetName + "Distances");
       makeFileDir(this.outputFileDir);
    }
     
   
    public static EuclideanDistance getMeasure(OptionalMeasures measure) throws Exception{
        /*EuclideanDistance ms = null;*/
        switch(measure){
            case Euclidean:{
                return new EuclideanDistance();
                }
            case DTW_R1:{
                DTW ms = new DTW();
                ms.setR(1);
                return ms;
                }
            case DTW_R02:{
                DTW ms = new DTW();
                ms.setR(0.2);
                return ms;
                }
            case DTW_R05:{
                DTW ms = new DTW();
                ms.setR(0.5);
                return ms; 
                } 
            case DTW_R08:{
                DTW ms = new DTW();
                ms.setR(0.8);
                return ms; 
                }       
            case DDTW_R1:{
                DTW ms = new DTW();
                ms.setR(1);
                return ms;
                }
            case DDTW_R02:{
                DTW ms = new DTW();
                ms.setR(0.2);
                return ms;
                }
            case DDTW_R05:{
                DTW ms = new DTW();
                ms.setR(0.5);
                return ms; 
                } 
            case DDTW_R08:{
                DTW ms = new DTW();
                ms.setR(0.8);
                return ms;  
                }
            case WDTW_R1:{
                return new WeightedDTW(1);
                }
            case WDTW_R02:{
                return new WeightedDTW(0.2);
                }
            case WDTW_R05:{
                return new WeightedDTW(0.5);
                }
            case WDTW_R08:{
                return new WeightedDTW(0.8);
                }
            case WDDTW_R1:{
                return new WeightedDTW(1);
                }
            case WDDTW_R02:{
                return new WeightedDTW(0.2);
                }
            case WDDTW_R05:{
                return new WeightedDTW(0.5);
                }
            case WDDTW_R08:{
                return new WeightedDTW(0.8);
                }
            case LCSS_3_1:{
                return new LCSSDistance(3,1);
                }
            case ERP_05_5:{
                return new ERPDistance(0.5,5);
                }
            case MSM_01:{
                return new MSMDistance(0.1);
                }
            case TWE_0005_05:{
                return new TWEDistance(0.005, 0.5);
                }
            default:{
                throw new Exception("Unsupported classifier type");
                }
        }
            
    }
    
    
    public void distancesCalculateAndWriteToFile(Instances train, Instances test) throws FileNotFoundException, IOException, Exception{
      this.train = train;
      this.derTrain = null;
      this.test = test;
      this.derTest = null;
      for(int c = 0; c < this.measuresToUse.length; c++){
            if(isDerivative(this.measuresToUse[c])){
                this.usesDer = true;
            }
      }
        
      if(usesDer){
          this.derTrain = this.df.process(train);
          this.derTest = this.df.process(test);
      }
      
      for(int c = 0; c < this.measuresToUse.length; c++){
          EuclideanDistance ms =  getMeasure(this.measuresToUse[c]);
          
          String filePathTrainToTrainDistance = this.outputFileDirStr + "/" + this.datasetName + "Distances" + "/" + datasetName + "_" + this.measuresToUse[c].toString()+ "_" + "train" + ".txt";
          
          File fileTrainToTrainDistance = new File(filePathTrainToTrainDistance);
          try {
             fileTrainToTrainDistance.createNewFile();
          } catch (IOException e) {
            System.out.println("Exception Occurred:");
            e.printStackTrace();
          }
          
          FileWriter writerTrainToTrainDistance = new FileWriter(fileTrainToTrainDistance);
          
          if(isDerivative(this.measuresToUse[c])){
            
            for(int i = 0; i < this.derTrain.numInstances(); i++){
                writerTrainToTrainDistance.write(String.valueOf((int)this.derTrain.instance(i).classValue()));
                for(int j = 0; j < this.derTrain.numInstances(); j++){
                    writerTrainToTrainDistance.write(","+ String.valueOf(ms.distance(this.derTrain.instance(i),this.derTrain.instance(j))));
                }
                writerTrainToTrainDistance.write("\n");
            }
          }else{
            for(int i = 0; i < this.train.numInstances(); i++){
                writerTrainToTrainDistance.write(String.valueOf((int)this.train.instance(i).classValue()));
                for(int j = 0; j < this.train.numInstances(); j++){
                    writerTrainToTrainDistance.write(","+ String.valueOf(ms.distance(this.train.instance(i),this.train.instance(j))));
                }
                writerTrainToTrainDistance.write("\n");
            }
          }
          writerTrainToTrainDistance.close();
          
          
          
          String filePathTestToTrainDistance = this.outputFileDirStr + "/" + this.datasetName + "Distances" + "/" + datasetName + "_" + this.measuresToUse[c].toString()+ "_" + "test" + ".txt";
          
          File fileTestToTrainDistance = new File(filePathTestToTrainDistance);
          try { 
             fileTestToTrainDistance.createNewFile();
          } catch (IOException e) {
            System.out.println("Exception Occurred:");
            e.printStackTrace();
          }
          
          FileWriter writerTestToTrainDistance = new FileWriter(fileTestToTrainDistance);
          
          if(isDerivative(this.measuresToUse[c])){
            
            for(int i = 0; i < this.derTest.numInstances(); i++){
                writerTestToTrainDistance.write(String.valueOf((int)this.derTest.instance(i).classValue()));
                for(int j = 0; j < this.derTrain.numInstances(); j++){
                    writerTestToTrainDistance.write(","+ String.valueOf(ms.distance(this.derTest.instance(i),this.derTrain.instance(j))));
                }
                writerTestToTrainDistance.write("\n");
            }
          }else{
            for(int i = 0; i < this.test.numInstances(); i++){
                writerTestToTrainDistance.write(String.valueOf((int)this.test.instance(i).classValue()));
                for(int j = 0; j < this.train.numInstances(); j++){
                    writerTestToTrainDistance.write(","+ String.valueOf(ms.distance(this.test.instance(i),this.train.instance(j))));
                }
                writerTestToTrainDistance.write("\n");
            }
          }
          writerTestToTrainDistance.close();
          
      }
    }
    
    

    public static void  main(String[] args)throws FileNotFoundException, IOException, Exception{
        
        String outputFileDirStr = "/home/dy1n16/TSCResults";
        String dataDirStr = "/home/dy1n16/NewTSCProblems";
        String datasetName = "GunPoint";
        
        DistancesFileOutput dfo = new DistancesFileOutput(outputFileDirStr, datasetName);
        
        
        Instances train = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TRAIN");
        Instances test = ClassifierTools.loadData(dataDirStr+"/"+datasetName+"/"+datasetName+"_TEST");
        dfo.distancesCalculateAndWriteToFile(train, test);
       
     
    }
    
    

}