package opt.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class OLetterRecogBackPropAbaloneTest {
    private static int rows = 2000;
    private static Instance[] instances= initializeInstances();

    private static int inputLayer = 16, hiddenLayer = 10, outputLayer = 26, trainingIterations = 20000;

    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {

        //instances = initializeInstances();

        for(int i = 0; i < oa.length-2; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        //oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length-2; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i]); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double[] predicted = new double[26];
            double[] actual = new double[26];
            start = System.nanoTime();
            for(int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                String label = instances[i].getLabel().toString();
                String[] labelArr = label.split(",");
                for (int k = 0; k<26; k++) {
                    predicted[k] = Double.parseDouble(labelArr[k]);
                }

                String pred = networks[i].getOutputValues().toString();
                String[] predArr = pred.split(",");
                for (int k = 0; k<26; k++) {
                    actual[k] = Double.parseDouble(predArr[k]);
                }

                int largestActualIndex = -1;
                double maxVal = -1;
                for (int k = 0; k<26; k++) {
                    if (actual[k]>maxVal) {
                        maxVal = actual[k];
                        largestActualIndex = k;
                    }
                }

                if (Math.abs(predicted[largestActualIndex] - 1.0)<1e-5) {
                    correct++;
                }
                else incorrect++;
                //predicted = Double.parseDouble(instances[j].getLabel().toString());
                //actual = Double.parseDouble(networks[i].getOutputValues().toString());

                //double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;

            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct/(correct+incorrect)*100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());

                double[] labels = new double[26];
                String labelToSet = network.getOutputValues().toString();
                String[] labelToSetArr = labelToSet.split(",");
                for (int k = 0; k<26; k++) {
                    labels[k] = Double.parseDouble(labelToSetArr[k]);
                }


                //example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                example.setLabel(new Instance(labels));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[rows][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/letter-recognitionFull.txt")));

            for(int i = 0; i < rows; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[16]; // 16 attributes
                attributes[i][1] = new double[26];

                for(int j = 0; j < 16; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][(int)Double.parseDouble(scan.next())-1] = 1;
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

//        //check that read data table is correct
//        System.out.println(rows+ "======rows");
//        for (int i = 0; i < 16; i++)
//            System.out.print(attributes[168][0][i] + " ");
//        System.out.println();
//        for (int i = 0; i < 26; i++)
//            System.out.print(attributes[168][1][i] + " ");

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // set labels
            instances[i].setLabel(new Instance(attributes[i][1]));
        }

        return instances;
    }
}
