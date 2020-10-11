package opt.test;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.*;
import shared.tester.*;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Based on the XORTest test class, this class uses a standard FeedForwardNetwork
 * and various optimization problems.
 * 
 * See numbered explanations for what each piece of the method does to address
 * the neural network optimization problem.
 * 
 * @author Jesse Rosalia <https://github.com/theJenix>
 * @date 2013-03-05
 */
public class OEvalXORTestNoBackpropSimAnneal {
    private static int rows = 1000;
    private static double[][][] data10;  // this array will store letter_recognition data

    /**
     * Tests out the perceptron with the classic xor test
     * @param args ignored
     */
    public static void main(String[] args) {
        // 1) Construct data instances for training.  These will also be run
        //    through the network at the bottom to verify the output

        data10 = new double[rows][2][];
        for (int i = 0; i<rows; i++) {
            data10[i][0] = new double[16];
            data10[i][1] = new double[26];
        }

        readCSV();

        for (int i = 0; i < 16; i++)
            System.out.print(data10[188][0][i] + " ");
        System.out.println();
        for (int i = 0; i < 26; i++)
            System.out.print(data10[188][1][i] + " ");

//        //code to verify readCSV
//        for (int i=0; i<rows; i++) {
//            for (int j=0; j<2; j++) {
//                if (j == 0) {
//                    for (int k = 0; k<16; k++) {
//                        System.out.print(data10[i][j][k] + " ");
//                    }
//                    System.out.print(" -label= ");
//                }
//                else {
//                    System.out.println(data10[i][j][0]);
//                }
//            }
//
//        }


//        int[] labels = { 0, 1 };
//        double[][][] data = {
//               { { 1, 1, 1, 1 }, { 0 } },
//               { { 1, 0, 1, 0 }, { 1 } },
//               { { 0, 1, 0, 1 }, { 1 } },
//               { { 0, 0, 0, 0 }, { 0 } }
//        };
        Instance[] patterns = new Instance[data10.length];
        //Instance[] patterns = new Instance[4];
        for (int i = 0; i < patterns.length; i++) {
            patterns[i] = new Instance(data10[i][0]);
            patterns[i].setLabel(new Instance(data10[i][1]));
        }

        // 2) Instantiate a network using the FeedForwardNeuralNetworkFactory.  This network
        //    will be our classifier.
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        // 2a) These numbers correspond to the number of nodes in each layer.
        //     This network has 4 input nodes, 3 hidden nodes in 1 layer, and 1 output node in the output layer.
        //FeedForwardNetwork network = factory.createClassificationNetwork(new int[] { 4, 3, 1 });
        FeedForwardNetwork network = factory.createClassificationNetwork(new int[] { 16,10,26 });

        // 3) Instantiate a measure, which is used to evaluate each possible set of weights.
        ErrorMeasure measure = new SumOfSquaresError();

        // 4) Instantiate a DataSet, which adapts a set of instances to the optimization problem.
        DataSet set = new DataSet(patterns);

        // 5) Instantiate an optimization problem, which is used to specify the dataset, evaluation
        //    function, mutator and crossover function (for Genetic Algorithms), and any other
        //    parameters used in optimization.
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(
            set, network, measure);

        // 6) Instantiate a specific OptimizationAlgorithm, which defines how we pick our next potential
        //    hypothesis. Simulated Annealing is used with the default Weka values.
        OptimizationAlgorithm o = new SimulatedAnnealing(100, 0.5, nno);

        // 7) Instantiate a trainer.  The FixtIterationTrainer takes another trainer (in this case,
        //    an OptimizationAlgorithm) and executes it a specified number of times.
        FixedIterationTrainer fit = new FixedIterationTrainer(o, 60000);

        // 8) Run the trainer.  This may take a little while to run, depending on the OptimizationAlgorithm,
        //    size of the data, and number of iterations.
        fit.train();


        // 9) Once training is done, get the optimal solution from the OptimizationAlgorithm.  These are the
        //    optimal weights found for this network.
        Instance opt = o.getOptimal();
        network.setWeights(opt.getData());

        //10) Run the training data through the network with the weights discovered through optimization, and
        //    print out the expected label and result of the classifier for each instance.
        for (int i = 0; i < patterns.length; i++) {
            network.setInputValues(patterns[i].getData());
            network.run();
            System.out.println("~~");
            System.out.println(patterns[i].getLabel());
            System.out.println(network.getOutputValues());
        }
    }

    public static void readCSV() {
        String csvFile = "letter-recognition5000.csv";
        BufferedReader br = null;
        String line = "";
        String cvsSplitBy = ",";

        try {

            br = new BufferedReader(new FileReader(csvFile));

            for (int j = 0; j < rows; j++) {
                line = br.readLine();
                // use comma as separator
                String[] values = line.split(cvsSplitBy);
//                for (String s:values) {
//                    System.out.print(s+ " ");
//                }
                for (int i = 0; i < values.length-1; i++) {
                    data10[j][0][i] = Double.valueOf(values[i]);
                }
                int classLabel = (int)Math.round(Double.valueOf(values[values.length - 1]));
                data10[j][1][classLabel-1] = 1.0;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}


