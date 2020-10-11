package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.Distribution;
import opt.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 
 * @author kmandal
 * @version 1.0
 */
public class OColorPatternLMaxKColoringTest {
    /** The n value */
    private static final int N = 100; // number of vertices
    private static final int L =4; // L adjacent nodes per vertex
    //private static final int L =6;
    private static final int K = 8; // K possible colors
    private static List<String> gaResult = new ArrayList<>();
    private static List<Long> gaTime = new ArrayList<>();
    private static List<String> mimicResult = new ArrayList<>();
    private static List<Long> mimicTime = new ArrayList<>();

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] l = {1,2,3,4,5,6,7};
        System.out.print("adjacent_nodes, ");
        for (int num : l) {
            System.out.print(num + ", ");
            util(50, num, 8);
        }
        System.out.println();

        //now print GA result
        System.out.print("GA_solution, ");
        for (int i = 0; i < l.length; i++) {
            System.out.print(gaResult.get(i) + ", ");
        }
        System.out.println();

        //now print GA time
        System.out.print("GA_exec_tm, ");
        for (int i = 0; i < l.length; i++) {
            System.out.print(gaTime.get(i) + ", ");
        }
        System.out.println();

        //now print MIMIC result
        System.out.print("MIMIC_solution, ");
        for (int i = 0; i < l.length; i++) {
            System.out.print(mimicResult.get(i) + ", ");
        }
        System.out.println();

        //now print MIMIC time
        System.out.print("MIMIC_exec_tm, ");
        for (int i = 0; i < l.length; i++) {
            System.out.print(mimicTime.get(i) + ", ");
        }
        System.out.println();
    }

    public static void util(int N, int L, int K) {
        Random random = new Random(N*L);
        // create the random velocity
        Vertex[] vertices = new Vertex[N];
        for (int i = 0; i < N; i++) {
            Vertex vertex = new Vertex();
            vertices[i] = vertex;
            vertex.setAdjMatrixSize(L);
            for(int j = 0; j <L; j++ ){
                vertex.getAadjacencyColorMatrix().add(random.nextInt(N*L));
            }
        }
        /*for (int i = 0; i < N; i++) {
            Vertex vertex = vertices[i];
            System.out.println(Arrays.toString(vertex.getAadjacencyColorMatrix().toArray()));
        }*/
        // for rhc, sa, and ga we use a permutation based encoding
        MaxKColorFitnessFunction ef = new MaxKColorFitnessFunction(vertices);
        Distribution odd = new DiscretePermutationDistribution(K);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new SingleCrossOver();
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);

        Distribution df = new DiscreteDependencyTree(.1);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        long starttime;
        FixedIterationTrainer fit;

        //List<Integer> galist = new ArrayList<>(Arrays.asList(1,2,5,10,15,20,30,40,50,80,100));
        List<Integer> galist = new ArrayList<>(Arrays.asList(50));

        for (int i : galist) {
            starttime = System.currentTimeMillis();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 10, 60, gap);
            fit = new FixedIterationTrainer(ga, i);
            fit.train();
            //System.out.println("GA: " + ef.value(ga.getOptimal()));
            ef.value(ga.getOptimal());
            if (ef.foundConflict().equalsIgnoreCase("Failed to find Max-K Color combination !")) {
                gaResult.add("Not_Found");
            }
            else gaResult.add("Found");
            gaTime.add(System.currentTimeMillis() - starttime);
        }
        //System.out.println(galist);
//        System.out.println(gaResult);
//        System.out.println(gaTime);


        List<Integer> mimicList = new ArrayList<>(Arrays.asList(1,2,5,10,15,20,30,40,50,80,100));
        for (int i : galist) {
            starttime = System.currentTimeMillis();
            MIMIC mimic = new MIMIC(200, 100, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            //System.out.println("GA: " + ef.value(ga.getOptimal()));
            ef.value(mimic.getOptimal());
            if (ef.foundConflict().equalsIgnoreCase("Failed to find Max-K Color combination !")) {
                mimicResult.add("Not_Found");
            }
            else mimicResult.add("Found");
            mimicTime.add(System.currentTimeMillis() - starttime);
        }
        //System.out.println(mimicList);
//        System.out.println(mimicResult);
//        System.out.println(mimicTime);
    }
}
