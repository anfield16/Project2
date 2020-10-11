package opt.test;

import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
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
public class OAccuracyMaxKColoringTest {
    /** The n value */
    private static final int N = 50; // number of vertices
    //private static final int L =4; // L adjacent nodes per vertex
    private static final int L =6; // TODO CHANGE TO A LARGER L
    private static final int K = 8; // K possible colors
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
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

        List<Integer> iterationList = new ArrayList<>(Arrays.asList(10, 100,200,500,1000,2000,4000,10000, 20000,                                                                     40000, 60000, 100000));
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        for (int i : iterationList) {
            starttime = System.currentTimeMillis();
            fit = new FixedIterationTrainer(rhc, i);
            fit.train();
            //System.out.println("RHC: " + ef.value(rhc.getOptimal()));
            ef.value(rhc.getOptimal());
            if (ef.foundConflict().equalsIgnoreCase("Failed to find Max-K Color combination !")) {
                System.out.print("Not Found");
            }
            else System.out.print("Found");
            //System.out.println("Time : " + (System.currentTimeMillis() - starttime));
        }
        System.out.println();
        System.out.println("============================");

        for (int i : iterationList) {
            starttime = System.currentTimeMillis();
            SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .1, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            ef.value(sa.getOptimal());
            if (ef.foundConflict().equalsIgnoreCase("Failed to find Max-K Color combination !")) {
                System.out.print("Not Found");
            }
            else System.out.print("Found");
            //System.out.println("Time : " + (System.currentTimeMillis() - starttime));
        }
        System.out.println();
        System.out.println("============================");

        List<Integer> galist = new ArrayList<>(Arrays.asList(1,2,5,10,15,20,30,40,50,80,100));
        List<String> gaResult = new ArrayList<>();
        List<Long> gaTime = new ArrayList<>();
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
        System.out.println(galist);
        System.out.println(gaResult);
        System.out.println(gaTime);

        System.out.println("============================");

        List<Integer> mimicList = new ArrayList<>(Arrays.asList(1,2,5,10,15,20,30,40,50,80,100));
        List<String> mimicResult = new ArrayList<>();
        List<Long> mimicTime = new ArrayList<>();
        for (int i : mimicList) {
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
        System.out.println(mimicList);
        System.out.println(mimicResult);
        System.out.println(mimicTime);
    }
}
