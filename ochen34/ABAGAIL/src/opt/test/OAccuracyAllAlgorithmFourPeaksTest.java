package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.FourPeaksEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.*;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class OAccuracyAllAlgorithmFourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 4;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


        long starttime;
        FixedIterationTrainer fit;
        Map<String, Long> execTime = new HashMap<>();
        List<Long> runTime = new ArrayList<>();
        List<Integer> iterationList = new ArrayList<>(Arrays.asList(10, 100,500,1000,4000,7000,10000,15000, 20000, 40000, 60000, 80000));
        System.out.print("Iterations, ");
        for (int i : iterationList) System.out.print(i + ", ");
        System.out.println();
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        System.out.print("RHC,");
        for (int i : iterationList) {
            starttime = System.currentTimeMillis();
            fit = new FixedIterationTrainer(rhc, i);
            fit.train();
            System.out.print(ef.value(rhc.getOptimal()) + ", ");
            if (i == 500) execTime.put("RHC", System.currentTimeMillis() - starttime);
            runTime.add(System.currentTimeMillis() - starttime);
        }
        System.out.println();
        System.out.print("RHC_time, ");
        for (Long t : runTime) System.out.print(t + ", ");
        System.out.println();

        runTime.clear();
        SimulatedAnnealing sa = new SimulatedAnnealing(120, .35, hcp);
        System.out.print("SA,");
        for (int i : iterationList) {
            starttime = System.currentTimeMillis();
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal()) + ", ");
            if (i == 500) execTime.put("SA", System.currentTimeMillis() - starttime);
            runTime.add(System.currentTimeMillis() - starttime);
        }
        System.out.println();
        System.out.print("SA_time, ");
        for (Long t : runTime) System.out.print(t + ", ");

        List<Integer> gaList = new ArrayList<>(Arrays.asList(2, 10, 50, 100,200,300, 500,1000,2000, 3000,4000,5000));
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        System.out.print("GA,");
        for (int i : iterationList) {
            starttime = System.currentTimeMillis();
            fit = new FixedIterationTrainer(ga, i);
            fit.train();
            System.out.print(ef.value(ga.getOptimal()) + ", ");
            if (i == 500) execTime.put("GA", System.currentTimeMillis() - starttime);
        }
        System.out.println();


        MIMIC mimic = new MIMIC(200, 20, pop);
        System.out.print("MIMIC,");
        for (int i : gaList) {
            if (i<=1000) {
                starttime = System.currentTimeMillis();
                fit = new FixedIterationTrainer(mimic, i);
                fit.train();
                System.out.print(ef.value(mimic.getOptimal()) + ", ");
                if (i == 500) execTime.put("MIMIC", System.currentTimeMillis() - starttime);
            }
        }
        System.out.println();
        System.out.println(execTime);
    }
}
