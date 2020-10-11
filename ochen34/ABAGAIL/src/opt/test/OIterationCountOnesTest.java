package opt.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.example.CountOnesEvaluationFunction;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class OIterationCountOnesTest {
    /** The n value */
    private static final int N = 80;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        List<Integer> iterationList = new ArrayList<>(Arrays.asList(1,5,10,20,50,100,150,200,500,1000,1500,2000));
        FixedIterationTrainer fit;

        System.out.print("RandomizedHillClimbing,");
        for (int i:iterationList) {
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            fit = new FixedIterationTrainer(rhc, i);
            fit.train();
            System.out.print(ef.value(rhc.getOptimal())+ ",");
        }
        System.out.println();

        System.out.print("SimulatedAnnealing,");
        for (int i:iterationList) {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal())+ ",");
        }
        System.out.println();

        System.out.print("GeneticAlgorithm,");
        for (int i:iterationList) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, i);
            fit.train();
            System.out.print(ef.value(ga.getOptimal()) + ",");
        }
        System.out.println();

        System.out.print("MIMC,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(50, 10, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();
    }
}