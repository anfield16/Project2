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
public class OAnnealCountOnesTest {
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

        System.out.print("cool0.95,");
        for (int i : iterationList) {
            System.out.print(i + ",");
        }
        System.out.println();
        System.out.print("sa_cool0.95,");
        for (int i:iterationList) {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, 0.95, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal())+ ",");
        }
        System.out.println();

        System.out.print("sa_cool0.1,");
        for (int i:iterationList) {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, 0.1, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal())+ ",");
        }
        System.out.println();

        System.out.print("sa_cool0.01,");
        for (int i:iterationList) {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, 0.01, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal())+ ",");
        }
        System.out.println();

        System.out.print("sa_cool0.001,");
        for (int i:iterationList) {
            SimulatedAnnealing sa = new SimulatedAnnealing(100, 0.001, hcp);
            fit = new FixedIterationTrainer(sa, i);
            fit.train();
            System.out.print(ef.value(sa.getOptimal())+ ",");
        }
        System.out.println();
    }
}