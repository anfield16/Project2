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
public class OGeneticCountOnesTest {
    /** The n value */
    private static final int N = 200;
    
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

        List<Integer> iterationList = new ArrayList<>(Arrays.asList(20,30, 40, 80, 100, 150, 200,
                                                                    500, 1000, 1500, 2000, 2500, 3000));
        FixedIterationTrainer fit;

        System.out.print("population size,");
        for (int i : iterationList) {
            System.out.print(i + ",");
        }
        System.out.println();
        System.out.print("population,");
        for (int i:iterationList) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(i, 20, 0, gap);
            fit = new FixedIterationTrainer(ga, 150);
            fit.train();
            System.out.print(ef.value(ga.getOptimal()) + ",");
        }
        System.out.println();

        List<Integer> toMateList = new ArrayList<>(Arrays.asList(1,2,3,5,20,30,40,50,60,70,80));
        System.out.print("toMate value,");
        for (int i : toMateList) {
            System.out.print(i + ",");
        }
        System.out.println();
        System.out.print("toMate,");
        for (int i : toMateList) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(80, i, 0, gap);
            fit = new FixedIterationTrainer(ga, 150);
            fit.train();
            System.out.print(ef.value(ga.getOptimal()) + ",");
        }
        System.out.println();

        List<Integer> toMutateList = new ArrayList<>(Arrays.asList(1,2,3,5,20,30,40,50,60,70,80,100,120));
        System.out.print("toMutate value,");
        for (int i : toMutateList) {
            System.out.print(i + ",");
        }
        System.out.println();
        System.out.print("toMutate,");
        for (int i : toMutateList) {
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(80, 40, i, gap);
            fit = new FixedIterationTrainer(ga, 150);
            fit.train();
            System.out.print(ef.value(ga.getOptimal()) + ",");
        }
        System.out.println();
    }
}