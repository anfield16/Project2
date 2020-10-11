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
public class OMIMCCountOnesTest {
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

        List<Integer> iterationList = new ArrayList<>(Arrays.asList(1,5, 10,25,50,100,150,200,300,500));
        FixedIterationTrainer fit;

        System.out.print("Iterations,");
        for (int i : iterationList) {
            System.out.print(i + ",");
        }
        System.out.println();

        System.out.print("keep_1,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(25, 1, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();

        System.out.print("keep_5,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(25, 5, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();

        System.out.print("keep_10,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(25, 10, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();

        System.out.print("keep_16,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(25, 16, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();

        System.out.print("keep_24,");
        for (int i:iterationList) {
            MIMIC mimic = new MIMIC(25, 24, pop);
            fit = new FixedIterationTrainer(mimic, i);
            fit.train();
            System.out.print(ef.value(mimic.getOptimal()) + ",");
        }
        System.out.println();

//        List<Integer> iterationList = new ArrayList<>(Arrays.asList(1,5, 10,25,50,100,150,200,300,500));
//        FixedIterationTrainer fit;
//
//        System.out.print("Iterations,");
//        for (int i : iterationList) {
//            System.out.print(i + ",");
//        }
//        System.out.println();
//
//
//
//        System.out.print("take_11,");
//        for (int i:iterationList) {
//            MIMIC mimic = new MIMIC(11, 10, pop);
//            fit = new FixedIterationTrainer(mimic, i);
//            fit.train();
//            System.out.print(ef.value(mimic.getOptimal()) + ",");
//        }
//        System.out.println();
//
//        System.out.print("take_25,");
//        for (int i:iterationList) {
//            MIMIC mimic = new MIMIC(25, 10, pop);
//            fit = new FixedIterationTrainer(mimic, i);
//            fit.train();
//            System.out.print(ef.value(mimic.getOptimal()) + ",");
//        }
//        System.out.println();
//
//        System.out.print("take_100,");
//        for (int i:iterationList) {
//            MIMIC mimic = new MIMIC(25, 10, pop);
//            fit = new FixedIterationTrainer(mimic, i);
//            fit.train();
//            System.out.print(ef.value(mimic.getOptimal()) + ",");
//        }
//        System.out.println();
//
//        System.out.print("take_500,");
//        for (int i:iterationList) {
//            MIMIC mimic = new MIMIC(500, 10, pop);
//            fit = new FixedIterationTrainer(mimic, i);
//            fit.train();
//            System.out.print(ef.value(mimic.getOptimal()) + ",");
//        }
//        System.out.println();
    }
}