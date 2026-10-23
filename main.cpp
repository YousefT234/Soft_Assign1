#include <bits/stdc++.h>
using namespace std;

#include "Types.h"
#include "Interfaces.h"
#include "Selection.h"
#include "Crossover.h"
#include "Mutation.h"
#include "Replacement.h"
#include "GeneticAlgorithm.h"

int main() {
    {
        cout << "=== Running Binary Optimization (OneMax) ===" << "\n";
        GAParameters params;
        params.pop_size = 40;
        params.chromosome_length = 30;
        params.max_generations = 50;
        params.crossover_rate = 0.85;
        params.mutation_rate = 0.02;
        params.elitism_count = 2;
        params.opt_type = OptimizationType::MAXIMIZE;

        auto fitness = [](const vector<int>& genes) {
            return static_cast<double>(accumulate(genes.begin(), genes.end(), 0));
        };

        auto feasibility = [](const vector<int>&) {
            return true;
        };

        auto init = [](int length, mt19937& rng) {
            uniform_int_distribution<int> dist(0, 1);
            vector<int> genes(length);
            for (auto& g : genes) g = dist(rng);
            return genes;
        };

        GeneticAlgorithm<int> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<TournamentSelection<int>>(3));
        ga.set_crossover(make_shared<MultiPointCrossover<int>>(2));
        ga.set_mutation(make_shared<BitFlipMutation<int>>([](const int& v, mt19937&) { return 1 - v; }));
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<int>>(2));

        ga.run(true);
        cout << "Binary Best Raw Score: " << ga.get_best_solution().raw_fitness << "\n\n";
    }

    {
        cout << "=== Running Integer Permutation Optimization (Order Crossover & Swap) ===" << "\n";
        int n = 10;
        GAParameters params;
        params.pop_size = 50;
        params.chromosome_length = n;
        params.max_generations = 100;
        params.crossover_rate = 0.9;
        params.mutation_rate = 0.2;
        params.opt_type = OptimizationType::MINIMIZE;

        auto fitness = [](const vector<int>& genes) {
            double cost = 0.0;
            for (int i = 0; i < static_cast<int>(genes.size()); ++i) {
                cost += abs(genes[i] - i);
            }
            return cost;
        };

        auto feasibility = [](const vector<int>& genes) {
            unordered_set<int> seen(genes.begin(), genes.end());
            return seen.size() == genes.size();
        };

        auto init = [](int length, mt19937& rng) {
            vector<int> genes(length);
            iota(genes.begin(), genes.end(), 0);
            shuffle(genes.begin(), genes.end(), rng);
            return genes;
        };

        GeneticAlgorithm<int> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<RouletteWheelSelection<int>>());
        ga.set_crossover(make_shared<OrderCrossover<int>>());
        ga.set_mutation(make_shared<SwapMutation<int>>());
        ga.set_replacement(make_shared<SteadyStateReplacement<int>>());

        ga.run(true);
        cout << "Integer Permutation Best Distance to Sorted: " << ga.get_best_solution().raw_fitness << "\n\n";
    }

    {
        cout << "=== Running Floating-Point Optimization (Sphere Function) ===" << "\n";
        GAParameters params;
        params.pop_size = 60;
        params.chromosome_length = 5;
        params.max_generations = 80;
        params.crossover_rate = 0.8;
        params.mutation_rate = 0.1;
        params.opt_type = OptimizationType::MINIMIZE;

        auto fitness = [](const vector<double>& genes) {
            double sum = 0.0;
            for (double x : genes) sum += x * x;
            return sum;
        };

        auto feasibility = [](const vector<double>& genes) {
            for (double x : genes) {
                if (x < -5.12 || x > 5.12) return false;
            }
            return true;
        };

        auto init = [](int length, mt19937& rng) {
            uniform_real_distribution<double> dist(-5.12, 5.12);
            vector<double> genes(length);
            for (auto& g : genes) g = dist(rng);
            return genes;
        };

        GeneticAlgorithm<double> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<TournamentSelection<double>>(2));
        ga.set_crossover(make_shared<MultiPointCrossover<double>>(1));
        ga.set_mutation(make_shared<UniformFPMutation<double>>(-5.12, 5.12));
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<double>>(2));

        ga.run(true);
        cout << "Floating Point Best Sphere Cost: " << ga.get_best_solution().raw_fitness << "\n";
    }

    return 0;
}
