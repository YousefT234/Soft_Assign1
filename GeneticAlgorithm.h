#pragma once
#include <bits/stdc++.h>
#include "Types.h"
#include "Interfaces.h"
using namespace std;

template <typename T>
class GeneticAlgorithm {
public:
    using FitnessFunc = function<double(const vector<T>&)>;
    using FeasibilityFunc = function<bool(const vector<T>&)>;
    using GeneInitializerFunc = function<vector<T>(int, mt19937&)>;

private:
    GAParameters params;
    FitnessFunc fitness_fn;
    FeasibilityFunc feasibility_fn;
    GeneInitializerFunc init_fn;

    shared_ptr<ISelection<T>> selection_op;
    shared_ptr<ICrossover<T>> crossover_op;
    shared_ptr<IMutation<T>> mutation_op;
    shared_ptr<IReplacement<T>> replacement_op;

    mt19937 rng;
    vector<Individual<T>> population;
    Individual<T> best_solution;

public:
    GeneticAlgorithm(GAParameters p,
                     FitnessFunc fit_f,
                     FeasibilityFunc feas_f,
                     GeneInitializerFunc init_f)
        : params(p), fitness_fn(move(fit_f)), feasibility_fn(move(feas_f)), init_fn(move(init_f)) {
        random_device rd;
        rng.seed(rd());
    }

    void set_parameters(const GAParameters& p) { params = p; }
    GAParameters get_parameters() const { return params; }

    void set_selection(shared_ptr<ISelection<T>> op) { selection_op = move(op); }
    void set_crossover(shared_ptr<ICrossover<T>> op) { crossover_op = move(op); }
    void set_mutation(shared_ptr<IMutation<T>> op) { mutation_op = move(op); }
    void set_replacement(shared_ptr<IReplacement<T>> op) { replacement_op = move(op); }

    void evaluate(Individual<T>& ind) {
        ind.is_feasible = feasibility_fn(ind.genes);
        ind.raw_fitness = fitness_fn(ind.genes);

        if (!ind.is_feasible) {
            if (params.opt_type == OptimizationType::MINIMIZE) {
                ind.fitness = ind.raw_fitness + params.penalty_factor;
            } else {
                ind.fitness = ind.raw_fitness - params.penalty_factor;
            }
        } else {
            ind.fitness = ind.raw_fitness;
        }
    }

    void initialize_population() {
        population.resize(params.pop_size);
        for (auto& ind : population) {
            ind.genes = init_fn(params.chromosome_length, rng);
            evaluate(ind);
        }
        best_solution = population[0];
        for (const auto& ind : population) {
            update_best(ind);
        }
    }

    void update_best(const Individual<T>& ind) {
        if (params.opt_type == OptimizationType::MINIMIZE) {
            if (ind.fitness < best_solution.fitness) {
                best_solution = ind;
            }
        } else {
            if (ind.fitness > best_solution.fitness) {
                best_solution = ind;
            }
        }
    }

    void step() {
        vector<Individual<T>> offspring_pop;
        offspring_pop.reserve(params.pop_size);

        while (offspring_pop.size() < params.pop_size) {
            Individual<T> parent1 = selection_op->select(population, params.opt_type, rng);
            Individual<T> parent2 = selection_op->select(population, params.opt_type, rng);

            auto [child1, child2] = crossover_op->crossover(
                parent1, parent2, params.crossover_rate, rng);

            mutation_op->mutate(child1, params.mutation_rate, rng);
            mutation_op->mutate(child2, params.mutation_rate, rng);

            evaluate(child1);
            evaluate(child2);

            offspring_pop.push_back(move(child1));
            if (offspring_pop.size() < params.pop_size) {
                offspring_pop.push_back(move(child2));
            }
        }

        population = replacement_op->replace(population, offspring_pop, params.pop_size, params.opt_type);

        for (const auto& ind : population) {
            update_best(ind);
        }
    }

    void run(bool verbose = false) {
        initialize_population();
        for (int gen = 0; gen < params.max_generations; ++gen) {
            step();
            if (verbose && (gen == 0 || (gen + 1) % 10 == 0 || gen + 1 == params.max_generations)) {
                cout << "Generation " << (gen + 1)
                     << " | Best Fitness: " << best_solution.fitness
                     << " | Raw: " << best_solution.raw_fitness
                     << " | Feasible: " << (best_solution.is_feasible ? "Yes" : "No")
                     << "\n";
            }
        }
    }

    Individual<T> get_best_solution() const { return best_solution; }
    const vector<Individual<T>>& get_population() const { return population; }
};
