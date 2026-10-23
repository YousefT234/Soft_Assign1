#pragma once
#include <bits/stdc++.h>
#include "Interfaces.h"
using namespace std;

template <typename T>
class TournamentSelection : public ISelection<T> {
private:
    int tournament_size;

public:
    explicit TournamentSelection(int k = 3) : tournament_size(max<int>(1, k)) {}

    Individual<T> select(const vector<Individual<T>>& population, OptimizationType opt_type, mt19937& rng) override {
        uniform_int_distribution<int> dist(0, population.size() - 1);
        int best_idx = dist(rng);

        for (int i = 1; i < tournament_size; ++i) {
            int idx = dist(rng);
            if (opt_type == OptimizationType::MINIMIZE) {
                if (population[idx].fitness < population[best_idx].fitness) {
                    best_idx = idx;
                }
            } else {
                if (population[idx].fitness > population[best_idx].fitness) {
                    best_idx = idx;
                }
            }
        }
        return population[best_idx];
    }
};

template <typename T>
class RouletteWheelSelection : public ISelection<T> {
public:
    Individual<T> select(const vector<Individual<T>>& population, OptimizationType opt_type, mt19937& rng) override {
        int n = population.size();
        vector<double> scores(n);
        double min_f = population[0].fitness;
        double max_f = population[0].fitness;

        for (const auto& ind : population) {
            min_f = min(min_f, ind.fitness);
            max_f = max(max_f, ind.fitness);
        }

        double sum_scores = 0.0;
        for (int i = 0; i < n; ++i) {
            if (opt_type == OptimizationType::MAXIMIZE) {
                scores[i] = population[i].fitness - min_f + 1e-6;
            } else {
                scores[i] = max_f - population[i].fitness + 1e-6;
            }
            sum_scores += scores[i];
        }

        uniform_real_distribution<double> dist(0.0, sum_scores);
        double r = dist(rng);
        double cumulative = 0.0;

        for (int i = 0; i < n; ++i) {
            cumulative += scores[i];
            if (cumulative >= r) {
                return population[i];
            }
        }
        return population.back();
    }
};
