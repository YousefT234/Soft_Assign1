#pragma once
#include <bits/stdc++.h>
#include "Interfaces.h"
using namespace std;

template <typename T>
class ElitistGenerationalReplacement : public IReplacement<T> {
private:
    int elitism_count;

public:
    explicit ElitistGenerationalReplacement(int elite = 2) : elitism_count(elite) {}

    vector<Individual<T>> replace(
        const vector<Individual<T>>& current_pop,
        const vector<Individual<T>>& offspring_pop,
        int pop_size,
        OptimizationType opt_type) override {

        vector<Individual<T>> sorted_parents = current_pop;
        auto comp = [opt_type](const Individual<T>& a, const Individual<T>& b) {
            if (opt_type == OptimizationType::MINIMIZE) return a.fitness < b.fitness;
            return a.fitness > b.fitness;
        };
        sort(sorted_parents.begin(), sorted_parents.end(), comp);

        vector<Individual<T>> new_pop;
        new_pop.reserve(pop_size);

        int num_elites = min(elitism_count, static_cast<int>(sorted_parents.size()));
        for (int i = 0; i < num_elites && new_pop.size() < pop_size; ++i) {
            new_pop.push_back(sorted_parents[i]);
        }

        for (int i = 0; i < offspring_pop.size() && new_pop.size() < pop_size; ++i) {
            new_pop.push_back(offspring_pop[i]);
        }

        for (int i = num_elites; i < sorted_parents.size() && new_pop.size() < pop_size; ++i) {
            new_pop.push_back(sorted_parents[i]);
        }

        return new_pop;
    }
};

template <typename T>
class SteadyStateReplacement : public IReplacement<T> {
public:
    vector<Individual<T>> replace(
        const vector<Individual<T>>& current_pop,
        const vector<Individual<T>>& offspring_pop,
        int pop_size,
        OptimizationType opt_type) override {

        vector<Individual<T>> merged;
        merged.reserve(current_pop.size() + offspring_pop.size());
        merged.insert(merged.end(), current_pop.begin(), current_pop.end());
        merged.insert(merged.end(), offspring_pop.begin(), offspring_pop.end());

        auto comp = [opt_type](const Individual<T>& a, const Individual<T>& b) {
            if (opt_type == OptimizationType::MINIMIZE) return a.fitness < b.fitness;
            return a.fitness > b.fitness;
        };
        sort(merged.begin(), merged.end(), comp);

        if (merged.size() > pop_size) {
            merged.resize(pop_size);
        }
        return merged;
    }
};
