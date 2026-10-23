#pragma once
#include <bits/stdc++.h>
using namespace std;

enum class OptimizationType {
    MINIMIZE,
    MAXIMIZE
};

template <typename T>
struct Individual {
    vector<T> genes;
    double raw_fitness = 0.0;
    double fitness = 0.0;
    bool is_feasible = true;

    Individual() = default;
    explicit Individual(int length) : genes(length) {}
    explicit Individual(vector<T> g) : genes(move(g)) {}
};

struct GAParameters {
    int pop_size = 50;
    int chromosome_length = 20;
    int max_generations = 100;
    double crossover_rate = 0.8;
    double mutation_rate = 0.05;
    int elitism_count = 2;
    double penalty_factor = 1000.0;
    OptimizationType opt_type = OptimizationType::MINIMIZE;
};
