#pragma once
#include <bits/stdc++.h>
#include "Types.h"
using namespace std;

template <typename T>
class ISelection {
public:
    virtual ~ISelection() = default;
    virtual Individual<T> select(const vector<Individual<T>>& population, OptimizationType opt_type, mt19937& rng) = 0;
};

template <typename T>
class ICrossover {
public:
    virtual ~ICrossover() = default;
    virtual pair<Individual<T>, Individual<T>> crossover(
        const Individual<T>& parent1,
        const Individual<T>& parent2,
        double pc,
        mt19937& rng) = 0;
};

template <typename T>
class IMutation {
public:
    virtual ~IMutation() = default;
    virtual void mutate(Individual<T>& ind, double pm, mt19937& rng) = 0;
};

template <typename T>
class IReplacement {
public:
    virtual ~IReplacement() = default;
    virtual vector<Individual<T>> replace(
        const vector<Individual<T>>& current_pop,
        const vector<Individual<T>>& offspring_pop,
        int pop_size,
        OptimizationType opt_type) = 0;
};
