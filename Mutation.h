#pragma once
#include <bits/stdc++.h>
#include "Interfaces.h"
using namespace std;

template <typename T>
class BitFlipMutation : public IMutation<T> {
private:
    function<T(const T&, mt19937&)> alter_gene_fn;

public:
    BitFlipMutation() {
        alter_gene_fn = [](const T& val, mt19937&) {
            return static_cast<T>(!val);
        };
    }

    explicit BitFlipMutation(function<T(const T&, mt19937&)> fn) : alter_gene_fn(move(fn)) {}

    void mutate(Individual<T>& ind, double pm, mt19937& rng) override {
        uniform_real_distribution<double> dist(0.0, 1.0);
        for (auto& gene : ind.genes) {
            if (dist(rng) < pm) {
                gene = alter_gene_fn(gene, rng);
            }
        }
    }
};

template <typename T>
class SwapMutation : public IMutation<T> {
public:
    void mutate(Individual<T>& ind, double pm, mt19937& rng) override {
        uniform_real_distribution<double> p_dist(0.0, 1.0);
        if (p_dist(rng) < pm && ind.genes.size() >= 2) {
            uniform_int_distribution<int> dist(0, ind.genes.size() - 1);
            int idx1 = dist(rng);
            int idx2 = dist(rng);
            while (idx1 == idx2) {
                idx2 = dist(rng);
            }
            swap(ind.genes[idx1], ind.genes[idx2]);
        }
    }
};

template <typename T>
class InversionMutation : public IMutation<T> {
public:
    void mutate(Individual<T>& ind, double pm, mt19937& rng) override {
        uniform_real_distribution<double> p_dist(0.0, 1.0);
        if (p_dist(rng) < pm && ind.genes.size() >= 2) {
            uniform_int_distribution<int> dist(0, ind.genes.size() - 1);
            int idx1 = dist(rng);
            int idx2 = dist(rng);
            if (idx1 > idx2) swap(idx1, idx2);
            reverse(ind.genes.begin() + idx1, ind.genes.begin() + idx2 + 1);
        }
    }
};

template <typename T>
class UniformFPMutation : public IMutation<T> {
private:
    T lower_bound;
    T upper_bound;

public:
    UniformFPMutation(T lb, T ub) : lower_bound(lb), upper_bound(ub) {}

    void mutate(Individual<T>& ind, double pm, mt19937& rng) override {
        uniform_real_distribution<double> p_dist(0.0, 1.0);
        uniform_real_distribution<double> val_dist(static_cast<double>(lower_bound), static_cast<double>(upper_bound));

        for (auto& gene : ind.genes) {
            if (p_dist(rng) < pm) {
                gene = static_cast<T>(val_dist(rng));
            }
        }
    }
};
