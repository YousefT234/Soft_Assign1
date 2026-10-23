#pragma once
#include <bits/stdc++.h>
#include "Interfaces.h"
using namespace std;

template <typename T>
class MultiPointCrossover : public ICrossover<T> {
private:
    int num_points;

public:
    explicit MultiPointCrossover(int points = 2) : num_points(max<int>(1, points)) {}

    pair<Individual<T>, Individual<T>> crossover(
        const Individual<T>& parent1,
        const Individual<T>& parent2,
        double pc,
        mt19937& rng) override {

        uniform_real_distribution<double> p_dist(0.0, 1.0);
        if (p_dist(rng) > pc || parent1.genes.size() < 2) {
            return {parent1, parent2};
        }

        int len = parent1.genes.size();
        int actual_points = min(num_points, len - 1);

        vector<int> cuts;
        cuts.reserve(actual_points);
        uniform_int_distribution<int> idx_dist(1, len - 1);

        while (cuts.size() < actual_points) {
            int pt = idx_dist(rng);
            if (find(cuts.begin(), cuts.end(), pt) == cuts.end()) {
                cuts.push_back(pt);
            }
        }
        sort(cuts.begin(), cuts.end());
        cuts.push_back(len);

        Individual<T> child1(len);
        Individual<T> child2(len);

        int start = 0;
        bool swap_parents = false;

        for (int pt : cuts) {
            for (int i = start; i < pt; ++i) {
                if (!swap_parents) {
                    child1.genes[i] = parent1.genes[i];
                    child2.genes[i] = parent2.genes[i];
                } else {
                    child1.genes[i] = parent2.genes[i];
                    child2.genes[i] = parent1.genes[i];
                }
            }
            start = pt;
            swap_parents = !swap_parents;
        }

        return {child1, child2};
    }
};

template <typename T>
class OrderCrossover : public ICrossover<T> {
public:
    pair<Individual<T>, Individual<T>> crossover(
        const Individual<T>& parent1,
        const Individual<T>& parent2,
        double pc,
        mt19937& rng) override {

        uniform_real_distribution<double> p_dist(0.0, 1.0);
        int len = parent1.genes.size();
        if (p_dist(rng) > pc || len < 2) {
            return {parent1, parent2};
        }

        uniform_int_distribution<int> dist(0, len - 1);
        int pt1 = dist(rng);
        int pt2 = dist(rng);
        if (pt1 > pt2) swap(pt1, pt2);

        auto create_ox_child = [&](const Individual<T>& p1, const Individual<T>& p2) {
            Individual<T> child(len);
            unordered_set<T> copied;

            for (int i = pt1; i <= pt2; ++i) {
                child.genes[i] = p1.genes[i];
                copied.insert(p1.genes[i]);
            }

            int current_idx = (pt2 + 1) % len;
            for (int i = 0; i < len; ++i) {
                int p2_idx = (pt2 + 1 + i) % len;
                T gene = p2.genes[p2_idx];
                if (copied.find(gene) == copied.end()) {
                    child.genes[current_idx] = gene;
                    copied.insert(gene);
                    current_idx = (current_idx + 1) % len;
                }
            }
            return child;
        };

        return {create_ox_child(parent1, parent2), create_ox_child(parent2, parent1)};
    }
};
