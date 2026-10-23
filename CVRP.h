#pragma once
#include <bits/stdc++.h>
#include "Types.h"
#include "Interfaces.h"
using namespace std;

// CVRP instance, node 0 is the depot and 1..n are the customers
struct CVRPInstance {
    string name = "instance";
    int n = 0;
    int capacity = 0;
    int vehicles = 0;
    vector<int> demand;
    vector<vector<double>> dist;

    int total_demand() const { return accumulate(demand.begin(), demand.end(), 0); }

    // the example from the OR-Tools CVRP page: 16 customers, 4 vehicles, capacity 15
    static CVRPInstance ortools_example() {
        CVRPInstance I;
        I.name = "ortools-example";
        I.n = 16;
        I.capacity = 15;
        I.vehicles = 4;
        I.demand = {0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8};
        const int m[17][17] = {
            {0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662},
            {548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210},
            {776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754},
            {696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358},
            {582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244},
            {274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708},
            {502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480},
            {194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856},
            {308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514},
            {194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468},
            {536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354},
            {502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844},
            {388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730},
            {354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536},
            {468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194},
            {776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798},
            {662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0}};
        I.dist.assign(17, vector<double>(17));
        for (int i = 0; i < 17; ++i)
            for (int j = 0; j < 17; ++j) I.dist[i][j] = m[i][j];
        return I;
    }
};

// The chromosome is a permutation of the customers. decode() splits it into
// routes, so the fitness and feasibility functions work on the decoded routes.
class CVRPProblem {
public:
    struct Decoded {
        vector<vector<int>> routes;
        double cost = 0.0;
        int excess_vehicles = 0;
    };

    explicit CVRPProblem(const CVRPInstance& inst) : I(inst) {}

    Decoded decode(const vector<int>& perm) const {
        if (cache_valid && perm == cache_key) return cache_val;
        Decoded r = split(perm);
        cache_key = perm; cache_val = r; cache_valid = true;
        return r;
    }

    double fitness(const vector<int>& perm) const {
        Decoded d = decode(perm);
        return d.cost + d.excess_vehicles * EXCESS_VEHICLE_PENALTY;
    }

    bool is_feasible(const vector<int>& perm) const {
        if ((int)perm.size() != I.n) return false;
        vector<char> seen(I.n + 1, 0);
        for (int c : perm) {
            if (c < 1 || c > I.n || seen[c]) return false;
            seen[c] = 1;
        }
        Decoded d = decode(perm);
        if ((int)d.routes.size() > I.vehicles) return false;
        for (auto& route : d.routes) {
            long long load = 0;
            for (int c : route) load += I.demand[c];
            if (load > I.capacity) return false;
        }
        return true;
    }

    vector<int> init(int length, mt19937& rng) const {
        vector<int> tour(length);
        iota(tour.begin(), tour.end(), 1);
        double u = uniform_real_distribution<double>(0, 1)(rng);
        if (u < 0.2) {
            vector<int> left = tour; tour.clear();
            int cur = 0;
            while (!left.empty()) {
                sort(left.begin(), left.end(), [&](int a, int b) { return I.dist[cur][a] < I.dist[cur][b]; });
                int pick = rng() % min<int>(3, left.size());
                cur = left[pick];
                tour.push_back(cur);
                left.erase(left.begin() + pick);
            }
        } else {
            shuffle(tour.begin(), tour.end(), rng);
        }
        return tour;
    }

    const CVRPInstance& instance() const { return I; }

    static constexpr double EXCESS_VEHICLE_PENALTY = 500.0;

private:
    const CVRPInstance& I;
    mutable vector<int> cache_key;
    mutable Decoded cache_val;
    mutable bool cache_valid = false;

    // dp[k][j] = best cost of serving the first j customers using k routes
    Decoded split(const vector<int>& perm) const {
        const int n = (int)perm.size();
        const int K = min(I.vehicles, n);
        const double INF = 1e18;

        vector<double> cum(n + 1, 0.0);
        for (int j = 2; j <= n; ++j) cum[j] = cum[j - 1] + I.dist[perm[j - 2]][perm[j - 1]];

        vector<vector<double>> dp(K + 1, vector<double>(n + 1, INF));
        vector<vector<int>> pred(K + 1, vector<int>(n + 1, -1));
        dp[0][0] = 0.0;
        for (int k = 1; k <= K; ++k)
            for (int i = 0; i < n; ++i) {
                if (dp[k - 1][i] >= INF) continue;
                long long load = 0;
                for (int j = i + 1; j <= n; ++j) {
                    load += I.demand[perm[j - 1]];
                    if (load > I.capacity) break;
                    double c = dp[k - 1][i] + I.dist[0][perm[i]] + (cum[j] - cum[i + 1]) + I.dist[perm[j - 1]][0];
                    if (c < dp[k][j]) { dp[k][j] = c; pred[k][j] = i; }
                }
            }

        int bestK = -1;
        for (int k = 1; k <= K; ++k)
            if (dp[k][n] < INF && (bestK < 0 || dp[k][n] < dp[bestK][n])) bestK = k;

        Decoded out;
        if (bestK > 0) {
            int j = n;
            for (int k = bestK; k >= 1; --k) {
                int i = pred[k][j];
                out.routes.emplace_back(perm.begin() + i, perm.begin() + j);
                j = i;
            }
            reverse(out.routes.begin(), out.routes.end());
        } else {
            // no split fits in K routes, so fill routes one after another
            vector<int> cur; long long load = 0;
            for (int c : perm) {
                if (!cur.empty() && load + I.demand[c] > I.capacity) { out.routes.push_back(cur); cur.clear(); load = 0; }
                cur.push_back(c); load += I.demand[c];
            }
            if (!cur.empty()) out.routes.push_back(cur);
            out.excess_vehicles = max(0, (int)out.routes.size() - I.vehicles);
        }
        out.cost = route_cost(out.routes);
        return out;
    }

public:
    double route_cost(const vector<vector<int>>& routes) const {
        double total = 0;
        for (auto& r : routes) {
            if (r.empty()) continue;
            total += I.dist[0][r.front()];
            for (size_t i = 1; i < r.size(); ++i) total += I.dist[r[i - 1]][r[i]];
            total += I.dist[r.back()][0];
        }
        return total;
    }
};

// picks one of the given mutations at random each time it is called
template <typename T>
class RandomChoiceMutation : public IMutation<T> {
    vector<shared_ptr<IMutation<T>>> ops;
public:
    explicit RandomChoiceMutation(vector<shared_ptr<IMutation<T>>> o) : ops(move(o)) {}
    void mutate(Individual<T>& ind, double pm, mt19937& rng) override {
        ops[rng() % ops.size()]->mutate(ind, pm, rng);
    }
};
