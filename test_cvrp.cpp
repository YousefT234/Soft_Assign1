#include <bits/stdc++.h>
using namespace std;
#include "Types.h"
#include "Interfaces.h"
#include "Selection.h"
#include "Crossover.h"
#include "Mutation.h"
#include "Replacement.h"
#include "GeneticAlgorithm.h"
#include "CVRP.h"

// Compares the GA with a brute force optimum on small instances
int main() {
    int ok = 0, total = 0;
    for (unsigned seed = 1; seed <= 5; ++seed) {
        CVRPInstance inst = CVRPInstance::generate(9, seed, 60, 0);
        CVRPProblem problem(inst);

        vector<int> p(inst.n); iota(p.begin(), p.end(), 1);
        double opt = 1e18;
        do { opt = min(opt, problem.fitness(p)); } while (next_permutation(p.begin(), p.end()));

        GAParameters prm; prm.pop_size = 60; prm.max_generations = 150; prm.chromosome_length = inst.n;
        prm.mutation_rate = 0.3; prm.crossover_rate = 0.9;
        GeneticAlgorithm<int> ga(prm,
            [&](const vector<int>& g) { return problem.fitness(g); },
            [&](const vector<int>& g) { return problem.is_feasible(g); },
            [&](int l, mt19937& r) { return problem.init(l, r); });
        ga.set_selection(make_shared<TournamentSelection<int>>(3));
        ga.set_crossover(make_shared<OrderCrossover<int>>());
        ga.set_mutation(make_shared<InversionMutation<int>>());
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<int>>(2));
        ga.run();
        auto b = ga.get_best_solution();
        bool match = fabs(b.raw_fitness - opt) < 1e-6 && b.is_feasible;
        cout << "seed " << seed << ": brute-force optimum " << opt << " | GA " << b.raw_fitness
             << (match ? "  OK" : "  (differs)") << "\n";
        ok += match; ++total;
    }
    cout << ok << "/" << total << " instances solved to proven optimality\n";
    return ok == total ? 0 : 1;
}
