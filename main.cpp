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

// reads one value from the user, pressing Enter keeps the default
template <typename T>
T ask(const string& text, T default_value) {
    cout << text << " [default " << default_value << "]: ";
    string line;
    if (!getline(cin, line) || line.empty()) return default_value;
    istringstream ss(line);
    T value;
    if (ss >> value) return value;
    cout << "  not a valid number, using the default\n";
    return default_value;
}

int main() {
    cout << "Capacitated Vehicle Routing Problem solved with the GA toolbox\n";
    cout << "Enter the GA settings below, or press Enter to keep the default.\n\n";

    GAParameters params;
    params.pop_size = 100;
    params.max_generations = 3000;
    params.crossover_rate = 0.9;
    params.mutation_rate = 0.3;
    params.elitism_count = 2;
    params.penalty_factor = 1000.0;
    params.opt_type = OptimizationType::MINIMIZE;
    int tournament = 5, runs = 5, vehicles = 0;
    bool verbose = false;

    params.pop_size = ask("Population size", params.pop_size);
    params.max_generations = ask("Number of generations", params.max_generations);
    params.crossover_rate = ask("Crossover rate (0 to 1)", params.crossover_rate);
    params.mutation_rate = ask("Mutation rate (0 to 1)", params.mutation_rate);
    tournament = ask("Tournament size", tournament);
    runs = ask("Number of runs (best one is shown)", runs);
    vehicles = ask("Number of vehicles (0 = use the instance's 4)", vehicles);
    cout << "\n";

    CVRPInstance inst = CVRPInstance::ortools_example();
    if (vehicles > 0) inst.vehicles = vehicles;
    CVRPProblem problem(inst);
    params.chromosome_length = inst.n;

    cout << "Instance: " << inst.name << " | customers: " << inst.n << " | capacity: " << inst.capacity
         << " | vehicles: " << inst.vehicles << " | total demand: " << inst.total_demand() << "\n";
    cout << "GA: pop=" << params.pop_size << " gens=" << params.max_generations << " pc=" << params.crossover_rate
         << " pm=" << params.mutation_rate << " tournament=" << tournament << " runs=" << runs << "\n\n";

    Individual<int> best;
    bool have_best = false;
    double total_time = 0;

    for (int r = 0; r < runs; ++r) {
        auto fitness     = [&](const vector<int>& g) { return problem.fitness(g); };
        auto feasibility = [&](const vector<int>& g) { return problem.is_feasible(g); };
        auto init        = [&](int len, mt19937& rng) { return problem.init(len, rng); };

        GeneticAlgorithm<int> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<TournamentSelection<int>>(tournament));
        ga.set_crossover(make_shared<OrderCrossover<int>>());
        ga.set_mutation(make_shared<RandomChoiceMutation<int>>(vector<shared_ptr<IMutation<int>>>{
            make_shared<SwapMutation<int>>(), make_shared<InversionMutation<int>>()}));
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<int>>(params.elitism_count));

        auto t0 = chrono::steady_clock::now();
        ga.run(verbose);
        double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        total_time += secs;

        Individual<int> res = ga.get_best_solution();
        cout << "Run " << (r + 1) << ": cost = " << fixed << setprecision(2) << res.raw_fitness
             << " | feasible: " << (res.is_feasible ? "yes" : "no") << " | " << setprecision(2) << secs << " s\n";
        if (!have_best || res.fitness < best.fitness) { best = res; have_best = true; }
    }

    auto d = problem.decode(best.genes);
    auto num = [](double v) {
        ostringstream os;
        if (fabs(v - round(v)) < 1e-9) os << (long long)round(v);
        else os << fixed << setprecision(2) << v;
        return os.str();
    };

    cout << "\nObjective: " << num(d.cost) << "\n";
    long long total_load = 0;
    for (size_t i = 0; i < d.routes.size(); ++i) {
        cout << "Route for vehicle " << i << ":\n 0 Load(0)";
        long long load = 0;
        for (int c : d.routes[i]) {
            cout << " ->  " << c << " Load(" << load << ")";
            load += inst.demand[c];
        }
        cout << " ->  0 Load(" << load << ")\n";
        cout << "Distance of the route: " << num(problem.route_cost({d.routes[i]})) << "m\n";
        cout << "Load of the route: " << load << "\n\n";
        total_load += load;
    }
    cout << "Total Distance of all routes: " << num(d.cost) << "m\n";
    cout << "Total Load of all routes: " << total_load << "\n";
    cout << "Feasible: " << (best.is_feasible ? "yes" : "NO") << "\n";
    return best.is_feasible ? 0 : 1;
}
