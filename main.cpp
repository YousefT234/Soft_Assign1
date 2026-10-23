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

int main() {
    {
        cout << "=== Running Binary Optimization (OneMax) ===" << "\n";
        GAParameters params;
        params.pop_size = 40;
        params.chromosome_length = 30;
        params.max_generations = 50;
        params.crossover_rate = 0.85;
        params.mutation_rate = 0.02;
        params.elitism_count = 2;
        params.opt_type = OptimizationType::MAXIMIZE;

        auto fitness = [](const vector<int>& genes) {
            return static_cast<double>(accumulate(genes.begin(), genes.end(), 0));
        };

        auto feasibility = [](const vector<int>&) {
            return true;
        };

        auto init = [](int length, mt19937& rng) {
            uniform_int_distribution<int> dist(0, 1);
            vector<int> genes(length);
            for (auto& g : genes) g = dist(rng);
            return genes;
        };

        GeneticAlgorithm<int> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<TournamentSelection<int>>(3));
        ga.set_crossover(make_shared<MultiPointCrossover<int>>(2));
        ga.set_mutation(make_shared<BitFlipMutation<int>>([](const int& v, mt19937&) { return 1 - v; }));
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<int>>(2));

        ga.run(true);
        cout << "Binary Best Raw Score: " << ga.get_best_solution().raw_fitness << "\n\n";
    }

    {
        cout << "=== Running Integer Permutation Optimization (Order Crossover & Swap) ===" << "\n";
        int n = 10;
        GAParameters params;
        params.pop_size = 50;
        params.chromosome_length = n;
        params.max_generations = 100;
        params.crossover_rate = 0.9;
        params.mutation_rate = 0.2;
        params.opt_type = OptimizationType::MINIMIZE;

        auto fitness = [](const vector<int>& genes) {
            double cost = 0.0;
            for (int i = 0; i < genes.size(); ++i) {
                cost += abs(genes[i] - static_cast<int>(i));
            }
            return cost;
        };

        auto feasibility = [](const vector<int>& genes) {
            unordered_set<int> seen(genes.begin(), genes.end());
            return seen.size() == genes.size();
        };

        auto init = [](int length, mt19937& rng) {
            vector<int> genes(length);
            iota(genes.begin(), genes.end(), 0);
            shuffle(genes.begin(), genes.end(), rng);
            return genes;
        };

        GeneticAlgorithm<int> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<RouletteWheelSelection<int>>());
        ga.set_crossover(make_shared<OrderCrossover<int>>());
        ga.set_mutation(make_shared<SwapMutation<int>>());
        ga.set_replacement(make_shared<SteadyStateReplacement<int>>());

        ga.run(true);
        cout << "Integer Permutation Best Distance to Sorted: " << ga.get_best_solution().raw_fitness << "\n\n";
    }

    {
        cout << "=== Running Floating-Point Optimization (Sphere Function) ===" << "\n";
        GAParameters params;
        params.pop_size = 60;
        params.chromosome_length = 5;
        params.max_generations = 80;
        params.crossover_rate = 0.8;
        params.mutation_rate = 0.1;
        params.opt_type = OptimizationType::MINIMIZE;

        auto fitness = [](const vector<double>& genes) {
            double sum = 0.0;
            for (double x : genes) sum += x * x;
            return sum;
        };

        auto feasibility = [](const vector<double>& genes) {
            for (double x : genes) {
                if (x < -5.12 || x > 5.12) return false;
            }
            return true;
        };

        auto init = [](int length, mt19937& rng) {
            uniform_real_distribution<double> dist(-5.12, 5.12);
            vector<double> genes(length);
            for (auto& g : genes) g = dist(rng);
            return genes;
        };

        GeneticAlgorithm<double> ga(params, fitness, feasibility, init);
        ga.set_selection(make_shared<TournamentSelection<double>>(2));
        ga.set_crossover(make_shared<MultiPointCrossover<double>>(1));
        ga.set_mutation(make_shared<UniformFPMutation<double>>(-5.12, 5.12));
        ga.set_replacement(make_shared<ElitistGenerationalReplacement<double>>(2));

        ga.run(true);
        cout << "Floating Point Best Sphere Cost: " << ga.get_best_solution().raw_fitness << "\n";
    }

    return 0;
}
