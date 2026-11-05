#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "maze.hpp"

class MazePolicyBase {
public:
    virtual ~MazePolicyBase() = default;
    virtual int operator()(const MazeEnv::State& state) const = 0;
};

class MazePolicyQLearning : public MazePolicyBase {
public:
    explicit MazePolicyQLearning(const MazeEnv& e)
        : env(e), q(e.max_x * e.max_y * 4, 0.0), rng(2022),
          epsilon(0.1), alpha(0.1), gamma(0.95) {}

    int operator()(const MazeEnv::State& state) const override {
        return greedy_action(state);
    }

    std::vector<int> learn(int iterations = 10000, int verbose_freq = 0) {
        std::vector<int> lengths;
        lengths.reserve(iterations);
        for (int episode = 0; episode < iterations; ++episode) {
            const int steps = run_episode();
            lengths.push_back(steps);
            if (verbose_freq > 0 && episode % verbose_freq == 0)
                std::cout << "episode_step: " << steps << std::endl;
        }
        return lengths;
    }

    void print_policy() const {
        static const char action_vis[] = "<>^v";
        for (int y = 0; y < env.max_y; ++y) {
            for (int x = 0; x < env.max_x; ++x) {
                const MazeEnv::State state(x, y);
                if (!env.is_valid_state(state)) std::cout << '#';
                else if (env.is_goal_state(state)) std::cout << 'G';
                else std::cout << action_vis[(*this)(state)];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

private:
    MazeEnv env;
    std::vector<double> q;
    mutable std::mt19937 rng;
    double epsilon, alpha, gamma;

    int locate(const MazeEnv::State& state, int action) const {
        return (state.second * env.max_x + state.first) * 4 + action;
    }

    int greedy_action(const MazeEnv::State& state) const {
        int action = 0;
        double value = q[locate(state, 0)];
        int ties = 1;
        for (int candidate = 1; candidate < 4; ++candidate) {
            if (q[locate(state, candidate)] > value) {
                value = q[locate(state, candidate)];
                action = candidate;
                ties = 1;
            } else if (q[locate(state, candidate)] == value) {
                ++ties;
                std::uniform_int_distribution<int> tie(1, ties);
                if (tie(rng) == 1) action = candidate;
            }
        }
        return action;
    }

    int epsilon_greedy(const MazeEnv::State& state) {
        std::uniform_real_distribution<double> probability(0.0, 1.0);
        if (probability(rng) < epsilon) {
            std::uniform_int_distribution<int> action(0, 3);
            return action(rng);
        }
        return greedy_action(state);
    }

    int run_episode() {
        MazeEnv::State state = env.reset();
        int steps = 0;
        while (!env.done() && steps < 100000) {
            const int action = epsilon_greedy(state);
            const MazeEnv::StepResult result = env.step(action);
            const double next_value = result.done
                ? 0.0
                : q[locate(result.next_state, greedy_action(result.next_state))];
            const int index = locate(state, action);
            q[index] += alpha * (result.reward + gamma * next_value - q[index]);
            state = result.next_state;
            ++steps;
        }
        return steps;
    }
};

// Planning transitions are queried from MazeEnv directly, as allowed by the
// assignment, so the experiment does not need a second model implementation.
class MazePolicyDynaQ : public MazePolicyBase {
public:
    explicit MazePolicyDynaQ(const MazeEnv& e, bool plus = false)
        : env(e), q(e.max_x * e.max_y * 4, 0.0), seen(e.max_x * e.max_y * 4, false),
          last_time(e.max_x * e.max_y * 4, 0), rng(2022),
          epsilon(0.1), alpha(0.1), gamma(0.95), kappa(plus ? 0.0001 : 0.0),
          clock(0), planning_steps(50) {}

    int operator()(const MazeEnv::State& state) const override {
        return greedy_action(state);
    }

    void set_wall(const MazeEnv::State& state, bool wall) {
        env.set_wall(state, wall);
    }

    std::vector<int> learn(int iterations = 10000, int verbose_freq = 0,
                           int requested_planning_steps = 50) {
        set_planning_steps(requested_planning_steps);
        std::vector<int> lengths;
        lengths.reserve(iterations);
        for (int episode = 0; episode < iterations; ++episode) {
            const int steps = run_episode();
            lengths.push_back(steps);
            if (verbose_freq > 0 && episode % verbose_freq == 0)
                std::cout << "episode_step: " << steps << std::endl;
        }
        return lengths;
    }

    void print_policy() const {
        static const char action_vis[] = "<>^v";
        for (int y = 0; y < env.max_y; ++y) {
            for (int x = 0; x < env.max_x; ++x) {
                const MazeEnv::State state(x, y);
                if (!env.is_valid_state(state)) std::cout << '#';
                else if (env.is_goal_state(state)) std::cout << 'G';
                else std::cout << action_vis[(*this)(state)];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

protected:
    MazeEnv env;
    std::vector<double> q;
    std::vector<bool> seen;
    std::vector<unsigned long long> last_time;
    std::vector<int> model_pairs;
    mutable std::mt19937 rng;
    double epsilon, alpha, gamma, kappa;
    unsigned long long clock;
    int planning_steps;

    int locate(const MazeEnv::State& state, int action) const {
        return (state.second * env.max_x + state.first) * 4 + action;
    }

    MazeEnv::State decode_state(int index) const {
        index /= 4;
        return MazeEnv::State(index % env.max_x, index / env.max_x);
    }

    int decode_action(int index) const { return index % 4; }

    int greedy_action(const MazeEnv::State& state) const {
        int action = 0;
        double value = q[locate(state, 0)];
        int ties = 1;
        for (int candidate = 1; candidate < 4; ++candidate) {
            if (q[locate(state, candidate)] > value) {
                value = q[locate(state, candidate)];
                action = candidate;
                ties = 1;
            } else if (q[locate(state, candidate)] == value) {
                ++ties;
                std::uniform_int_distribution<int> tie(1, ties);
                if (tie(rng) == 1) action = candidate;
            }
        }
        return action;
    }

    int epsilon_greedy(const MazeEnv::State& state) {
        std::uniform_real_distribution<double> probability(0.0, 1.0);
        if (probability(rng) < epsilon) {
            std::uniform_int_distribution<int> action(0, 3);
            return action(rng);
        }
        return greedy_action(state);
    }

    void remember_state(const MazeEnv::State& state) {
        if (!env.is_valid_state(state) || env.is_goal_state(state)) return;
        for (int action = 0; action < 4; ++action) {
            const int index = locate(state, action);
            if (!seen[index]) {
                seen[index] = true;
                model_pairs.push_back(index);
            }
        }
    }

    void update(const MazeEnv::State& state, int action,
                const MazeEnv::StepResult& result, bool planning) {
        const int index = locate(state, action);
        const double bonus = planning ? kappa * std::sqrt(
            static_cast<double>(clock - last_time[index])) : 0.0;
        const double next_value = result.done
            ? 0.0
            : q[locate(result.next_state, greedy_action(result.next_state))];
        q[index] += alpha * (result.reward + bonus + gamma * next_value - q[index]);
    }

    void plan() {
        if (model_pairs.empty()) return;
        std::uniform_int_distribution<std::size_t> pair(0, model_pairs.size() - 1);
        for (int i = 0; i < planning_steps; ++i) {
            const int index = model_pairs[pair(rng)];
            const MazeEnv::State state = decode_state(index);
            if (!env.is_valid_state(state)) continue;
            env.set_state(state);
            const MazeEnv::StepResult result = env.step(decode_action(index));
            update(state, decode_action(index), result, true);
        }
    }

    int run_episode() {
        MazeEnv::State state = env.reset();
        int steps = 0;
        while (!env.done() && steps < 100000) {
            remember_state(state);
            const int action = epsilon_greedy(state);
            const MazeEnv::StepResult result = env.step(action);
            ++clock;
            const int index = locate(state, action);
            last_time[index] = clock;
            remember_state(result.next_state);
            update(state, action, result, false);
            plan();
            // Planning temporarily moves the environment to sampled states.
            // Restore the real trajectory before taking the next action.
            env.set_state(result.next_state);
            state = result.next_state;
            ++steps;
        }
        return steps;
    }

private:
    void set_planning_steps(int steps) {
        assert(steps >= 0);
        planning_steps = steps;
    }
};

class MazePolicyDynaQPlus : public MazePolicyDynaQ {
public:
    explicit MazePolicyDynaQPlus(const MazeEnv& e) : MazePolicyDynaQ(e, true) {}
};

static double average(const std::vector<int>& values, std::size_t begin,
                      std::size_t end) {
    if (begin >= end) return 0.0;
    long long total = 0;
    for (std::size_t i = begin; i < end; ++i) total += values[i];
    return static_cast<double>(total) / static_cast<double>(end - begin);
}

static void print_comparison(const char* name, const std::vector<int>& lengths,
                             std::size_t change_episode) {
    const std::size_t split = std::min(change_episode, lengths.size());
    std::cout << name << "  before=" << average(lengths, 0, split)
              << " after=" << average(lengths, split, lengths.size()) << '\n';
}

int main() {
    const int max_x = 9, max_y = 6;
    const int start_x = 0, start_y = 2;
    const int target_x = 8, target_y = 0;
    const int initial_maze[max_y][max_x] = {
        {0, 0, 0, 0, 0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0, 0, 0, 1, 0},
        {0, 0, 1, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 1, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0}
    };

    // Blocking Maze: remove the short route after the first half of training.
    MazeEnv blocking(initial_maze, max_x, max_y, start_x, start_y,
                     target_x, target_y);
    MazePolicyDynaQ blocking_q(blocking);
    MazePolicyDynaQPlus blocking_plus(blocking);
    std::vector<int> blocking_q_lengths, blocking_plus_lengths;
    for (int episode = 0; episode < 1000; ++episode) {
        if (episode == 500) {
            blocking_q.set_wall(MazeEnv::State(3, 0), true);
            blocking_q.set_wall(MazeEnv::State(3, 1), true);
            blocking_plus.set_wall(MazeEnv::State(3, 0), true);
            blocking_plus.set_wall(MazeEnv::State(3, 1), true);
        }
        blocking_q_lengths.push_back(blocking_q.learn(1, 0, 20).front());
        blocking_plus_lengths.push_back(blocking_plus.learn(1, 0, 20).front());
    }
    print_comparison("Blocking Dyna-Q     ", blocking_q_lengths, 500);
    print_comparison("Blocking Dyna-Q+    ", blocking_plus_lengths, 500);

    // Shortcut Maze: open a direct route after the first half of training.
    MazeEnv shortcut(initial_maze, max_x, max_y, start_x, start_y,
                     target_x, target_y);
    MazePolicyDynaQ shortcut_q(shortcut);
    MazePolicyDynaQPlus shortcut_plus(shortcut);
    std::vector<int> shortcut_q_lengths, shortcut_plus_lengths;
    for (int episode = 0; episode < 1000; ++episode) {
        if (episode == 500) {
            // Opening the upper part of the right barrier creates the new
            // route used by the Shortcut Maze example.
            shortcut_q.set_wall(MazeEnv::State(7, 1), false);
            shortcut_q.set_wall(MazeEnv::State(7, 2), false);
            shortcut_plus.set_wall(MazeEnv::State(7, 1), false);
            shortcut_plus.set_wall(MazeEnv::State(7, 2), false);
        }
        shortcut_q_lengths.push_back(shortcut_q.learn(1, 0, 20).front());
        shortcut_plus_lengths.push_back(shortcut_plus.learn(1, 0, 20).front());
    }
    print_comparison("Shortcut Dyna-Q     ", shortcut_q_lengths, 500);
    print_comparison("Shortcut Dyna-Q+    ", shortcut_plus_lengths, 500);
    return 0;
}
