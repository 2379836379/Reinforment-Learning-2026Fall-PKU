#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include "tictactoe.hpp"

using namespace std;

class TicTacToePolicyBase{
    public:
        virtual TicTacToe::Action operator()(const TicTacToe::State& state) const = 0;
        virtual ~TicTacToePolicyBase() = default;
};

// A baseline policy: always take the first available square.
class TicTacToePolicyFirstEmpty : public TicTacToePolicyBase{
    public:
        TicTacToe::Action operator()(const TicTacToe::State& state) const override {
            vector<TicTacToe::Action> actions = state.action_space();
            assert(!actions.empty());
            return actions.front();
        }
};

namespace {

using Values = unordered_map<uint32_t, double>;

uint32_t state_key(const TicTacToe::State& state){
    return (static_cast<uint32_t>(state.board) << 2) |
           static_cast<uint32_t>(state.turn);
}

// Values are from X's point of view: win = 1, draw = 0.5, loss = 0.
double terminal_value(const TicTacToe::State& state){
    if (state.test_win()){
        return (0b11 - state.turn) == TicTacToe::PLAYER_X ? 1.0 : 0.0;
    }
    return 0.5;
}

bool terminal(const TicTacToe::State& state){
    return state.test_win() || state.full();
}

TicTacToe::State successor(const TicTacToe::State& state,
                           const TicTacToe::Action& action){
    TicTacToe::State next(state);
    next.put(action);
    return next;
}

double lookup(const Values& values, const TicTacToe::State& state){
    Values::const_iterator it = values.find(state_key(state));
    return it == values.end() ? 0.5 : it->second;
}

double successor_value(const Values& values, const TicTacToe::State& state){
    return terminal(state) ? terminal_value(state) : lookup(values, state);
}

} // namespace

// TD(0) value-estimation policy for X, against the fixed first-empty policy O.
class TicTacToePolicyDefault : public TicTacToePolicyBase{
    public:
        explicit TicTacToePolicyDefault(int episodes = 100000,
                                        double alpha = 0.1,
                                        double epsilon = 0.2)
            : values(), rng(20261002u){
            train(episodes, alpha, epsilon);
        }

        TicTacToe::Action operator()(const TicTacToe::State& state) const override {
            vector<TicTacToe::Action> actions = state.action_space();
            assert(!actions.empty());
            if (state.turn == TicTacToe::PLAYER_X){
                return greedy_action(state);
            }
            // O is intentionally the fixed opponent from the assignment.
            return actions.front();
        }

        double value(const TicTacToe::State& state) const {
            return lookup(values, state);
        }

    private:
        Values values;
        mt19937 rng;

        TicTacToe::Action greedy_action(const TicTacToe::State& state) const {
            vector<TicTacToe::Action> actions = state.action_space();
            TicTacToe::Action best = actions.front();
            double best_value = -1.0;
            for (size_t i = 0; i < actions.size(); ++i){
                double candidate = successor_value(values, successor(state, actions[i]));
                if (candidate > best_value){
                    best_value = candidate;
                    best = actions[i];
                }
            }
            return best;
        }

        TicTacToe::Action training_action(const TicTacToe::State& state,
                                          double epsilon){
            vector<TicTacToe::Action> actions = state.action_space();
            assert(!actions.empty());
            uniform_real_distribution<double> probability(0.0, 1.0);
            if (probability(rng) < epsilon){
                uniform_int_distribution<size_t> pick(0, actions.size() - 1);
                return actions[pick(rng)];
            }
            return greedy_action(state);
        }

        void train(int episodes, double alpha, double initial_epsilon){
            for (int episode = 0; episode < episodes; ++episode){
                TicTacToe env(false);
                TicTacToe::State current = env.get_state();
                double epsilon = max(0.02, initial_epsilon *
                                     (1.0 - static_cast<double>(episode) / episodes));

                while (!env.done()){
                    values.emplace(state_key(current), 0.5);
                    TicTacToe::Action action;
                    if (current.turn == TicTacToe::PLAYER_X){
                        action = training_action(current, epsilon);
                    } else {
                        vector<TicTacToe::Action> actions = current.action_space();
                        action = actions.front();
                    }
                    bool accepted = env.step(action);
                    assert(accepted);
                    TicTacToe::State next = env.get_state();
                    double target = terminal(next) ? terminal_value(next)
                                                    : lookup(values, next);
                    values[state_key(current)] +=
                        alpha * (target - values[state_key(current)]);
                    current = next;
                }
            }
        }
};

int main(){
    // Train X with the TD(0) update, then show one verbose evaluation game.
    TicTacToePolicyDefault policy;
    TicTacToe env(true);
    while (!env.done()){
        TicTacToe::State state = env.get_state();
        env.step(policy(state));
    }

    int winner = env.winner();
    cout << "Final result: ";
    if (winner == TicTacToe::PLAYER_X){
        cout << "X wins";
    } else if (winner == TicTacToe::PLAYER_O){
        cout << "O wins";
    } else {
        cout << "draw";
    }
    cout << endl;
    return winner == TicTacToe::PLAYER_X ? 0 : 1;
}
