#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

using namespace std;

class GridWorld{
    public:
        static const int NORTH = 0;
        static const int SOUTH = 1;
        static const int EAST = 2;
        static const int WEST = 3;
        static const int SIZE = 5;
        static const char ACTION_NAME[][16];
        typedef pair<int, int> State;
        typedef pair<State, double> Transition;

        explicit GridWorld(int x = 0, int y = 0, bool verbose = false)
            : x(0), y(0), verbose(verbose){
            set_state(x, y);
        }

        State state() const {
            return make_pair(x, y);
        }

        void set_state(int new_x, int new_y){
            if (new_x < 0 || new_x >= SIZE || new_y < 0 || new_y >= SIZE){
                throw invalid_argument("GridWorld state is outside the 5x5 grid");
            }
            x = new_x;
            y = new_y;
            if (verbose){
                cout << "State reset: (" << x << "," << y << ")" << endl;
            }
        }

        void reset(){
            set_state(0, 0);
        }

        Transition step(int action){
            State old_state = state();
            Transition result = transition(old_state, action);
            x = result.first.first;
            y = result.first.second;
            if (verbose){
                cout << "State: (" << old_state.first << "," << old_state.second << ")" << endl;
                cout << "Action: " << ACTION_NAME[action] << endl;
                cout << "Reward: " << result.second << endl;
                cout << "New State: (" << x << "," << y << ")" << endl << endl;
            }
            return result;
        }

        // The model used by both the environment and dynamic programming.
        static Transition transition(State current, int action){
            if (action < NORTH || action > WEST){
                throw invalid_argument("GridWorld action must be in [0, 4)");
            }

            // Special states are checked before ordinary movement.
            if (current == make_pair(1, 0)){
                return make_pair(make_pair(1, 4), 10.0);
            }
            if (current == make_pair(3, 0)){
                return make_pair(make_pair(3, 2), 5.0);
            }

            int next_x = current.first;
            int next_y = current.second;
            switch (action){
                case NORTH: --next_y; break;
                case SOUTH: ++next_y; break;
                case EAST:  ++next_x; break;
                case WEST:  --next_x; break;
            }

            if (next_x < 0 || next_x >= SIZE || next_y < 0 || next_y >= SIZE){
                return make_pair(current, -1.0);
            }
            return make_pair(make_pair(next_x, next_y), 0.0);
        }

        int sample_action() const {
            return rand() % 4;
        }

    private:
        int x;
        int y;
        bool verbose;
};

const char GridWorld::ACTION_NAME[][16] = {
    "NORTH(0,-1)", "SOUTH(0,1)", "EAST(1,0)", "WEST(-1,0)"
};

typedef array<array<double, GridWorld::SIZE>, GridWorld::SIZE> ValueGrid;

// Evaluate the uniform random policy with synchronous Bellman expectation updates.
ValueGrid evaluate_random_policy(double gamma = 0.9,
                                 double tolerance = 1e-12,
                                 int* iterations = nullptr){
    ValueGrid values{};
    ValueGrid updated{};
    int iteration = 0;

    while (true){
        double delta = 0.0;
        for (int y = 0; y < GridWorld::SIZE; ++y){
            for (int x = 0; x < GridWorld::SIZE; ++x){
                double expected = 0.0;
                GridWorld::State current = make_pair(x, y);
                for (int action = GridWorld::NORTH;
                     action <= GridWorld::WEST; ++action){
                    GridWorld::Transition result =
                        GridWorld::transition(current, action);
                    expected += 0.25 *
                        (result.second + gamma *
                         values[result.first.second][result.first.first]);
                }
                updated[y][x] = expected;
                delta = max(delta, abs(expected - values[y][x]));
            }
        }
        values = updated;
        ++iteration;
        if (delta < tolerance){
            break;
        }
    }

    if (iterations != nullptr){
        *iterations = iteration;
    }
    return values;
}

int main(){
    int iterations = 0;
    ValueGrid values = evaluate_random_policy(0.9, 1e-12, &iterations);

    cout << "Random-policy state values (gamma=0.90)" << endl;
    cout << fixed << setprecision(2);
    for (int y = 0; y < GridWorld::SIZE; ++y){
        for (int x = 0; x < GridWorld::SIZE; ++x){
            cout << setw(8) << values[y][x];
        }
        cout << endl;
    }
    cout << "Iterations: " << iterations << endl;
    return 0;
}
