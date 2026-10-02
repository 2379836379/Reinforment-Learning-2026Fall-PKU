#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

using namespace std;

class JackCarRental{
    public:
        static const int MAX_CAR = 20;
        static const int MOVE_LIMIT = 5;
        static const double MOVE_COST;
        static const double RENT_PRICE;
        static const double MEAN_REQUEST_1;
        static const double MEAN_REQUEST_2;
        static const double MEAN_RETURN_1;
        static const double MEAN_RETURN_2;

        typedef pair<int, int> State;
        typedef pair<State, double> Transition;

        bool verbose;

        JackCarRental(int car_1 = 0, int car_2 = 0, bool verbose = false)
            : verbose(verbose), car_1(0), car_2(0), day(0), engine(20261002u),
              request_1(MEAN_REQUEST_1), request_2(MEAN_REQUEST_2),
              return_1(MEAN_RETURN_1), return_2(MEAN_RETURN_2){
            set_state(car_1, car_2);
        }

        State state() const {
            return make_pair(car_1, car_2);
        }

        void set_state(int new_car_1, int new_car_2){
            if (!valid_state(new_car_1, new_car_2)){
                throw invalid_argument("car counts must be in [0, 20]");
            }
            car_1 = new_car_1;
            car_2 = new_car_2;
            if (verbose){
                cout << "State set to (" << car_1 << ", " << car_2 << ")" << endl;
            }
        }

        void reset(){
            day = 0;
            set_state(0, 0);
        }

        Transition step(int action){
            int low = max(-min(MOVE_LIMIT, car_2), -car_2);
            int high = min(MOVE_LIMIT, car_1);
            if (action < low || action > high){
                throw invalid_argument("move action is infeasible for current state");
            }

            ++day;
            if (verbose){
                cout << "\nDay: " << day << " State: (" << car_1
                     << ", " << car_2 << ")" << endl;
            }
            car_1 -= action;
            car_2 += action;
            double reward = -abs(action) * MOVE_COST;
            int request_1_count = request_1(engine);
            int request_2_count = request_2(engine);
            int rented_1 = min(car_1, request_1_count);
            int rented_2 = min(car_2, request_2_count);
            car_1 -= rented_1;
            car_2 -= rented_2;
            reward += RENT_PRICE * (rented_1 + rented_2);
            car_1 = min(MAX_CAR, car_1 + return_1(engine));
            car_2 = min(MAX_CAR, car_2 + return_2(engine));

            if (verbose){
                cout << "\tAction: " << action << ", reward: " << reward
                     << ", new state: (" << car_1 << ", " << car_2 << ")"
                     << endl;
            }
            return make_pair(state(), reward);
        }

        int sample_action(){
            int low = -min(MOVE_LIMIT, car_2);
            int high = min(MOVE_LIMIT, car_1);
            uniform_int_distribution<int> distribution(low, high);
            return distribution(engine);
        }

        static bool valid_state(int first, int second){
            return first >= 0 && first <= MAX_CAR &&
                   second >= 0 && second <= MAX_CAR;
        }

    private:
        int car_1;
        int car_2;
        int day;
        default_random_engine engine;
        poisson_distribution<int> request_1;
        poisson_distribution<int> request_2;
        poisson_distribution<int> return_1;
        poisson_distribution<int> return_2;
};

const int JackCarRental::MAX_CAR;
const int JackCarRental::MOVE_LIMIT;
const double JackCarRental::MOVE_COST = 2.0;
const double JackCarRental::RENT_PRICE = 10.0;
const double JackCarRental::MEAN_REQUEST_1 = 3.0;
const double JackCarRental::MEAN_REQUEST_2 = 4.0;
const double JackCarRental::MEAN_RETURN_1 = 3.0;
const double JackCarRental::MEAN_RETURN_2 = 2.0;

namespace {

const int N = JackCarRental::MAX_CAR + 1;
const double GAMMA = 0.9;
const double EVALUATION_TOLERANCE = 1e-6;
const int MAX_POISSON_COUNT = JackCarRental::MAX_CAR;

typedef array<array<double, N>, N> Grid;
typedef array<array<int, N>, N> Policy;
typedef array<double, N> Probability;

struct LocationModel{
    // next_probability[c][next] = P(next inventory | c cars before rentals)
    array<Probability, N> next_probability;
    // Expected number rented at the location when c cars are available.
    array<double, N> expected_rented;
};

Probability poisson_probabilities(double mean){
    Probability probability{};
    double total = 0.0;
    double term = exp(-mean);
    for (int count = 0; count < MAX_POISSON_COUNT; ++count){
        probability[count] = term;
        total += term;
        term *= mean / (count + 1);
    }
    // Count MAX_POISSON_COUNT represents that count and all larger counts.
    probability[MAX_POISSON_COUNT] = max(0.0, 1.0 - total);
    return probability;
}

LocationModel build_location_model(double request_mean, double return_mean){
    Probability request_probability = poisson_probabilities(request_mean);
    Probability return_probability = poisson_probabilities(return_mean);
    LocationModel model{};

    for (int available = 0; available < N; ++available){
        for (int demand = 0; demand < N; ++demand){
            int rented = min(available, demand);
            model.expected_rented[available] +=
                request_probability[demand] * rented;
            int after_rental = available - rented;
            for (int returned = 0; returned < N; ++returned){
                int next = min(JackCarRental::MAX_CAR,
                               after_rental + returned);
                model.next_probability[available][next] +=
                    request_probability[demand] * return_probability[returned];
            }
        }
    }
    return model;
}

bool action_valid(int car_1, int car_2, int action){
    return action >= -min(JackCarRental::MOVE_LIMIT, car_2) &&
           action <= min(JackCarRental::MOVE_LIMIT, car_1);
}

double action_value(const Grid& values, int car_1, int car_2, int action,
                    const LocationModel& location_1,
                    const LocationModel& location_2){
    // Moving into a full lot is capped at the capacity, matching the
    // environment transition and keeping every successor in [0, 20].
    int available_1 = min(JackCarRental::MAX_CAR, car_1 - action);
    int available_2 = min(JackCarRental::MAX_CAR, car_2 + action);
    double expected_future = 0.0;

    for (int next_1 = 0; next_1 < N; ++next_1){
        double probability_1 = location_1.next_probability[available_1][next_1];
        if (probability_1 == 0.0){
            continue;
        }
        for (int next_2 = 0; next_2 < N; ++next_2){
            expected_future += probability_1 *
                location_2.next_probability[available_2][next_2] *
                values[next_1][next_2];
        }
    }

    double expected_rental_income = JackCarRental::RENT_PRICE *
        (location_1.expected_rented[available_1] +
         location_2.expected_rented[available_2]);
    double move_cost = JackCarRental::MOVE_COST * abs(action);
    return expected_rental_income - move_cost + GAMMA * expected_future;
}

Grid evaluate_policy(const Policy& policy, const Grid& initial_values,
                     const LocationModel& location_1,
                     const LocationModel& location_2){
    Grid values = initial_values;
    Grid updated{};

    for (int iteration = 0; iteration < 10000; ++iteration){
        double delta = 0.0;
        for (int car_1 = 0; car_1 < N; ++car_1){
            for (int car_2 = 0; car_2 < N; ++car_2){
                updated[car_1][car_2] = action_value(
                    values, car_1, car_2, policy[car_1][car_2],
                    location_1, location_2);
                delta = max(delta, abs(updated[car_1][car_2] -
                                       values[car_1][car_2]));
            }
        }
        values = updated;
        if (delta < EVALUATION_TOLERANCE){
            break;
        }
    }
    return values;
}

Policy solve_policy_iteration(){
    LocationModel location_1 = build_location_model(
        JackCarRental::MEAN_REQUEST_1, JackCarRental::MEAN_RETURN_1);
    LocationModel location_2 = build_location_model(
        JackCarRental::MEAN_REQUEST_2, JackCarRental::MEAN_RETURN_2);
    Policy policy{};
    Grid values{};

    while (true){
        values = evaluate_policy(policy, values, location_1, location_2);
        bool stable = true;

        for (int car_1 = 0; car_1 < N; ++car_1){
            for (int car_2 = 0; car_2 < N; ++car_2){
                int old_action = policy[car_1][car_2];
                int best_action = old_action;
                double best_value = action_value(
                    values, car_1, car_2, old_action, location_1, location_2);
                for (int action = -JackCarRental::MOVE_LIMIT;
                     action <= JackCarRental::MOVE_LIMIT; ++action){
                    if (!action_valid(car_1, car_2, action)){
                        continue;
                    }
                    double candidate = action_value(
                        values, car_1, car_2, action, location_1, location_2);
                    if (candidate > best_value + 1e-10){
                        best_value = candidate;
                        best_action = action;
                    }
                }
                policy[car_1][car_2] = best_action;
                if (best_action != old_action){
                    stable = false;
                }
            }
        }
        if (stable){
            return policy;
        }
    }
}

} // namespace

int main(){
    Policy policy = solve_policy_iteration();
    cout << "Optimal policy (rows: cars at A, columns: cars at B)" << endl;
    cout << "Positive action: A -> B; negative action: B -> A" << endl;
    for (int car_1 = 0; car_1 < N; ++car_1){
        for (int car_2 = 0; car_2 < N; ++car_2){
            cout << setw(4) << policy[car_1][car_2];
        }
        cout << endl;
    }
    return 0;
}
