#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <utility>
#include <vector>
#include "blackjack.hpp"

class BlackjackPolicyBase{
    public:
        virtual int operator() (const Blackjack::State& state)=0;
};

class BlackjackPolicyDefault : public BlackjackPolicyBase{
    public:
        int operator() (const Blackjack::State& state){
            if (state.turn == Blackjack::PLAYER){
                return state.player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
            } else {
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            }
        }
};

class BlackjackPolicyLearnableDefault : public BlackjackPolicyBase{
    static constexpr const char *ACTION_NAME = "HS";
    public:
        int operator() (const Blackjack::State& state){
            if (state.turn == Blackjack::DEALER){
                return state.dealer_sum >= 17 ? Blackjack::STICK : Blackjack::HIT;
            } else {
                return policy[state.dealer_shown][state.player_ace][state.player_sum];
            }
        }
        void update_policy(){
            // Policy improvement: choose the action with the largest
            // estimated action value.  Sums below 11 and sum 21 are not part
            // of the book's state space and retain their obvious actions.
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 11; player_sum <= 20; ++ player_sum){
                        policy[dealer_shown][player_ace][player_sum] =
                            value[dealer_shown][player_ace][player_sum][Blackjack::HIT]
                                > value[dealer_shown][player_ace][player_sum][Blackjack::STICK]
                            ? Blackjack::HIT : Blackjack::STICK;
                    }
                    policy[dealer_shown][player_ace][21] = Blackjack::STICK;
                }
            }
        }
        // n iterations for each start (initial state and first action), no need to modify.
        void update_value(Blackjack& env, int n=10000){
            set_value_initial();

            // Exploring starts: evaluate both possible first actions from
            // every state represented in Sutton's blackjack example.  Later
            // player actions follow the current policy; dealer actions are
            // handled by BlackjackPolicyLearnableDefault::operator().
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 11; player_sum <= 20; ++ player_sum){
                        for (int first_action = Blackjack::HIT;
                             first_action <= Blackjack::STICK; ++ first_action){
                            for (int sample = 0; sample < n; ++ sample){
                                env.reset(dealer_shown, bool(player_ace), player_sum);
                                episode.clear();

                                Blackjack::State state = env.state();
                                int action = first_action;
                                bool done = false;
                                while (not done){
                                    state = env.state();
                                    if (state.turn == Blackjack::PLAYER){
                                        episode.emplace_back(state, action);
                                    }
                                    Blackjack::StepResult result = env.step(action);
                                    done = result.done;
                                    if (done){
                                        // Rewards are zero before termination,
                                        // so the terminal player reward is the
                                        // return for every recorded step.
                                        double reward = result.player_reward;
                                        for (const EpisodeStep& step : episode){
                                            int& count = state_action_count[
                                                step.dealer_shown][step.player_ace]
                                                [step.player_sum][step.action];
                                            double& estimate = value[
                                                step.dealer_shown][step.player_ace]
                                                [step.player_sum][step.action];
                                            ++ count;
                                            estimate += (reward - estimate) / count;
                                        }
                                    } else {
                                        state = env.state();
                                        action = (*this)(state);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        void print_policy() const {
            cout << setw(10) << "Player Without Ace" << "\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][0][player_sum]];
                }
                cout << "\t\t\t\t" ;
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << ACTION_NAME[policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        void print_value() const {
            cout << setw(40) << "Player Without Ace" << setw(20) <<"\t\t\t" << "Player With Usable Ace." << endl;
            for (int player_sum = 21; player_sum >= 11; -- player_sum){
                for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][0][player_sum][
                        policy[dealer_shown][0][player_sum]];
                }
                cout << "\t";
                for (int dealer_shown = 1; dealer_shown <= 10 ; ++ dealer_shown){
                    cout << fixed << setprecision(2) << setw(6) << value[dealer_shown][1][player_sum][
                        policy[dealer_shown][1][player_sum]];
                }
                cout << endl;
            }
            cout << endl;
        }
        
        BlackjackPolicyLearnableDefault(){
            set_value_initial();
            for (int dealer_shown = 1; dealer_shown <= 10; ++ dealer_shown){
                for (int player_ace = 0; player_ace <= 1; ++ player_ace){
                    for (int player_sum = 0; player_sum <= 21; ++ player_sum){
                        int& action = policy[dealer_shown][player_ace][player_sum];
                        action = player_sum >= 20 ? Blackjack::STICK : Blackjack::HIT;
                    }
                }
            }
        }
    private:
        // 11: dealer_shown (A-10); 
        // 2: player_usable_ace (true/false);
        // 22: player_sum (only need to consider 11-20, because HIT when sum<11 and STICK when sum=21 are always best action.)
        int policy[11][2][22];
        // 11:dealer_shown; 2:player_usable_ace; 22:player_sum; 2:action (0:HIT, 1:STICK)
        double value[11][2][22][2];
        int state_action_count[11][2][22][2];

        // record a episode sampled (only player's steps).
        struct EpisodeStep{
            // state: (dealer_shown, player_ace, player_sum)
            int dealer_shown;
            int player_ace;
            int player_sum;
            // the action taken at state
            int action;
            EpisodeStep(const Blackjack::State& state, int action){
                dealer_shown = state.dealer_shown;
                player_ace = int(state.player_ace);
                player_sum = state.player_sum;
                this->action = action;
            }
            EpisodeStep(){}
        };
        vector<EpisodeStep> episode; 

        void set_value_initial(){
            memset(value, 0, sizeof(value));
            memset(state_action_count, 0, sizeof(state_action_count));
        }
};

// Demonstrative play of player & dealer with default policy 
int main(){
    Blackjack env;
    BlackjackPolicyLearnableDefault policy;
    for (int iteration = 0; iteration < 10; ++ iteration){
        policy.update_value(env);
        policy.update_policy();
    }
    policy.print_policy();
    return 0;
}
