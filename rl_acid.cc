/*
  Two kinds of hard MDP problems defeat commonly used RL algorithms like Q-learning.

  (1) Antishaping.  If rewards in the vicinity of a start state favor
  staying near a start state, then reward values far from the start
  state are irrelevant.  The name comes from "reward shaping" which is
  typically used to make RL easier.  Here, we use it to make RL harder.

  (2) Combolock.  When most actions lead towards the start state
  uniform random exploration is relatively useless.  The name comes
  from "combination lock" where knowing the right sequence of steps to
  take is the problem.

  Here, we create families of learning problems which exhibits these
  characteristics.  

  We deliberately simplify in several ways to exhibit the problem.  In
  particular, all transitions are deterministic and only two actions
  are available in each state.

  These problems are straightforwardly solvable by special-case
  algorithms.  They can be solved by general purpose RL algorithms in
  the E^3 family.  They are not easily solved by Q-learning style
  algorithms.
*/
#include<vector>
#include<stdint.h>
#include<stdlib.h>
#include<iostream>
#include<algorithm>

using namespace std;

typedef pair<size_t, float> state_reward;

enum action { go_left, go_right };

bool compare(state_reward sr1, state_reward sr2) { return sr1.second < sr2.second; }

vector<state_reward> make_translation(size_t num_states)
{
  vector<state_reward> translation;
  for (size_t i =0; i < num_states; i++)
    translation.push_back(state_reward(i,drand48()));
  sort(translation.begin(), translation.end(), compare);
  return translation;
}

struct mdp {
private:
  uint32_t state;//the current state
  float total_reward;
  uint32_t num_steps;
  
  uint32_t start_state;
  uint32_t horizon;
  vector<pair<state_reward,state_reward> >  dynamics; //state -> action -> (state,reward)  
public:
  pair<pair<uint32_t,float>, float> next_state(action a)
  {
    pair<state_reward, state_reward> actions = dynamics[state];
    state_reward sr;
    if (a == go_left)
      sr = actions.first;
    else
      sr = actions.second;
    
    state = sr.first;
    total_reward += sr.second;
    num_steps++;
    float tr = -1.;
    if (num_steps == horizon)//end of an epoch
      {
	tr = total_reward / horizon;
	total_reward = 0;
	state = start_state;
	num_steps = 0;
      }
    return pair<state_reward,float>(sr,tr);
  }

  void create_antishape(size_t num_states)
  {
    total_reward = 0;
    num_steps = 0;
    horizon = num_states*2;

    vector<state_reward> translation = make_translation(num_states);

    start_state = translation[0].first;
    state = start_state;
    dynamics.resize(num_states);
    for (size_t i = 0; i < num_states; i++)
      {
	uint32_t left_state = i==0? 0 : i-1;
	uint32_t right_state = min(i+1,num_states-1);
	
	float left_reward = 0.25f / (float) (left_state+1);
	float right_reward = 0.25f / (float) (right_state+1);
	if (right_state == num_states-1)
	  right_reward = 1.f;
	
	state_reward sr_left(translation[left_state].first, left_reward);
	state_reward sr_right(translation[right_state].first, right_reward);
	
	dynamics[translation[i].first] = pair<state_reward,state_reward>(sr_left, sr_right);
      }
  }

  void create_combolock(size_t num_states)
  {
    total_reward = 0;
    num_steps = 0;
    horizon = num_states*2;

    vector<state_reward> translation = make_translation(num_states);

    start_state = translation[0].first;
    state = start_state;
    dynamics.resize(num_states);
    for (size_t i = 0; i < num_states; i++)
      {
	uint32_t left_state = 0;
	uint32_t right_state = 0;

	if (drand48() < 0.5)
	  left_state = min(i+1,num_states-1);
	else
	  right_state = min(i+1,num_states-1);
	
	float left_reward = 0;
	float right_reward = 0;
	if (right_state == num_states-1)
	  right_reward = 1.f;
	if (left_state == num_states-1)
	  left_reward = 1.f;
	
	state_reward sr_left(translation[left_state].first, left_reward);
	state_reward sr_right(translation[right_state].first, right_reward);
	
	dynamics[translation[i].first] = pair<state_reward,state_reward>(sr_left, sr_right);
      }
  }
  mdp():state(0),total_reward(0),num_steps(0),start_state(0),horizon(0) {}
};


void run_mdp(mdp& m, size_t num_epochs)
{  // a trivial controller that takes random actions
  while (num_epochs > 0)
    {
      action a = go_right;
      if (drand48() < 0.5)
	a = go_left;
      pair<state_reward,float> sf = m.next_state(a);
      if (sf.second >= 0.)
	{
	  cout << "average reward = " << sf.second << endl;
	  num_epochs--;
	}
    }
}

const size_t num_states = 100;
int main(int argc, char* argv[])
{
  mdp m;
  cout << "random controller on combolock" << endl;
  m.create_combolock(num_states);
  run_mdp(m,20);

  cout << "random controller on antishape" << endl;
  m.create_antishape(num_states);
  run_mdp(m,20);
}
