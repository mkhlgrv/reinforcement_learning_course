import gym
import numpy as np
import time
env = gym.make('Taxi-v3', new_step_api=True, render_mode='human')

state_n = 500
action_n = 6

class CEM():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        self.policy = np.ones((self.state_n, self.action_n)) / self.action_n
    
    def get_action(self, state):
        return int(np.random.choice(np.arange(action_n),
            p = self.policy[state, :]))
    def update_policy(self, elite_trajectories):
        
        pre_policy = np.ones((self.state_n, self.action_n)) / self.action_n
        
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                pre_policy[state][action] += 1
        for state in range(self.state_n):
            if sum(pre_policy[state])>0:
                self.policy[state] = pre_policy[state] / sum(pre_policy[state])
                
def get_state(obs):
    return int(obs)


def get_trajectory(agent, trajectory_len):
    
    trajectory = {'states':[],
                 'actions':[],
                 'total_reward': 0}
    
    obs = env.reset()
    state = get_state(obs)
    
    for _ in range(trajectory_len):
        
        action = agent.get_action(state)
        trajectory['states'].append(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _, _  = env.step(action)
        state = get_state(obs)
        
        env.render(mode="human")
        time.sleep(10)
        
        trajectory['total_reward']+= reward
        
        if done:
            break
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    
    quantile = np.quantile([trajectory['total_reward'] for trajectory in trajectories],
                q_param)
    
    elite_trajectories = [trajectory for trajectory in trajectories if trajectory['total_reward']> quantile]
    return elite_trajectories
    
    

agent = CEM(state_n, action_n)

epoch_n = 10
trajectory_n = 10 # k
trajectory_len = 1000
q_param = 0.8

for _ in range(epoch_n):
    trajectories = [get_trajectory(agent, trajectory_len) for _ in range(trajectory_n)]
    
    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    # print(mean_total_reward)
    elite_trajectories = get_elite_trajectories(trajectories, q_param)
    
    if len(elite_trajectories)>0:
        agent.update_policy(elite_trajectories)