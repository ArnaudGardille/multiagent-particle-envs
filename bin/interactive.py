#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
from time import sleep

from mpe.environment import MultiAgentEnv
from mpe.policy import InteractivePolicy
import mpe.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    print("scenario", scenario)
    # create world
    world = scenario.make_world()
    print("world", world)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    #done = False
    #while not done:
    for i in range(25):
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        done = sum(done_n) ==  len(policies)
        print(done_n, sum(done_n), len(policies))
        # render all agent views
        env.render()
        sleep(0.1)
        # display rewards
        for agent in env.world.agents:
            print(agent.name + " reward: %0.3f" % env._get_reward(agent))
