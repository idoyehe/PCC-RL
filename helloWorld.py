import gym

from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1

env = gym.make('CartPole-v1')


class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        arch = [32, 64]
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi": arch, "vf": arch}],
                                          feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess


model = PPO1(MyMlpPolicy, env, verbose=1, timesteps_per_actorbatch=250)
model.learn(total_timesteps=25000)
# model.save("ppo1_cartpole")
#
# del model  # remove to demonstrate saving and loading
#
# model = PPO1.load("ppo1_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
