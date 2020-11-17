# Based on https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd

ENV_NAME="gmpos-v1"

import numpy as np
import sys
import numpy
import random
import gym
from gym.envs.registration import register
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

# https://stackoverflow.com/a/39662359
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

if is_notebook():
    import matplotlib.pyplot as plt

class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()
        self.d1 = Dense(64, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.dout = Dense(action_dim, activation="tanh")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.dout(x)


class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()
        self.d1 = Dense(64, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.dout = Dense(1, activation="linear")

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.dout(x)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def __len__(self):
        return len(self.dones)

    def get_all_items(self):
        states = tf.constant(self.states, dtype=tf.float32)
        actions = tf.constant(self.actions, dtype=tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype=tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype=tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype=tf.float32)

        return tf.data.Dataset.from_tensor_slices(
            (states, actions, rewards, dones, next_states)
        )

    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]


class Distributions:
    def sample(self, mean, std):
        distribution = tfp.distributions.Normal(mean, std)
        return distribution.sample()

    def entropy(self, mean, std):
        distribution = tfp.distributions.Normal(mean, std)
        return distribution.entropy()

    def logprob(self, mean, std, value_data):
        distribution = tfp.distributions.Normal(mean, std)
        return tf.expand_dims(distribution.log_prob(value_data), 1)

    def kl_divergence(self, mean1, std1, mean2, std2):
        distribution1 = tfp.distributions.Normal(mean1, std1)
        distribution2 = tfp.distributions.Normal(mean2, std2)

        return tf.expand_dims(
            tfp.distributions.kl_divergence(distribution1, distribution2), 1
        )


class PolicyFunction:
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def monte_carlo_discounted(self, datas):
        returns = []
        running_add = 0

        for i in reversed(range(len(datas))):
            running_add = running_add * self.gamma + datas[i]
            returns.insert(0, running_add)

        return tf.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae = 0
        adv = []

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)

        return tf.stack(adv)


class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        is_training_mode,
        policy_kl_range,
        policy_params,
        value_clip,
        entropy_coef,
        vf_loss_coef,
        minibatch,
        PPO_epochs,
        gamma,
        lam,
        learning_rate,
    ):
        self.policy_kl_range = policy_kl_range
        self.policy_params = policy_params
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.minibatch = minibatch
        self.PPO_epochs = PPO_epochs
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.std = tf.ones([1, action_dim])

        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)

        self.critic = Critic_Model(state_dim, action_dim)
        self.critic_old = Critic_Model(state_dim, action_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = Memory()
        self.policy_function = PolicyFunction(gamma, lam)
        self.distributions = Distributions()

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    # Loss for PPO
    def get_loss(
        self,
        action_mean,
        values,
        old_action_mean,
        old_values,
        next_values,
        actions,
        rewards,
        dones,
    ):
        # Don't use old value in backpropagation
        Old_values = tf.stop_gradient(old_values)

        # Getting general advantages estimator
        Advantages = self.policy_function.generalized_advantage_estimation(
            values, rewards, next_values, dones
        )
        Returns = tf.stop_gradient(Advantages + values)
        Advantages = tf.stop_gradient(
            (Advantages - tf.math.reduce_mean(Advantages))
            / (tf.math.reduce_std(Advantages) + 1e-6)
        )

        # Finding the ratio (pi_theta / pi_theta__old):
        logprobs = self.distributions.logprob(action_mean, self.std, actions)
        Old_logprobs = tf.stop_gradient(
            self.distributions.logprob(old_action_mean, self.std, actions)
        )
        ratios = tf.math.exp(
            logprobs - Old_logprobs
        )  # ratios = old_logprobs / logprobs

        # Finding KL Divergence
        Kl = self.distributions.kl_divergence(
            old_action_mean, self.std, action_mean, self.std
        )

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss = tf.where(
            tf.logical_and(Kl >= self.policy_kl_range, ratios >= 1),
            ratios * Advantages - self.policy_params * Kl,
            ratios * Advantages - self.policy_kl_range,
        )
        pg_loss = tf.math.reduce_mean(pg_loss)

        # Getting entropy from the action probability
        dist_entropy = tf.math.reduce_mean(
            self.distributions.entropy(action_mean, self.std)
        )

        # Getting critic loss by using Clipped critic value
        vpredclipped = old_values + tf.clip_by_value(
            values - Old_values, -self.value_clip, self.value_clip
        )  # Minimize the difference between old value and new value
        vf_losses1 = tf.math.square(Returns - values) * 0.5  # Mean Squared Error
        vf_losses2 = tf.math.square(Returns - vpredclipped) * 0.5  # Mean Squared Error
        critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2))

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        loss = (
            (critic_loss * self.vf_loss_coef)
            - (dist_entropy * self.entropy_coef)
            - pg_loss
        )
        return loss

    @tf.function
    def act(self, state):
        state = tf.expand_dims(tf.cast(state, dtype=tf.float32), 0)
        action_mean = self.actor(state)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.distributions.sample(action_mean, self.std)
        else:
            action = action_mean

        return tf.squeeze(action, 0)

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states):
        with tf.GradientTape() as tape:
            action_mean, values = self.actor(states), self.critic(states)
            old_action_mean, old_values = self.actor_old(states), self.critic_old(
                states
            )
            next_values = self.critic(next_states)

            loss = self.get_loss(
                action_mean,
                values,
                old_action_mean,
                old_values,
                next_values,
                actions,
                rewards,
                dones,
            )

        gradients = tape.gradient(
            loss, self.actor.trainable_variables + self.critic.trainable_variables
        )
        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.actor.trainable_variables + self.critic.trainable_variables,
            )
        )

    # Update the model
    def update_ppo(self):
        # print("updating ppo..")
        batch_size = int(len(self.memory) / self.minibatch)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):
            for (
                states,
                actions,
                rewards,
                dones,
                next_states,
            ) in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(states, actions, rewards, dones, next_states)

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_weights(self):
        self.actor.save_weights("ppo_models/actor_ppo", save_format="tf")
        self.actor_old.save_weights("ppo_models/actor_old_ppo", save_format="tf")
        self.critic.save_weights("ppo_models/critic_ppo", save_format="tf")
        self.critic_old.save_weights("ppo_models/critic_old_ppo", save_format="tf")

    def load_weights(self):
        self.actor.load_weights("ppo_models/actor_ppo")
        self.actor_old.load_weights("ppo_models/actor_old_ppo")
        self.critic.load_weights("ppo_models/critic_ppo")
        self.critic_old.load_weights("ppo_models/critic_old_ppo")


def plot(datas):
    print("----------")

    if len(datas) == 0:
        print('No data!')
        return

    if is_notebook():
        plt.plot(datas)
        plt.plot()
        plt.xlabel("Episode")
        plt.ylabel("Datas")
        plt.show()

    print("Max :", np.max(datas))
    print("Min :", np.min(datas))
    print("Avg :", np.mean(datas))


highest = -999


def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update):
    global highest
    ############################################
    state = env.reset()
    done = False
    total_reward = 0
    n_steps_this_ep = 0
    ############################################

    while not done:
        action = agent.act(state).numpy()
        next_state, reward, done, _ = env.step(action)

        #if n_steps_this_ep == 0:
        #    print(f"first step:\nreward: {reward:03f}\nnext_state: {next_state}")

        n_steps_this_ep += 1
        t_updates += 1
        total_reward += reward

        if training_mode:
            agent.save_eps(
                state.tolist(), action, reward, float(done), next_state.tolist()
            )

        state = next_state

        if render:
            env.render()

        if training_mode:
            if t_updates % n_update == 0:
                agent.update_ppo()
                t_updates = 0

        if done:
            if total_reward > highest:
                env.render
                print("RECORD!", total_reward)
                highest = total_reward
            return total_reward, n_steps_this_ep, t_updates


def main(
    motor,
    n_update=50,
    learning_rate=3e-4,
    entropy_coef=0.01,
    action_muting=0.5,
    PPO_epochs=5,
    steps_per_episode=100,
    goal_type="bumps",
    n_episode=1000,
    minibatch=32,
    fixed_seed=False
):

    # 
    # Default hyperparams
    # 

    load_weights = False  # If you want to load the agent, set this to True
    save_weights = True  # If you want to save the agent, set this to True
    training_mode = True  # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = 300  # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    using_google_drive = False

    render = False  # If you want to display the image. Turn this off if you run this in Google Collab
    # n_update            = 50 # How many episode before you update the Policy. ocommended set to 128 for Discrete
    # n_steps_per_episode = 100 # How many steps per episode, before reset?
    n_plot_batch = 1000  # How many episode you want to plot the result
    # n_episode = 1000  # How many episode you want to run
    n_saved = 100  # How many episode to run before saving the weights

    policy_kl_range = 0.03  # Set to 0.0008 for Discrete
    policy_params = 5  # Set to 20 for Discrete
    value_clip = 2 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    # entropy_coef        = 0.01 # How much randomness of action you will get
    vf_loss_coef = 1.0  # Just set to 1
    # minibatch = 32  # How many batch per update. size of batch = n_update / minibatch. Rocommended set to 4 for Discrete
    # PPO_epochs          = 5 # How many epoch per update

    gamma = 0.99  # Just set to 0.99
    lam = 0.95  # Just set to 0.95
    # learning_rate       = 3e-4 # 3e-4 # Just set to 0.95

    env_name = ENV_NAME
    env = gym.make(env_name)
    env.motor = motor
    env.action_muting = action_muting
    env.steps_per_episode = steps_per_episode
    env.goal = goal_type

    if fixed_seed:
        print(f'USING FIXED SEED: {fixed_seed}')
        random.seed(fixed_seed)
        tf.random.set_seed(fixed_seed)
        env.seed(fixed_seed)
        env.action_space.seed(fixed_seed)
        env.observation_space.seed(fixed_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(action_dim)

    agent = Agent(
        state_dim,
        action_dim,
        training_mode,
        policy_kl_range,
        policy_params,
        value_clip,
        entropy_coef,
        vf_loss_coef,
        minibatch,
        PPO_epochs,
        gamma,
        lam,
        learning_rate,
    )
    #############################################
    if using_google_drive:
        from google.colab import drive

        drive.mount("/test")

    if load_weights:
        agent.load_weights()
        print("Weight Loaded")

    rewards = []
    batch_rewards = []
    batch_solved_reward = []

    times = []
    batch_times = []

    total_steps = 0
    t_updates = 0

    for i_episode in range(1, n_episode + 1):
        total_reward, steps_this_ep, t_updates = run_episode(
            env, agent, state_dim, render, training_mode, t_updates, n_update
        )
        msg = "Episode {} \t t_reward: {} \t steps: {} \t ".format(
                i_episode, total_reward, steps_this_ep
            )
        print(msg)
        open('result-log.txt', 'a').write(msg+"\n")
        total_steps += steps_this_ep
        batch_rewards.append(int(total_reward))
        batch_times.append(steps_this_ep)

        if save_weights:
            if i_episode % n_saved == 0:
                agent.save_weights()
                print("weights saved")

        if reward_threshold:
            if len(batch_solved_reward) == 100:
                if np.mean(batch_solved_reward) >= reward_threshold:
                    for reward in batch_rewards:
                        rewards.append(reward)

                    for time in batch_times:
                        times.append(time)

                    print("You solved task after {} episode".format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)

        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)

            for reward in batch_rewards:
                rewards.append(reward)

            for time in batch_times:
                times.append(time)

            batch_rewards = []
            batch_times = []

            print("========== Cummulative ==========")
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)

    for reward in batch_rewards:
        rewards.append(reward)

    for time in batch_times:
        times.append(time)

    print("Configuration:\n", repr(locals()))
    print("========== Final ==========")
    # Plot the reward, times for every episode
    plot(rewards)
    plot(times)
    rmn = np.min(rewards)
    rmx = np.max(rewards)
    rma = np.mean(rewards)
    n_goals = env.n_goals
    goals_per_step = n_goals / total_steps
    goals_per_episode = n_goals / i_episode
    print(f"# GOALS: {n_goals}\nGOALS PER STEP: {goals_per_step:04f}\nGOALS PER EPISODE: {goals_per_episode:04f}")
    env.close()
    return {
        "n_goals": n_goals,
        "rewards": {"min": rmn, "max": rmx, "mean": rma},
        "total_steps": total_steps,
        "mean_steps": np.mean(times),
        "goals_per_step": goals_per_step,
        "goals_per_episode": goals_per_episode
    }


if __name__ == "__main__":
    main()
