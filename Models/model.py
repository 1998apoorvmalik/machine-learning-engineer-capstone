import sys
import time
import random
import PIL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Utility.HuberLoss as Hloss
from keras.models import Sequential
from keras.layers import Dense, Dropout, regularizers, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam, RMSprop
from Utility.ReplayMemory import ReplayBuffer

class DQNAgent:
    
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n
        self.replay_buffer = ReplayBuffer(buffer_size = 1000000, batch_size = 32)
        self.target_update_frequency = 16
        self.target_update_counter = 0
        self.gamma = 0.95
        self.initial_epsilon = 1
        self.epsilon = self.initial_epsilon
        self.epsilon_decay_rate = 0.99995
        self.min_epsilon = 0.01
        self.rho = 0.95
        self.learning_rate = 0.00025
        self.training_scores = []
        
        # Main model
        self.model = self.build_model()

        # Target model
        self.target_model = self.build_model()
        # Set the weights of the target model to that of the main model.
        self.target_model.set_weights(self.model.get_weights())
        
        
    def build_model(self):
        # Neural Network architecture for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu', input_shape=self.state_size))
        model.add(Conv2D(filters=32, kernel_size=4, strides=2, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
                         
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=Hloss.huber_loss_mean, optimizer=RMSprop(lr=self.learning_rate, rho=self.rho, epsilon=self.min_epsilon), metrics=["accuracy"])
        model.summary()
        return model
    
    
    def reset_episode(self, initial_state):
        """Reset variables for a new episode."""
        # Gradually decrease exploration rate
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.min_epsilon)
        
        self.prev_state = self.preprocess_state(initial_state)
        self.prev_action = np.argmax(self.model.predict(self.prev_state))
        return self.prev_action
    
    def preprocess_state(self, state):
        # Preprocessing code
        return np.expand_dims(np.array(state), axis=0)
    
    
    def reset_exploration(self, epsilon=None):
        """Reset exploration rate used when training."""
        self.epsilon = epsilon if epsilon is not None else self.initial_epsilon
    
    
    def plot_scores(self, scores, rolling_window=100):
        """Plot scores and optional rolling mean using specified window."""
        plt.title("Scores")
        plt.xlabel("Episodes -->")
        plt.ylabel("Scores -->")
        plt.plot(scores)
        rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
        plt.plot(rolling_mean);
    
    
    def act(self, next_state, reward, done, mode="train", time_delay=None):
        """Pick next action and update weights of the neural network (when mode != 'test')."""
        next_state = self.preprocess_state(next_state)
        if mode == "test": 
            # Test mode: Simply produce an action
            action = np.argmax(self.model.predict(next_state))
            if time_delay != None:        # Adding time delay to watch the agent perform at a little slower pace.
                time.sleep(time_delay)
        else:
            # Exploration vs. exploitation
            do_exploration = np.random.uniform(0, 1) < self.epsilon
            if do_exploration:
                # Pick a random action
                action = np.random.randint(0, self.action_size)
            else:
                # Pick the best action from Q table
                action = np.argmax(self.model.predict(next_state))
            
            # Store the experience in replay memory
            self.replay_buffer.add(self.prev_state, self.prev_action, reward, next_state, done)
            
            # Learn
            self.replay(done)
            
        # Roll over current state, action for next step
        self.prev_state = next_state
        self.prev_action = action
        return action
    
    
    def replay(self, done):
        if self.replay_buffer.size() < self.replay_buffer.batch_size:
            return 
        
        terminal_state = done        # Determine if the episode has ended.
        minibatch = self.replay_buffer.sample()
        
        # X : states, y : predictions
        X = []
        y = []
        
        # We use the main network to predict the current Q-values
        prev_states = np.array([transition[0][0] for transition in minibatch])
        prev_qs = self.model.predict(prev_states)
        
        # We use the target network to predict the future Q-values
        next_states = np.array([transition[3][0] for transition in minibatch])
        next_qs = self.target_model.predict(next_states)
        
        for index, (prev_state, prev_action, reward, next_state, done) in enumerate(minibatch):
            # Setting the target for the model to improve upon.
            if not done:
                target = reward + (self.gamma * np.max(next_qs[index]))
            else:
                target = reward

            new_q_value = prev_qs[index]
            new_q_value[prev_action] = target

            X.append(prev_state)
            y.append(new_q_value)

        # Fit on all samples as one batch.
        self.model.fit(np.vstack(X), np.vstack(y), batch_size=self.replay_buffer.batch_size, 
                       verbose=0, shuffle=False)
        
        # If terminal state is encountered, increase the update counter.
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > self.target_update_frequency:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            
    
    def run(self, num_episodes=20000, mode="train", time_delay=None, score_threshold=None, weights_path=None, 
            scores_path=None, render=True):
        
        """Run agent in given reinforcement learning environment and return scores."""
        scores = []
        max_score = -np.inf
        min_score = np.inf
        max_avg_score = -np.inf
        avg_score = -np.inf
        renders = []
        for i_episode in range(1, num_episodes+1):
            # Initialize episode
            state = self.env.reset()
            action = self.reset_episode(state)
            total_reward = 0
            done = False

            # Roll out steps until done
            while not done:
                next_state, reward, done, info = self.env.step(action)
                total_reward += reward
                action = self.act(next_state, reward, done, mode, time_delay)
                if mode == 'train':
                    self.env.render()
                else:
                    renders.append(PIL.Image.fromarray(self.env.render(mode='rgb_array')))
            # Save final score
            scores.append(total_reward)
            # Print episode stats
            if mode == 'train':
                self.training_done = True
                
                if total_reward > max_score:
                    max_score = total_reward
                    
                if total_reward < min_score:
                    min_score = total_reward
                    
                if len(scores) > 100:
                    avg_score = np.mean(scores[-100:])
                    if avg_score > max_avg_score:
                        max_avg_score = avg_score
                
                if weights_path != None and i_episode%100 == 0:
                    self.model.save_weights(weights_path)
                    if scores_path != None:
                        logs = {"scores" : scores}
                        logs = pd.DataFrame.from_dict(data=logs, orient='index')
                        logs.to_csv(scores_path ,index=False)

                print("\rEpisode {}/{} | Episode Score: {} | Min. Score: {} | Max. Score: {} | Current Avg. Score: {} | Max. Average Score: {} | epsilon: {}"
                      .format(i_episode, num_episodes, total_reward, min_score, max_score, 
                              avg_score, max_avg_score, self.epsilon), end="")
                sys.stdout.flush()
            
            # Terminating loop if the agent achieves reward threshold
            if score_threshold != None and max_avg_score > score_threshold:
                print("\nEnvironment solved after {} episodes".format(i_episode))
                break
        
        # Close rendering
        self.env.close()
            
        if mode == "test":
            return renders, np.sum(scores)
        else:
            self.training_scores.append(scores)