import numpy as np
import time
import random

class Sapper:
    def __init__(self, size=5, num_mines=3):
        self.size = size
        self.num_mines = num_mines
        self.board = np.zeros((size, size), dtype=int)
        self.revealed = np.full((size, size), False, dtype=bool)
        self.mines = set()

        # Place mines randomly
        while len(self.mines) < num_mines:
            x, y = np.random.randint(size), np.random.randint(size)
            self.mines.add((x, y))
        
        # Fill in the numbers around the mines
        for (x, y) in self.mines:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in self.mines:
                        self.board[nx, ny] += 1

    def reset(self):
        self.revealed.fill(False)
        return self.get_state()

    def get_state(self):
        state = np.where(self.revealed, self.board, -1)
        return state.flatten()

    def step(self, action):
        x, y = divmod(action, self.size)
        if self.revealed[x, y]:  # If the cell is already revealed, penalize
            return self.get_state(), -1, False
        if (x, y) in self.mines: # Hit a mine 
            return self.get_state(), -10, True  # Game over
        self.revealed[x, y] = True
        return self.get_state(), 1, False  #  Safe cell

    def render(self):
        print("\n" + "="*10)
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if self.revealed[i, j]:
                    row += f" {self.board[i, j]} " if self.board[i, j] > 0 else " . "
                else:
                    row += " ▓ "  # Closed cell
            print(row)
        print("="*10)

#  Q-learning Agent
class QAgent:
    def __init__(self, size):
        self.q_table = np.zeros((size * size, size * size)) # For storing state-action values
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.lr = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.visited = set()  # Keep track of visited cells

    def choose_action(self, state):
        available_actions = [i for i in range(len(state)) if i not in self.visited] 
        if not available_actions:  # If no available actions, end the game
            return -1
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)  # Choose a random action
        # Convert state to index
        state_idx = np.argmax(state)
        return max(available_actions, key=lambda a: self.q_table[state_idx, a])

    def update_q(self, state, action, reward, next_state):
        state_idx = np.argmax(state)
        best_next = np.max(self.q_table[next_state])
        self.q_table[state_idx, action] = (1 - self.lr) * self.q_table[state_idx, action] + self.lr * (reward + self.gamma * best_next)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# Training
env = Sapper(size=5, num_mines=3)
agent = QAgent(size=5)

episodes = 500
for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        if action == -1:  # No available actions
            break
        next_state, reward, done = env.step(action)
        agent.update_q(state, action, reward, next_state)
        state = next_state
    agent.decay_epsilon()
    if episode % 50 == 0:
        print(f"Episode {episode}, Epsilon: {agent.epsilon:.4f}")
        
# Game
print("\nThe agent is playing a real game!\n")
env.reset()
done = False
while not done:
    env.render()
    time.sleep(0.5)
    action = agent.choose_action(env.get_state())
    if action == -1:
        break
    x, y = divmod(action, env.size)
    print(f"Agent revealed cell ({x}, {y})")
    agent.visited.add(action) 
    _, _, done = env.step(action)

print("\n🏆 Game Over!")
env.render()
