import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# --- パラメータ設定セクション ---
config = {
    "embed_dim": 1024,
    "num_heads": 8,
    "num_layers": 2,
    "action_dim": 1,
    "state_dim": 1,
    "num_heart": 1000,
    "memory_length": 10,
    "reward_scale": 10,
    "gamma": 0.99,  # 割引率をここに追加
    "learning_rate": 0.0001,
    "num_episodes": 1000,
    "steps_per_episode": 10,
    "teacher_data_ids": {
        0: 3506,
        90: 929,
        180: 9464,
        270: 2902
    }
}

# Transformer Block with QKV Attention for State-Action Modeling
class TransformerBlockWithQKV(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlockWithQKV, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, memory):
        attn_output, _ = self.attention(x, memory, memory)
        x = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.layernorm2(x + ff_output)
        return x

# 修正されたReinforcement Learning Model
class StateActionTransformerWithQKV(nn.Module):
    def __init__(self, state_dim, action_dim, embed_dim, num_heads, num_layers, num_heart, memory_length):
        super(StateActionTransformerWithQKV, self).__init__()
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.transformer_blocks1 = nn.ModuleList(
            [TransformerBlockWithQKV(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc2 = nn.Linear(embed_dim, num_heart)
        self.embedding = nn.Embedding(10000, embed_dim)
        self.transformer_blocks2 = nn.ModuleList(
            [TransformerBlockWithQKV(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc3 = nn.Linear(embed_dim, action_dim)
        
        # メモリバッファの初期化
        self.memory_length = memory_length
        self.memory = torch.zeros(memory_length, 1, embed_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x)).unsqueeze(0).unsqueeze(1)
        for transformer_block in self.transformer_blocks1:
            x = transformer_block(x, self.memory)
        
        x = self.fc2(x.squeeze(0).squeeze(0))
        x = torch.round(x * 10000).int()
        
        zero_indices = (x == 0).nonzero(as_tuple=True)[0]
        if len(zero_indices) > 1:
            x[zero_indices[1]:] = 0

        x = self.embedding(x.clamp(0, 9999)).unsqueeze(0)
        for transformer_block in self.transformer_blocks2:
            x = transformer_block(x, x)

        x = self.fc3(x.mean(dim=1))
        return x

# Generator Model
class Generator(nn.Module):
    def __init__(self, state_dim, embed_dim, num_heads, num_layers, num_heart):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(state_dim, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockWithQKV(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc2 = nn.Linear(embed_dim, num_heart)

    def forward(self, x):
        x = torch.relu(self.fc1(x)).unsqueeze(0)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, x)
        x = self.fc2(x).squeeze(0)
        return torch.round(x * 10000).int()

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(10000, embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockWithQKV(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(embed_dim, 1)

    def forward(self, x):
        x = self.embedding(x.clamp(0, 9999)).unsqueeze(0)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, x)
        x = x.mean(dim=1)
        return torch.sigmoid(self.fc(x))

def initialize_models(config):
    rl_model = StateActionTransformerWithQKV(
        config["state_dim"], config["action_dim"], config["embed_dim"], config["num_heads"], config["num_layers"], config["num_heart"], config["memory_length"]
    )
    generator = Generator(
        config["state_dim"], config["embed_dim"], config["num_heads"], config["num_layers"], config["num_heart"]
    )
    discriminator = Discriminator(config["num_heart"], config["embed_dim"], config["num_heads"], config["num_layers"])
    return rl_model, generator, discriminator

def initialize_optimizers(models, learning_rate):
    rl_optimizer = optim.Adam(models[0].parameters(), lr=learning_rate)
    gen_optimizer = optim.Adam(models[1].parameters(), lr=learning_rate)
    disc_optimizer = optim.Adam(models[2].parameters(), lr=learning_rate)
    return rl_optimizer, gen_optimizer, disc_optimizer

def compute_discounted_rewards(rewards, gamma):
    discounted_rewards = []
    cumulative_reward = 0
    for reward in reversed(rewards):
        cumulative_reward = reward + gamma * cumulative_reward
        discounted_rewards.insert(0, cumulative_reward)
    return discounted_rewards

class TrainingController:
    def __init__(self, rl_model, generator, discriminator, rl_optimizer, gen_optimizer, disc_optimizer, rl_criterion, gen_criterion, disc_criterion, reward_scale):
        self.rl_model = rl_model
        self.generator = generator
        self.discriminator = discriminator
        self.rl_optimizer = rl_optimizer
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.rl_criterion = rl_criterion
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.reward_scale = reward_scale

    def train_rl(self, states, discounted_rewards):
        self.rl_optimizer.zero_grad()
        rl_loss = -torch.tensor(discounted_rewards, requires_grad=True).sum() 
        rl_loss.backward()
        self.rl_optimizer.step()
        return rl_loss.item()

    def train_gan(self, state, target_language_ids):
        self.gen_optimizer.zero_grad()
        generated_ids = self.generator(state)
        fake_output_for_generator = self.discriminator(generated_ids)
        gen_loss = self.gen_criterion(fake_output_for_generator, torch.ones_like(fake_output_for_generator))
        gen_loss.backward()
        self.gen_optimizer.step()
        
        self.disc_optimizer.zero_grad()
        real_output = self.discriminator(target_language_ids)
        real_loss = self.disc_criterion(real_output, torch.ones_like(real_output))
        fake_output = self.discriminator(generated_ids.detach())
        fake_loss = self.disc_criterion(fake_output, torch.zeros_like(fake_output))
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        self.disc_optimizer.step()
        
        discriminator_reward = fake_output_for_generator.mean().item()
        return gen_loss.item(), disc_loss.item(), discriminator_reward


# Simulation Environment Class Definition
class SimulationEnvironment:
    def __init__(self, agent_start_pos=(50, 50), num_food=100, boundary=100):
        self.agent_pos = np.array(agent_start_pos, dtype=float)
        self.agent_angle = 0
        self.num_food = num_food
        self.boundary = boundary
        self.food_positions = [self._random_position() for _ in range(num_food)]
        self.food_radius = 1.0
        self.step_distance = 1.0
        self.reward_value = 1.0

    def _random_position(self):
        return np.array([random.uniform(0, self.boundary), random.uniform(0, self.boundary)])

    def _angle_to_food(self):
        distances = [np.linalg.norm(food - self.agent_pos) for food in self.food_positions]
        nearest_food_idx = np.argmin(distances)
        nearest_food = self.food_positions[nearest_food_idx]
        delta_x, delta_y = nearest_food - self.agent_pos
        angle_to_food = np.degrees(np.arctan2(delta_y, delta_x)) % 360
        return angle_to_food, nearest_food_idx, distances[nearest_food_idx]

    def reset(self):
        self.agent_pos = np.array([self.boundary / 2, self.boundary / 2], dtype=float)
        self.agent_angle = 0
        self.food_positions = [self._random_position() for _ in range(self.num_food)]
        return self._angle_to_food()[0]

    def step(self, action_angle):
        self.agent_angle = action_angle
        dx = self.step_distance * np.cos(np.radians(self.agent_angle))
        dy = self.step_distance * np.sin(np.radians(self.agent_angle))
        self.agent_pos += np.array([dx, dy])
        self.agent_pos = np.clip(self.agent_pos, 0, self.boundary)
        angle_to_food, nearest_food_idx, distance_to_food = self._angle_to_food()
        reward = 0.0
        if distance_to_food <= self.food_radius:
            reward = self.reward_value
            self.food_positions[nearest_food_idx] = self._random_position()
        return angle_to_food, reward

# Teacher data setup for the Discriminator model
def create_teacher_data(config):
    teacher_data = {}
    for direction, id_value in config["teacher_data_ids"].items():
        ids = [id_value] + [0] * (config["num_heart"] - 1)  # IDリストを1000次元にパディング
        teacher_data[direction] = torch.tensor(ids, dtype=torch.long)
    return teacher_data

teacher_data = create_teacher_data(config)

# Training loop
rl_model, generator, discriminator = initialize_models(config)
rl_optimizer, gen_optimizer, disc_optimizer = initialize_optimizers([rl_model, generator, discriminator], config["learning_rate"])
env = SimulationEnvironment()
controller = TrainingController(
    rl_model=rl_model,
    generator=generator,
    discriminator=discriminator,
    rl_optimizer=rl_optimizer,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    rl_criterion=nn.MSELoss(),
    gen_criterion=nn.BCELoss(),
    disc_criterion=nn.BCELoss(),
    reward_scale=config["reward_scale"]
)

for episode in range(config["num_episodes"]):
    state = torch.tensor([env.reset()], dtype=torch.float32)
    rewards = []
    states = []

    for step in range(config["steps_per_episode"]):
        action_vector = generator(state)
        action = action_vector[0].item()
        angle_to_food, proximity_reward = env.step(action)
        state = torch.tensor([angle_to_food], dtype=torch.float32)

        target_direction = int(state.item()) % 360
        target_language_ids = teacher_data.get(target_direction, teacher_data[0])
        gen_loss, disc_loss, discriminator_reward = controller.train_gan(state, target_language_ids)

        rewards.append(proximity_reward + config["reward_scale"] * discriminator_reward)
        states.append(state)

    # エピソード終了後に割引報酬を計算し、RLモデルに渡す
    discounted_rewards = compute_discounted_rewards(rewards, config["gamma"])
    rl_loss = controller.train_rl(states, discounted_rewards)

    print(f"Episode {episode}, Generator Loss: {gen_loss}, Discriminator Loss: {disc_loss}, RL Loss: {rl_loss}, Total Rewards: {sum(rewards)}")
