
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import matplotlib.pyplot as plt


# ==== Stage 1: Setup ====
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
writer = SummaryWriter()

lr = 1e-5
clip_param = 0.2
lambd_param = 0.95
gamma_param = 0.99
num_epochs_list = [20]
num_episodes = 512
ppo_batch_size = 512
mini_batch_size = 64
ppo_epoch = 4
entropy_coef_param = 0.01

input_channel = 23
action_size = 6

sorted_obs_order = [
    "CameraSensor_1", "CameraSensor_2", "CameraSensor_3",
    "AgentSensor", "AgentSensor_UP", "AgentSensor_DOWN", "AgentSensor_RIGHT", "AgentSensor_LEFT",
    "OtherAgentSensor", "OtherAgentSensor_UP", "OtherAgentSensor_DOWN", "OtherAgentSensor_RIGHT", "OtherAgentSensor_LEFT",
    "TableSensor", "OnionDispSensor", "DishDispSensor", "ServeDispSensor", "OvenDispSensor",
    "OnionSensor", "DishSensor", "SoupSensor", "CookingTimerSensor", "IsCookedSensor"
]

agent_movement_directions = ["LEFT", "RIGHT", "UP", "DOWN"]


# ==== Stage 2: Model Definitions ====
class Actor(nn.Module):
    def __init__(self, input_channel, action_dim):
        super(Actor, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channel, 96, 4, 4),
            nn.Conv2d(96, 96, 7),
            nn.Conv2d(96, 384, 1),
            nn.ReLU(),
            nn.Conv2d(384, 96, 1),
            nn.Conv2d(96, 192, 7),
            nn.Conv2d(192, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 192, 1),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(43200, action_dim)
        self.log_std = nn.Parameter(torch.ones((1, action_dim)))

    def forward(self, observation):
        x = self.convs(observation)
        x = self.flatten(x)
        logits = self.linear(x)
        std = self.log_std.exp().expand_as(logits)
        dist = Normal(logits, std)
        actions = dist.sample()
        scaled_actions = torch.cat([
            torch.tanh(actions[..., :4]),
            torch.sigmoid(actions[..., 4:])
        ], dim=-1)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return scaled_actions, log_probs, entropy


class Critic(nn.Module):
    def __init__(self, input_channel):
        super(Critic, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_channel, 96, 4, 4),
            nn.Conv2d(96, 96, 7),
            nn.Conv2d(96, 384, 1),
            nn.ReLU(),
            nn.Conv2d(384, 96, 1),
            nn.Conv2d(96, 192, 7),
            nn.Conv2d(192, 768, 1),
            nn.ReLU(),
            nn.Conv2d(768, 192, 1),
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(43200, 1)

    def forward(self, observation):
        x = self.convs(observation)
        x = self.flatten(x)
        return self.linear(x)


# ==== Stage 3-5: PPO, Critic, GAE ====
def compute_GAE(next_value, values, rewards, gamma, lambd):
    gae = 0
    returns = []
    values = values + [next_value]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * lambd * gae
        returns.append(gae + values[step])
    returns.reverse()
    return returns

def compute_GAE_dict(next_values, values_all, rewards_all, gamma, lambd):
    agents_name = ["agent1", "agent2"]
    returns_all = {"agent1": [], "agent2": []}
    for i, name in enumerate(agents_name):
        returns_all[name] = compute_GAE(next_values[i], values_all[name], rewards_all[name], gamma, lambd)
    return returns_all

def compute_Advantage_dict(values_all, returns_all):
    advantages_all = {"agent1": [], "agent2": []}
    for agent in ["agent1", "agent2"]:
        for i in range(len(returns_all[agent])):
            advantages_all[agent].append(returns_all[agent][i] - values_all[agent][i])
    return advantages_all


# Placeholder for Stage 6–8
def setupUnityEnv(url, time_scale=20.0):
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(time_scale=time_scale)
    env = UnityEnvironment(file_name=url, side_channels=[engine_channel])
    env.reset()
    return env

def addDirectionLayer(layer_dict, direction_obs):
    x_axis, y_axis = direction_obs.squeeze(0)
    for direction in agent_movement_directions:
        layer_dict[f"AgentSensor_{direction}"] = np.zeros_like(layer_dict["AgentSensor"])
    def chooseDirectionLayer(axis, axis_name):
        return {"x": ["LEFT", "RIGHT"], "y": ["DOWN", "UP"]}[axis_name][0 if axis < 0 else 1] if axis != 0 else None
    for d in [chooseDirectionLayer(x_axis, "x"), chooseDirectionLayer(y_axis, "y")]:
        if d: layer_dict[f"AgentSensor_{d}"] = layer_dict["AgentSensor"]
    return layer_dict

def addOtherAgentLayers(layer_dict, agent_name, other_agent_name):
    map_keys = {
        "OtherAgentSensor": "AgentSensor",
        "OtherAgentSensor_UP": "AgentSensor_UP",
        "OtherAgentSensor_DOWN": "AgentSensor_DOWN",
        "OtherAgentSensor_RIGHT": "AgentSensor_RIGHT",
        "OtherAgentSensor_LEFT": "AgentSensor_LEFT",
    }
    for k, v in map_keys.items():
        layer_dict[agent_name][k] = layer_dict[other_agent_name][v]
    return layer_dict[agent_name]

def plot_and_save_graph(total_rewards_all):
    for rewards_per_epoch in total_rewards_all:
        x = list(range(len(rewards_per_epoch)))
        y1 = [pair[0] for pair in rewards_per_epoch]
        y2 = [pair[1] for pair in rewards_per_epoch]
        plt.figure()
        plt.plot(x, y1, label="Agent 1", linewidth=2)
        plt.plot(x, y2, label="Agent 2", linewidth=2)
        plt.title("Total Rewards per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        filename = f"reward_graph_{len(rewards_per_epoch)}_epochs.png"
        plt.savefig(filename, dpi=300)
        plt.show()

def concat_tensor_dict(trajectory_dict):
    return {k: torch.cat(v) for k, v in trajectory_dict.items()}

def convert_list_into_tensor_dict(trajectory_dict):
    return {k: torch.stack(v) for k, v in trajectory_dict.items()}

def calc_total_reward(rewards_all):
    return [sum(r for r in rewards if r > 0) for rewards in rewards_all.values()]

def calc_ppo_loss(
    ppo_epoch,
    batch_size,
    num_agents,
    states_agent1,
    states_agent2,
    log_probs_agent1,
    log_probs_agent2,
    entropy_agent1,
    entropy_agent2,
    advantages_agent1,
    advantages_agent2,
    sigma_param,
):
    def ppo_iter():
        for _ in range(batch_size // mini_batch_size):
            batch_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield (
                states_agent1[batch_ids], states_agent2[batch_ids],
                log_probs_agent1[batch_ids], log_probs_agent2[batch_ids],
                entropy_agent1[batch_ids], entropy_agent2[batch_ids],
                advantages_agent1[batch_ids], advantages_agent2[batch_ids]
            )

    for _ in range(ppo_epoch):
        for (
            state_a1, state_a2,
            old_logp_a1, old_logp_a2,
            entropy_a1, entropy_a2,
            adv_a1, adv_a2
        ) in ppo_iter():

            _, new_logp_a1, new_entropy_a1 = agent_model(state_a1)
            _, new_logp_a2, new_entropy_a2 = agent_model(state_a2)

            ratio_a1 = (new_logp_a1 - old_logp_a1).exp()
            ratio_a2 = (new_logp_a2 - old_logp_a2).exp()

            clipped_a1 = torch.clamp(ratio_a1, 1 - clip_param, 1 + clip_param)
            clipped_a2 = torch.clamp(ratio_a2, 1 - clip_param, 1 + clip_param)

            loss_a1 = -torch.min(ratio_a1 * adv_a1.unsqueeze(1), clipped_a1 * adv_a1.unsqueeze(1)).mean()
            loss_a2 = -torch.min(ratio_a2 * adv_a2.unsqueeze(1), clipped_a2 * adv_a2.unsqueeze(1)).mean()

            entropy_bonus = -(new_entropy_a1.mean() + new_entropy_a2.mean())

            loss = (loss_a1 + loss_a2 + sigma_param * entropy_bonus) / num_agents

            agent_optimizer.zero_grad()
            loss.backward()
            agent_optimizer.step()

def mappo_update(ppo_epoch, batch_size, agents, states, log_probs, entropies, advs, sigma_param):
    calc_ppo_loss(ppo_epoch, batch_size, len(agents),
                  states[agents[0]], states[agents[1]],
                  log_probs[agents[0]], log_probs[agents[1]],
                  entropies[agents[0]], entropies[agents[1]],
                  advs[agents[0]], advs[agents[1]],
                  sigma_param)

def calc_critic_loss(epochs, batch_size, mini_batch_size, num_agents, s1, s2, v1, v2, r1, r2):
    def critic_iter():
        for _ in range(batch_size // mini_batch_size):
            ids = np.random.randint(0, batch_size, mini_batch_size)
            yield s1[ids], s2[ids], v1[ids], v2[ids], r1[ids], r2[ids]

    for _ in range(epochs):
        for st1, st2, val1, val2, ret1, ret2 in critic_iter():
            new_val1 = critic_model(st1)
            new_val2 = critic_model(st2)
            surr1_1 = (new_val1 - ret1).pow(2).mean()
            surr1_2 = (new_val2 - ret2).pow(2).mean()
            surr2_1 = (torch.clamp(new_val1, val1 - clip_param, val1 + clip_param) - ret1).pow(2).mean()
            surr2_2 = (torch.clamp(new_val2, val2 - clip_param, val2 + clip_param) - ret2).pow(2).mean()
            loss = (max(surr1_1, surr2_1) + max(surr1_2, surr2_2)) / num_agents
            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

def critic_update(epochs, batch_size, mini_batch_size, agents, states, values, returns):
    calc_critic_loss(epochs, batch_size, mini_batch_size, len(agents),
                     states[agents[0]], states[agents[1]],
                     values[agents[0]], values[agents[1]],
                     returns[agents[0]], returns[agents[1]])

# Final ONNX export
def export_model_to_onnx(model, example_input, path="exported_model.onnx"):
    torch.onnx.export(
        model,
        example_input.unsqueeze(0),
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )

def main():
    env_path = "/home/geralt/Downloads/CAI_platform_unity-main/test.x86_64"  # Replace with your Unity build path
    env = setupUnityEnv(env_path)
    agent_names = list(env.behavior_specs)
    agent1, agent2 = agent_names

    for num_epochs in num_epochs_list:
        global agent_model, critic_model, agent_optimizer, critic_optimizer

        agent_model = Actor(input_channel, action_size).to(device)
        critic_model = Critic(input_channel).to(device)
        agent_optimizer = optim.Adam(agent_model.parameters(), lr=lr)
        critic_optimizer = optim.Adam(critic_model.parameters(), lr=lr)

        total_rewards_epochs = []

        for epoch in range(num_epochs):
            states_all = {"agent1": [], "agent2": []}
            log_probs_all = {"agent1": [], "agent2": []}
            entropy_all = {"agent1": [], "agent2": []}
            values_all = {"agent1": [], "agent2": []}
            rewards_all = {"agent1": [], "agent2": []}

            env.reset()
            for episode in range(num_episodes):
                agents_obs = {}
                agents_obs_array = []
                dones = []

                for agent_name in agent_names:
                    spec = env.behavior_specs[agent_name]
                    decision_steps, terminal_steps = env.get_steps(agent_name)

                    if len(terminal_steps) > 0:
                        dones.append(1)
                        break
                    else:
                        dones.append(0)

                    agent_index = 0 if agent_name == agent1 else 1
                    reward = decision_steps.reward[0]
                    rewards_all[f"agent{agent_index+1}"].append(reward)

                    temp_input = {}
                    for index, obs_spec in enumerate(spec.observation_specs):
                        obs = decision_steps.obs[index]
                        if len(obs_spec.shape) == 3 and obs_spec.name == "CameraSensor":
                            for i in range(len(obs[0])):
                                temp_input[f"{obs_spec.name}_{i+1}"] = obs[0][i]
                        elif len(obs_spec.shape) == 3:
                            temp_input[obs_spec.name] = obs[0][0]
                        else:
                            temp_input = addDirectionLayer(temp_input, obs)

                    agents_obs[agent_name] = temp_input

                if dones and dones[-1] == 1:
                    break

                agents_obs[agent1] = addOtherAgentLayers(agents_obs, agent1, agent2)
                agents_obs[agent2] = addOtherAgentLayers(agents_obs, agent2, agent1)

                for name in agent_names:
                    obs_vector = [agents_obs[name][key] for key in sorted_obs_order]
                    agents_obs_array.append(obs_vector)

                obs_tensor = torch.tensor(np.array(agents_obs_array), dtype=torch.float32).to(device)
                actions, log_probs, entropy = agent_model(obs_tensor)
                values = critic_model(obs_tensor)

                for i, agent_name in enumerate(agent_names):
                    aid = f"agent{i+1}"
                    states_all[aid].append(obs_tensor[i])
                    log_probs_all[aid].append(log_probs[i].detach())
                    entropy_all[aid].append(entropy[i].detach())
                    values_all[aid].append(values[i].detach())
                    action_tensor = actions[i].unsqueeze(0)
                    env.set_actions(agent_name, ActionTuple(continuous=action_tensor.cpu().numpy()))

                env.step()

            next_values = []
            for agent_name in agent_names:
                spec = env.behavior_specs[agent_name]
                decision_steps, _ = env.get_steps(agent_name)
                temp_input = {}
                for index, obs_spec in enumerate(spec.observation_specs):
                    obs = decision_steps.obs[index]
                    if len(obs_spec.shape) == 3 and obs_spec.name == "CameraSensor":
                        for i in range(len(obs[0])):
                            temp_input[f"{obs_spec.name}_{i+1}"] = obs[0][i]
                    elif len(obs_spec.shape) == 3:
                        temp_input[obs_spec.name] = obs[0][0]
                    else:
                        temp_input = addDirectionLayer(temp_input, obs)

                agents_obs[agent_name] = temp_input

            agents_obs[agent1] = addOtherAgentLayers(agents_obs, agent1, agent2)
            agents_obs[agent2] = addOtherAgentLayers(agents_obs, agent2, agent1)

            final_obs_array = []
            for name in agent_names:
                obs_vector = [agents_obs[name][key] for key in sorted_obs_order]
                final_obs_array.append(obs_vector)

            final_tensor = torch.tensor(np.array(final_obs_array), dtype=torch.float32).to(device)
            next_values_tensor = critic_model(final_tensor).squeeze().detach()
            next_values = [next_values_tensor[i] for i in range(len(agent_names))]

            returns_all = compute_GAE_dict(next_values, values_all, rewards_all, gamma_param, lambd_param)
            advantages_all = compute_Advantage_dict(values_all, returns_all)

            states_all = convert_list_into_tensor_dict(states_all)
            log_probs_all = convert_list_into_tensor_dict(log_probs_all)
            entropy_all = convert_list_into_tensor_dict(entropy_all)
            values_all = concat_tensor_dict(values_all)
            returns_all = convert_list_into_tensor_dict(returns_all)
            advantages_all = concat_tensor_dict(advantages_all)

            dict_keys = ["agent1", "agent2"]
            total_rewards_epochs.append(calc_total_reward(rewards_all))

            mappo_update(ppo_epoch, ppo_batch_size, dict_keys, states_all, log_probs_all, entropy_all, advantages_all, entropy_coef_param)
            critic_update(ppo_epoch, ppo_batch_size, mini_batch_size, dict_keys, states_all, values_all, returns_all)

        total_rewards_all = [total_rewards_epochs]
        export_model_to_onnx(agent_model, states_all["agent1"][0], f"agent_model_{num_epochs}_epochs.onnx")
        plot_and_save_graph(total_rewards_all)

    env.close()


if __name__ == "__main__":
    main()