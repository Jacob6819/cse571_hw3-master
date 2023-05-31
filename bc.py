import torch
import torch.optim as optim
import numpy as np
from utils import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50,
                       batch_size=32):
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy.
    optimizer = optim.Adam(list(policy.parameters()))
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs) * episode_length // batch_size
    losses = []
    criterion = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        # TODO Students
        np.random.shuffle(idxs)
        flattened_obs = np.concatenate([expert_data[i]['observations'] for i in idxs])
        flattened_acts = np.concatenate([expert_data[i]['actions'] for i in idxs])
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here
            randomidxs = np.random.choice(len(flattened_obs), batch_size, replace=False)
            expert_observations, expert_actions = torch.tensor(flattened_obs[randomidxs]).float().to(
                device), torch.tensor(flattened_acts[randomidxs]).float().to(device)
            policy_data = policy(expert_observations)
            loss = criterion(policy_data, expert_actions).mean()
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 50 == 0:
            print('[%d] loss: %.8f' %
                  (epoch, running_loss / 10.))
        losses.append(loss.item())
