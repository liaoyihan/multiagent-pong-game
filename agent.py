
import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(AttentionLayer, self).__init__()

        self.W_q = nn.Linear(input_size, input_size)
        self.W_k = nn.Linear(input_size, input_size)
        self.W_v = nn.Linear(input_size, input_size)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        output = torch.matmul(attention_weights, v)

        return output


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        # self.bn0 = nn.BatchNorm1d(state_size)
        self.attention1 = AttentionLayer(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)

        self.attention2 = AttentionLayer(128)

        self.dropout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.attention3 = AttentionLayer(64)
        self.fc3 = nn.Linear(64, action_size)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # x = self.bn0(x)
        x = self.attention1(x)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.attention2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        # x = self.attention3(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        # self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # x = self.bn0(x)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)

    def forward(self, state):

        act = self.actor(state)
        value = self.critic(state)
        return act, value

    def predict(self, state):
        self.actor.eval()
        self.critic.eval()
        act = self.actor(state)
        value = self.critic(state)
        act = act.detach().numpy()
        value = value.detach().numpy()

        return act, value






if __name__ == '__main__':

    pass