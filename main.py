
import numpy as np
from environment import PongGame
from agent import ActorCritic
import torch
import torch.optim as optim
import time
import pandas as pd


PLAYTIMES = 1000000
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 512
LR = 0.000005

SAVE_INTERVAL = 1  # how many episodes are run between each save (model and excel file)
MEAN_INTERVAL = 100  # how many episodes are run before the mean time is calculated

def build_return(values, rewards_list):
    returns = []  # save all the returns from the game
    delta = 0
    # value_next = 0
    # values_v = values[::-1]
    # print(f'values_v = {values_v}')
    for i, r in enumerate(rewards_list[::-1]):  # 从后往前遍历奖励
        delta = r + GAMMA * delta
        returns.append(delta)  # 添加回报到列表
        # delta = 0
    returns.reverse()  # 反转列表

    return returns


def save_to_excel(episode: int, data, headings: list[str], file_name=None):
    if file_name is None:
        file_name = f"data/episode{episode}.xlsx"
    df = pd.DataFrame(data, columns=headings)
    df.to_excel(file_name, index=False)


if __name__ == '__main__':



    env = PongGame()

    agent_right_1 = ActorCritic(state_size=32, action_size=3)
    agent_right_2 = ActorCritic(state_size=32, action_size=3)
    agent_left_1 = ActorCritic(state_size=32, action_size=3)
    agent_left_2 = ActorCritic(state_size=32, action_size=3)

    optimizer_right_1 = optim.Adam(agent_right_1.parameters(), lr=LR)
    optimizer_right_2 = optim.Adam(agent_right_2.parameters(), lr=LR)
    optimizer_left_1 = optim.Adam(agent_left_1.parameters(), lr=LR)
    optimizer_left_2 = optim.Adam(agent_left_2.parameters(), lr=LR)

    durations = []
    data = []

    headings = ['Episode', 'Collision_Agent1', 'Collision_Agent2', 'Collision_Agent3', 'Collision_Agent4', 'Loss_Agent1', 'Loss_Agent2', 'Loss_Agent3', 'Loss_Agent4', 'Duration'] 

    for episode in range(1, PLAYTIMES + 1): # 循环训练次数
        env.reset()  # 重置游戏状态
        state_get = env.build_state()
        states_list = []  # 存储一系列游戏状态

        action_right_1_list = []  # 存储动作
        action_right_2_list = []  # 存储动作
        rewards_right_1_list = []  # 存储奖励
        rewards_right_2_list = []  # 存储奖励
        values_right_1 = []  # 存储Critic分数
        values_right_2 = []  # 存储Critic分数

        action_left_1_list = []  # 存储动作
        action_left_2_list = []  # 存储动作
        rewards_left_1_list = []  # 存储奖励
        rewards_left_2_list = []  # 存储奖励
        values_left_1 = []  # 存储Critic分数
        values_left_2 = []  # 存储Critic分数

        agent_right_1.train()
        agent_right_2.train()
        agent_left_1.train()
        agent_left_2.train()

        start = time.time()

        while True: # 循环直到游戏结束
            state_input = torch.FloatTensor(state_get).unsqueeze(0)

            # action_right_1_policy, value_right_1 = agent_right_1.predict(state_input)

            action_right_1_policy, value_right_1 = agent_right_1(state_input)
            action_right_2_policy, value_right_2 = agent_right_2(state_input)
            action_left_1_policy, value_left_1 = agent_left_1(state_input)
            action_left_2_policy, value_left_2 = agent_left_2(state_input)

            # action
            action_right_1_choice = torch.multinomial(action_right_1_policy, 1).item()
            action_right_2_choice = torch.multinomial(action_right_2_policy, 1).item()
            action_left_1_choice = torch.multinomial(action_left_1_policy, 1).item()
            action_left_2_choice = torch.multinomial(action_left_2_policy, 1).item()
            right_action = [action_right_1_choice, action_right_2_choice]
            left_action = [action_left_1_choice, action_left_2_choice]

            # 根据动作更新游戏状态
            next_state_get, reward_right_1, reward_right_2, reward_left_1, reward_left_2, done = env.step(action_right=right_action, action_left=left_action)
            state_get = next_state_get  # 更新游戏状态
            next_state_input = torch.FloatTensor(next_state_get).unsqueeze(0)

            _, value_right_1_next = agent_right_1(state_input)
            _, value_right_2_next = agent_right_2(state_input)
            _, value_left_1_next = agent_left_1(state_input)
            _, value_left_2_next = agent_left_2(state_input)

            delta_right_1 = reward_right_1 + GAMMA * value_right_1_next.item() * (1 - done) - value_right_1.item()
            delta_right_2 = reward_right_2 + GAMMA * value_right_2_next.item() * (1 - done) - value_right_2.item()
            delta_left_1 = reward_left_1 + GAMMA * value_left_1_next.item() * (1 - done) - value_left_1.item()
            delta_left_2 = reward_left_2 + GAMMA * value_left_2_next.item() * (1 - done) - value_left_2.item()


            # 计算 loss
            actor_right_1_lost = -torch.log(action_right_1_policy[0, action_right_1_choice]) * delta_right_1
            actor_right_2_lost = -torch.log(action_right_2_policy[0, action_right_2_choice]) * delta_right_2
            actor_left_1_lost = -torch.log(action_left_1_policy[0, action_left_1_choice]) * delta_left_1
            actor_left_2_lost = -torch.log(action_left_2_policy[0, action_left_2_choice]) * delta_left_2

            critic_right_1_loss = delta_right_1 ** 2
            critic_right_2_loss = delta_right_2 ** 2
            critic_left_1_loss = delta_left_1 ** 2
            critic_left_2_loss = delta_left_2 ** 2

            loss_right_1 = actor_right_1_lost + critic_right_1_loss
            loss_right_2 = actor_right_2_lost + critic_right_2_loss
            
            loss_left_1 = actor_left_1_lost + critic_left_1_loss
            loss_left_2 = actor_left_2_lost + critic_left_2_loss


            optimizer_right_1.zero_grad()
            loss_right_1.backward()
            optimizer_right_1.step()

            optimizer_right_2.zero_grad()
            loss_right_2.backward()
            optimizer_right_2.step()

            optimizer_left_1.zero_grad()
            loss_left_1.backward()
            optimizer_left_1.step()

            optimizer_left_2.zero_grad()
            loss_left_2.backward()
            optimizer_left_2.step()


            if done: # 如果游戏结束
                break # 跳出循环


        end = time.time()
        duration = end - start
        temp_data = [
            episode,
            env.left_paddle1_count,
            env.left_paddle2_count,
            env.right_paddle1_count,
            env.right_paddle2_count,
            eval(f"{loss_left_1}"),
            eval(f"{loss_left_2}"),
            eval(f"{loss_right_1}"), 
            eval(f"{loss_right_2}"), 
            duration
        ]
        data.append(temp_data)
        durations.append(duration)
        print(f"It took {duration} seconds to run episode {episode}.")
        if len(durations) % MEAN_INTERVAL == 0:
            print(f"The mean time of last {MEAN_INTERVAL} episodes is {sum(durations) / len(durations)}")
            durations.clear()
        print(f'episode = {episode}\t{right_action = }')
        print(f'episode = {episode}\t{left_action = }')
        print(f'episode = {episode}\tloss_right_1 = {loss_right_1}')
        print(f'episode = {episode}\tloss_right_2 = {loss_right_2}')
        print(f'episode = {episode}\tloss_left_1 = {loss_left_1}')
        print(f'episode = {episode}\tloss_left_2 = {loss_left_2}')
        print(f'episode = {episode}\t{env.left_paddle1_count = }')
        print(f'episode = {episode}\t{env.left_paddle2_count = }')
        print(f'episode = {episode}\t{env.right_paddle1_count = }')
        print(f'episode = {episode}\t{env.right_paddle2_count = }')
        print(f'episode = {episode}\t{env.reward_left_1 = }')
        print(f'episode = {episode}\t{env.reward_left_2 = }')
        print(f'episode = {episode}\t{env.reward_right_1 = }')
        print(f'episode = {episode}\t{env.reward_right_2 = }')




        if episode % SAVE_INTERVAL == 0:
            # 保存模型的权重
            torch.save(agent_right_1.state_dict(), f'./model/right_1_weights_{episode}.pth')
            torch.save(agent_right_2.state_dict(), f'./model/right_2_weights_{episode}.pth')
            torch.save(agent_left_1.state_dict(), f'./model/left_1_weights_{episode}.pth')
            torch.save(agent_left_2.state_dict(), f'./model/left_2_weights_{episode}.pth')

            save_to_excel(episode, data, headings)
    



