
import numpy as np
from environment import PongGame
from agent import ActorCritic
import torch
import torch.optim as optim

from main import LR


PLAYTIMES = 50
EPISODES = 10
GAMMA = 0.99  # 折扣因子
BATCH_SIZE = 32





if __name__ == '__main__':

    env = PongGame()
    agent_right_1 = ActorCritic(state_size=32, action_size=3)
    agent_right_2 = ActorCritic(state_size=32, action_size=3)
    agent_left_1 = ActorCritic(state_size=32, action_size=3)
    agent_left_2 = ActorCritic(state_size=32, action_size=3)

    # agent_right_1.load_state_dict(torch.load('./model/right_1_weights_6000.pth'))
    # agent_right_2.load_state_dict(torch.load('./model/right_2_weights_6000.pth'))
    # agent_left_1.load_state_dict(torch.load('./model/left_1_weights_6000.pth'))
    # agent_left_2.load_state_dict(torch.load('./model/left_2_weights_6000.pth'))


    
    agent_left_1.load_state_dict(torch.load('./model/left_1_weights_7700.pth'))
    agent_left_2.load_state_dict(torch.load('./model/left_2_weights_7700.pth'))
    agent_right_1.load_state_dict(torch.load('./model/right_1_weights_7700.pth'))
    agent_right_2.load_state_dict(torch.load('./model/right_2_weights_7700.pth'))
    
    
    for episode in range(PLAYTIMES):  # 循环训练次数
        env.reset()  # 重置游戏状态
        state_get = env.build_state()

        while True:  # 循环直到游戏结束
            state_input = torch.FloatTensor(state_get).unsqueeze(0)
            # print(state_input.shape)
            # asd
            # 获取决策
            # right
            action_right_1_pred, value_right_1_pred = agent_right_1.predict(state_input)  # 预测动作的概率
            action_right_2_pred, value_right_2_pred = agent_right_2.predict(state_input)  # 预测动作的概率

            action_right_1_probs = action_right_1_pred[0]
            action_right_2_probs = action_right_2_pred[0]
            action_right_1_probs /= action_right_1_probs.sum()
            action_right_2_probs /= action_right_2_probs.sum()

            value_right_1_pred = value_right_1_pred[0][0]
            value_right_2_pred = value_right_2_pred[0][0]

            # action_right_1_choice = np.random.choice([0, 1, 2], p=action_right_1_probs)  # 根据概率随机选择一个动作
            # action_right_2_choice = np.random.choice([0, 1, 2], p=action_right_2_probs)  # 根据概率随机选择一个动作
            action_right_1_probs_list = action_right_1_probs.tolist()
            action_right_2_probs_list = action_right_2_probs.tolist()
            action_right_1_choice = [0, 1, 2][action_right_1_probs_list.index(max(action_right_1_probs_list))]
            action_right_2_choice = [0, 1, 2][action_right_2_probs_list.index(max(action_right_2_probs_list))]



            right_action = [action_right_1_choice, action_right_2_choice]
            print(f'right_action = {right_action}')

            # left
            action_left_1_pred, value_left_1_pred = agent_left_1.predict(state_input)  # 预测动作的概率
            action_left_2_pred, value_left_2_pred = agent_left_2.predict(state_input)  # 预测动作的概率

            action_left_1_probs = action_left_1_pred[0]
            action_left_2_probs = action_left_2_pred[0]
            action_left_1_probs /= action_left_1_probs.sum()
            action_left_2_probs /= action_left_2_probs.sum()

            value_left_1_pred = value_left_1_pred[0][0]
            value_left_2_pred = value_left_2_pred[0][0]

            # action_left_1_choice = np.random.choice([0, 1, 2], p=action_left_1_probs)  # 根据概率随机选择一个动作
            # action_left_2_choice = np.random.choice([0, 1, 2], p=action_left_2_probs)  # 根据概率随机选择一个动作

            action_left_1_probs_list = action_left_1_probs.tolist()
            action_left_2_probs_list = action_left_2_probs.tolist()
            action_left_1_choice = [0, 1, 2][action_left_1_probs_list.index(max(action_left_1_probs_list))]
            action_left_2_choice = [0, 1, 2][action_left_2_probs_list.index(max(action_left_2_probs_list))]


            left_action = [action_left_1_choice, action_left_2_choice]
            print(f'left_action = {left_action}')

            # 根据动作更新游戏状态
            next_state_get, reward_right_1, reward_right_2, reward_left_1, reward_left_1, done = env.step(action_right=right_action,
                                                                       action_left=left_action)
            next_state_input = torch.FloatTensor(next_state_get).unsqueeze(0)
            state_get = next_state_get  # 更新游戏状态

            if done:  # 如果游戏结束
                break  # 跳出循环


        info = env.get_info()
        print(info)
        

