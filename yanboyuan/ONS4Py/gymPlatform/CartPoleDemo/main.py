import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable as V
import sys


'''
A2C算法实现。相关参考可见：https://github.com/openai/baselines/tree/master/baselines/a2c
原论文可见：https://arxiv.org/abs/1602.01783
'''

''' hyper-parameter definition '''
STATE_DIM = 4  # observation维度
ACTION_DIM = 2  # action space是Discrete(2)
STEP = 200000
SAMPLE_NUMS = 200  # 每一轮游戏，运行多少个sample
GAMMA = 0.99
MAX_NORM = 0.5

TEST_PERIOD = 50  # 50轮游戏测试一次。


class ActorNetwork(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))  # log(softmax(x,y))
        return out


class ValueNetwork(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def roll_out(actor_net: nn.Module, task: gym.wrappers.time_limit.TimeLimit,
             sample_nums: int, value_net: nn.Module, init_state: np.ndarray):
    """
    游戏不结束的情况下，最多运行sample_nums帧画面就结束游戏。只玩，不训练。
    :param actor_net: 行为网络，用于评估特定状态下采取各种行为的概率
    :param task: 游戏任务
    :param sample_nums: 一轮游戏最大帧数
    :param value_net: 价值网络，用于评估最终结束的时候的状态
    :param init_state: 初始状态
    :return:
    """
    # 构建存储状态，行为，行为奖励，游戏是否结束和最终奖励的变量
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0

    state = init_state

    for j in range(sample_nums):
        states.append(state)  # 存储当前状态
        log_softmax_action = actor_net(V(torch.Tensor([state])))  # 用行为网络评估出当前状态下的各种行为的概率
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM, p=softmax_action.data.numpy()[0])  # 根据行为的softmax概率选择行为
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]  # 根据选择的行为，生成one_hot列表。

        next_state, reward, done, _ = task.step(action)  # 采取行为，奖励是gym默认的。

        # 存储采取行为之后的过程与结果
        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:  # 如果采取行为后，游戏结束，则重启游戏并跳出for循环
            is_done = True
            state = task.reset()
            break

    if not is_done:
        final_r = value_net(V(torch.Tensor([final_state]))).data.numpy()  # 用价值网络评估最后的网络状态的好坏程度。

    return states, actions, rewards, final_r, state


def discount_reward(r, gamma, final_r):
    """
    计算经过折扣后的reward。其中第i步的reward_i = r[i] + final_r * np.exp(gamma, n-i)。
    即从后往前，第i步的reward除了该步得到的直接反馈以外，还需要计算其对最后一步的影响大小。
    :param r: 采取行为网络计算出来的下一步行为得到的反馈。
    :param gamma: 折扣因子
    :param final_r: 用价值网络评估的某个状态的得分
    :return:
    """
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    cart_env = gym.make('CartPole-v0')  # 构造环境

    init_state = cart_env.reset()  # 刷新环境

    # 初始化价值网络
    value_net = ValueNetwork(input_size=STATE_DIM, hidden_size=40, output_size=1)
    value_optim = torch.optim.Adam(value_net.parameters(), lr=0.01)

    # 初始化行为网络
    actor_net = ActorNetwork(input_size=STATE_DIM, hidden_size=40, action_size=ACTION_DIM)
    actor_optim = torch.optim.Adam(actor_net.parameters(), lr=0.01)

    for step in range(STEP):
        states, actions, rewards, final_r, current_state = roll_out(actor_net, cart_env,
                                                                    SAMPLE_NUMS, value_net, init_state)
        init_state = current_state

        #
        actions_var = V(torch.Tensor(actions).view(-1, ACTION_DIM))
        states_var = V(torch.Tensor(states).view(-1, STATE_DIM))

        '''训练行为网络'''
        actor_optim.zero_grad()
        log_softmax_actions = actor_net(states_var)
        vs = value_net(states_var).detach()  # detach 将变量从图中分离出来，该变量将不支持梯度。但是与原变量共享内存。
        # 计算Qs
        qs = V(torch.Tensor(discount_reward(rewards, GAMMA, final_r)))

        advantages = qs - vs
        # 将被采取的行为对应的行为网络计算出来的概率压缩成一维数据，然后乘以Qs，取均值
        actor_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var, 1) * advantages)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(actor_net.parameters(), MAX_NORM)  # 梯度标准化，最大标准值为0.5 TODO
        actor_optim.step()

        '''训练价值网络'''
        value_optim.zero_grad()
        target_values = qs
        values = value_net(states_var)
        criterion = nn.MSELoss()  # MSELoss是啥子

        value_loss = criterion(values, target_values)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(value_net.parameters(), MAX_NORM)
        value_optim.step()

        '''测试'''
        if (step+1) % 5 is 0:
            result = 0
            test_task = gym.make("CartPole-v0")
            for test_epi in range(100):
                state = test_task.reset()
                for test_step in range(200):
                    softmax_action = torch.exp(actor_net(V(torch.Tensor([state]))))
                    action = np.argmax(softmax_action.data.numpy()[0])
                    next_state, reward, done, _ = test_task.step(action)
                    result += reward
                    state = next_state
                    if done:
                        break
            result = result / 100.0
            print("step:", step + 1, "test result:", result)
            if result >= 195:
                sys.exit(0)


if __name__ == "__main__":
    main()
