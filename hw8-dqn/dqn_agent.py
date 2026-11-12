from agent import Agent
import numpy as np
import torch
import torch.nn.functional as F

from sample import Frame, SampleBatchNumpy

# DQN算法
class DQNAgent(Agent):

    def __init__(self, conf):
        # ε-贪心算法所需的参数
        self.action_dim = conf['action_dim']
        self.epsilon = conf.get('epsilon', 0.1)
        # 累计回报的衰减系数
        self.gamma = conf.get('gamma', 0.95)
        # 设备信息，如'cpu', 'cuda:0'等
        self.device = conf.get('device', 'cpu')
    
    # 设置推理和训练使用的模型
    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    # 输入单帧状态，采样探索性动作
    def predict(self, obs: 'Frame | np.ndarray | dict'):
        obs = Frame.convert(obs)
        if np.random.random() < self.epsilon:
            # 以 ε 的概率随机选择动作
            action = np.random.randint(self.action_dim)
        else:
            # 以 1-ε 的概率选择Q值最大的动作
            obs = obs.to_torch(device = self.device)
            with torch.no_grad():
                q_value = self.model.inference(obs)
            action = torch.argmax(q_value, dim = -1).item()
        return action
    
    # 输入单帧状态，计算最优动作
    def exploit(self, obs: 'Frame | np.ndarray | dict'):
        # 直接选择Q值最大的动作
        obs = Frame.convert(obs)
        obs = obs.to_torch(device = self.device)
        with torch.no_grad():
            q_value = self.model.inference(obs)
        action = torch.argmax(q_value, dim = -1).item()
        return action
    
    # 输入样本Batch，训练模型
    def learn(self, samples: SampleBatchNumpy):
        '''
        samples需要包含字段: obs, next_obs, action, reward, done
        '''
        samples = samples.to_torch(device = self.device)
        # 计算Q(s, a)
        q_values = self.model.inference(samples.obs)
        q_value = torch.gather(q_values, dim = 1, index = samples.action)
        # 计算Q(s', a)
        with torch.no_grad():
            next_q_values = self.model.inference(samples.next_obs)
            max_next_q_value = torch.max(next_q_values, dim = 1, keepdim = True)[0]
        # Q函数更新目标
        q_target = samples.reward + self.gamma * (1 - samples.done) * max_next_q_value
        loss = F.mse_loss(q_value, q_target)
        # 模型更新
        self.model.train(loss)
    
    def sample_process(self, samples: SampleBatchNumpy):
        '''
        samples需要包含字段: obs, next_obs, action, reward, done
        '''
        samples.action = samples.action.reshape((-1, 1)).astype(np.int64)
        samples.reward = samples.reward.reshape((-1, 1)).astype(np.float32)
        samples.done = samples.done.reshape((-1, 1)).astype(np.float32)
