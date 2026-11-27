from agent import Agent
from copy import deepcopy
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
        self.target_update_freq = conf.get('target_update_freq', 0)
        self.double_dqn = conf.get('double_dqn', False)
        self.huber_loss = conf.get('huber_loss', False)
        self.max_grad_norm = conf.get('max_grad_norm', 0.0)
        self.learn_steps = 0
        if self.double_dqn and self.target_update_freq <= 0:
            raise ValueError('Double DQN requires a target network')
    
    # 设置推理和训练使用的模型
    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.learn_steps = 0
        self.target_model = None
        if self.target_update_freq:
            self.target_model = deepcopy(self.model)
            self.target_model.optimizer = None
            self.target_model.requires_grad_(False)
            self.target_model.eval()

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
        多步样本的 reward 为累计折扣奖励，discount 为 gamma ** 实际步数。
        '''
        samples = samples.to_torch(device = self.device)
        # 计算Q(s, a)
        q_values = self.model.inference(samples.obs)
        q_value = torch.gather(q_values, dim = 1, index = samples.action)
        # 计算Q(s', a)
        with torch.no_grad():
            target_model = self.target_model if self.target_model is not None else self.model
            next_q_values = target_model.inference(samples.next_obs)
            if self.double_dqn:
                next_actions = self.model.inference(samples.next_obs).argmax(dim=1, keepdim=True)
                max_next_q_value = next_q_values.gather(1, next_actions)
            else:
                max_next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
        # Q函数更新目标
        # Aggregated replay samples carry gamma ** actual_horizon. Legacy
        # batches without this field retain the original one-step target.
        discount = samples.discount if 'discount' in samples.keys() else self.gamma
        q_target = samples.reward + discount * (1 - samples.done) * max_next_q_value
        loss = F.smooth_l1_loss(q_value, q_target) if self.huber_loss else F.mse_loss(q_value, q_target)
        # 模型更新
        grad_norm = self.model.optimize(loss, self.max_grad_norm)
        self.learn_steps += 1
        if self.target_model is not None and self.learn_steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return dict(loss=loss.item(), q_mean=q_value.detach().mean().item(),
                    target_mean=q_target.mean().item(), grad_norm=grad_norm)
    
    def sample_process(self, samples: SampleBatchNumpy):
        '''
        samples需要包含字段: obs, next_obs, action, reward, done
        '''
        samples.action = samples.action.reshape((-1, 1)).astype(np.int64)
        samples.reward = samples.reward.reshape((-1, 1)).astype(np.float32)
        samples.done = samples.done.reshape((-1, 1)).astype(np.float32)
        if 'discount' in samples.keys():
            samples.discount = samples.discount.reshape((-1, 1)).astype(np.float32)
