import numpy as np
import torch
import os
import pickle

# 检测是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """经验回放缓冲区，支持数据持久化"""
    def __init__(self, capacity):#初始化经验回放缓冲区
        self.capacity = capacity#指定缓冲区的最大容量
        self.buffer = []#创建空列表作为经验存储容器
        self.position = 0#当前位置指针，用于控制循环覆盖
    
    def push(self, state, action, reward, next_state, done):#将新的经验添加到缓冲区中，并在缓冲区满后循环覆盖旧经验
        """添加一条经验到缓冲区"""
        if len(self.buffer) < self.capacity:#动态扩展缓冲区直至达到预设容量
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)#直接覆盖当前位置的经验
        self.position = (self.position + 1) % self.capacity#当位置指针达到容量时自动归零
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size)#从缓冲区中随机采样一批经验
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in batch])#经验提取与解包
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )#张量转换与预处理
    
    def save(self, filepath):
        """将缓冲区数据保存到文件
        
        Args:
            filepath: 保存文件的路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存数据
        with open(filepath, 'wb') as f:
            pickle.dump(self.buffer, f)
        print(f"经验回放缓冲区已保存到 {filepath}，共 {len(self.buffer)} 条经验")
    
    def load(self, filepath):
        """从文件加载缓冲区数据
        
        Args:
            filepath: 加载文件的路径
            
        Returns:
            bool: 是否成功加载数据
        """
        if not os.path.exists(filepath):
            print(f"文件 {filepath} 不存在，无法加载经验回放缓冲区")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity
            print(f"已从 {filepath} 加载 {len(self.buffer)} 条经验到回放缓冲区")
            return True
        except Exception as e:
            print(f"加载经验回放缓冲区失败: {e}")
            return False
    
    def __len__(self):
        return len(self.buffer)


class OfflineBuffer:
    """离线数据缓冲区，用于管理多个历史数据集"""
    def __init__(self, capacity_per_buffer=500000):#定义了一个多缓冲区管理系统的初始化方法
        self.buffers = []
        self.capacity_per_buffer = capacity_per_buffer
    
    def add_buffer(self, buffer):
        """添加一个回放缓冲区到离线数据集合
        
        Args:
            buffer: ReplayBuffer对象
        """
        if isinstance(buffer, ReplayBuffer):
            self.buffers.append(buffer)
            print(f"已添加包含 {len(buffer)} 条经验的缓冲区到离线数据集合")
        else:
            raise TypeError("只能添加ReplayBuffer类型的对象")
    
    def load_from_file(self, filepath):
        """从文件加载一个回放缓冲区并添加到离线数据集合
        
        Args:
            filepath: 加载文件的路径
            
        Returns:
            bool: 是否成功加载数据
        """
        buffer = ReplayBuffer(self.capacity_per_buffer)
        if buffer.load(filepath):
            self.add_buffer(buffer)#加载成功则添加到缓冲区集合，加载失败返回False
            return True
        return False
    
    def load_from_directory(self, directory, pattern="*.buffer"):
        """从目录加载所有符合模式的回放缓冲区文件
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            int: 成功加载的缓冲区数量
        """
        import glob
        
        if not os.path.exists(directory):
            print(f"目录 {directory} 不存在")
            return 0
        
        files = glob.glob(os.path.join(directory, pattern))
        count = 0
        
        for file in files:
            if self.load_from_file(file):
                count += 1
        
        print(f"从目录 {directory} 加载了 {count} 个缓冲区")
        return count
    
    def sample(self, batch_size):
        """从所有离线缓冲区中采样经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            tuple: (state, action, reward, next_state, done) 元组
            
        Raises:
            ValueError: 如果没有可用的缓冲区或缓冲区为空
        """
        if not self.buffers:
            raise ValueError("没有可用的离线缓冲区")
        
        # 计算每个缓冲区的权重（基于其大小）
        buffer_sizes = [len(buffer) for buffer in self.buffers]
        total_size = sum(buffer_sizes)
        
        if total_size == 0:
            raise ValueError("所有离线缓冲区都为空")
        
        # 按比例从每个缓冲区采样
        buffer_weights = [size / total_size for size in buffer_sizes]
        buffer_samples = [int(weight * batch_size) for weight in buffer_weights]
        
        # 确保总样本数等于batch_size
        remaining = batch_size - sum(buffer_samples)
        if remaining > 0:
            # 将剩余的样本分配给最大的缓冲区
            max_idx = buffer_sizes.index(max(buffer_sizes))
            buffer_samples[max_idx] += remaining
        
        # 从每个缓冲区采样并合并结果
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i, buffer in enumerate(self.buffers):
            if buffer_samples[i] > 0 and len(buffer) > 0:
                s, a, r, ns, d = buffer.sample(min(buffer_samples[i], len(buffer)))
                states.append(s)
                actions.append(a)
                rewards.append(r)
                next_states.append(ns)
                dones.append(d)
        
        # 合并所有采样
        if not states:  # 如果没有成功采样任何数据
            raise ValueError("无法从离线缓冲区采样足够的数据")
        
        return (
            torch.cat(states),
            torch.cat(actions),
            torch.cat(rewards),
            torch.cat(next_states),
            torch.cat(dones)
        )
    
    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)