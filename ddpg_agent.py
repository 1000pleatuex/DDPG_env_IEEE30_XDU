# 训练超参数配置
HIDDEN_DIM = 512  # 神经网络隐藏层维度
BUFFER_SIZE = 500000  # 经验回放缓冲区大小
BATCH_SIZE = 256  # 训练批次大小
GAMMA = 0.99  # 折扣因子
TAU = 0.001  # 目标网络软更新系数
ACTOR_LR = 2e-4  # Actor网络学习率
CRITIC_LR = 2e-3  # Critic网络学习率
NOISE_SCALE = 0.3  # 探索噪声标准差
OFFLINE_RATIO = 0.2  # 离线数据采样比例
USE_OFFLINE_DATA = True  # 是否使用离线数据

import numpy as np #用于科学计算的基础库，提供多维数组（ndarray）​操作和数学函数。
import torch #主流的深度学习框架，支持动态计算图和GPU加速。
import torch.nn as nn #PyTorch的神经网络模块，提供预定义的神经网络层（如全连接层、卷积层）、损失函数（如交叉熵、MSE）和模型容器。
import torch.optim as optim #提供优化算法（如SGD、Adam）用于更新模型参数。
import os #Python标准库的操作系统接口​，提供与操作系统交互的功能，如文件路径操作、环境变量读取等。
from datetime import datetime #获取或格式化当前时间，常用于日志记录或实验时间戳。
from buffer import ReplayBuffer, OfflineBuffer #ReplayBuffer:在强化学习中存储经验（状态、动作、奖励等），用于随机采样;OfflineBuffer:在离线数据处理场景中管理历史数据

# 检测是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class Actor(nn.Module):
    """Actor网络：决策网络，用于生成动作"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Actor, self).__init__()
        
        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),#这是一个全连接层（线性变换层，将输入维度从state_dim变换到hidden_dim，数学表达式：y = xW^T + b
            nn.ReLU(),#激活函数层，使用ReLU（Rectified Linear Unit0）激活函数：f(x) = max(0, x)，引入非线性，增加模型表达能力
            nn.LayerNorm(hidden_dim),#层归一化（Layer Normalization），对每个样本的特征维度进行归一化（均值为0，方差为1），有助于稳定训练过程，加速收敛
            nn.Dropout(0.1)#训练时随机"丢弃"（置零）10%的神经元输出，防止过拟合的正则化技术
        )
        #全连接层：1.​全局连接性​：前一层的所有神经元与当前层的每个神经元均通过权重连接，形成“全连接”结构。2.输入为前一层的全部输出（通常需展平为向量，如 CNN 中卷积层后接全连接层时需将特征图展平）。
        # 残差块1
        self.res_block1 = nn.ModuleList([
            nn.Sequential(#顺序容器（Sequential Container）​，用于将多个神经网络层按顺序组合成一个模块。它允许你快速堆叠多个层，数据会按照定义的顺序依次通过这些层，无需手动编写每一层的传递逻辑。
                nn.Linear(hidden_dim, hidden_dim),#全连接层，将输入维度从hidden_dim变换到hidden_dim（输入输出维度一致）
                nn.ReLU(),#ReLU激活函数
                nn.LayerNorm(hidden_dim),#层归一化，对每个样本的特征维度进行归一化。
                nn.Dropout(0.1),#以10%的概率随机将神经元输出置零。
                nn.Linear(hidden_dim, hidden_dim)#第二个全连接层，再次映射到hidden_dim维。与第一个线性层形成“瓶颈结构”。
            ) for _ in range(2)#创建两个完全相同的残差路径
        ])
        
        # 残差块2，同上
        self.res_block2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(2)
        ])
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),#全连接层，将输入维度从hidden_dim压缩到一半（hidden_dim//2），降维减少参数数量，防止直接映射到动作空间时丢失太多信息
            nn.ReLU(),#ReLU激活函数
            nn.LayerNorm(hidden_dim // 2),#对压缩后的128维特征进行层归一化
            nn.Linear(hidden_dim // 2, action_dim),#最终的全连接层，映射到动作空间维度
            nn.Sigmoid()  # Sigmoid激活函数，将输出压缩到(0,1)区间，确保所有输出值为正数
        )
        
        # 初始化权重
        self.apply(self._init_weights)#对当前模型的所有子模块应用_init_weights函数进行权重初始化
    
    def _init_weights(self, m):#定义 _init_weights 函数，参数 m 代表当前正在处理的神经网络模块
        if isinstance(m, nn.Linear):#检查当前模块 m 是否是 nn.Linear（全连接层）。如果是，则执行后续的初始化逻辑；否则跳过
            nn.init.orthogonal_(m.weight.data)  # 使用正交初始化（生成的权重矩阵满足 W*W^T=I）
            if m.bias is not None:#检查当前 nn.Linear 是否有偏置项 m.bias
                nn.init.constant_(m.bias.data, 0)#如果有偏置，则将其初始化为​全0​
    
    def forward(self, state):#定义了一个神经网络的前向传播（forward）方法
        x = self.input_layer(state)#self.input_layer定义见19行，负责将输入数据映射到隐藏空间。state 是输入数据
        
        # 残差块1
        for res_layer in self.res_block1:#遍历神经网络中的第一个残差块（self.res_block1，见27行）的所有子层，并对输入数据依次执行这些层的计算。
            res = res_layer(x)
            x = x + res#原始输入 x 直接与变换后的结果 res 相加
            x = torch.relu(x)#每次残差相加后应用 ReLU 激活函数，引入非线性。
        
        # 残差块2，同上
        for res_layer in self.res_block2:
            res = res_layer(x)
            x = x + res
            x = torch.relu(x)
        
        return self.output_layer(x)#将最后一个残差块的输出 x 传递给 self.output_layer（定义见49行）。

class Critic(nn.Module):
    """Critic网络：评价网络，用于评估状态-动作值"""
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(Critic, self).__init__()
        
        # 状态编码器，将状态输入映射到隐藏空间。结构：线性层 → ReLU激活 → 层归一化 → Dropout(0.1)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 动作编码器，将动作输入映射到隐藏空间
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1)
        )
        
        # 残差块，定义3个残差块组成的ModuleList。每个残差块包含：线性层 → ReLU → 层归一化 → Dropout → 线性层。输入输出维度都是hidden_dim * 2（因为状态和动作特征会拼接）
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim * 2),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
            ) for _ in range(3)
        ])
        
        # 输出层，将特征映射到最终的Q值输出，结构：逐步降维（512×2 → 512 → 256 → 1），每层后都有ReLU和层归一化
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重，使用正交初始化(orthogonal)初始化线性层权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if m.bias is not None:#偏置初始化为0
                nn.init.constant_(m.bias.data, 0)
    
    def forward(self, state, action):#前向传播
        # 编码状态和动作
        state_features = self.state_encoder(state)#（state_encoder见89行）
        action_features = self.action_encoder(action)#（action_encoder见97行）
        
        # 特征融合
        x = torch.cat([state_features, action_features], dim=1)#将状态特征和动作特征拼接（concatenate）在一起。
        
        # 残差块处理，通过多个残差块（Residual Block）进一步提取特征。
        for res_block in self.res_blocks:
            res = res_block(x)
            x = x + res
            x = torch.relu(x)#确保非线性。
        
        return self.output_layer(x)#将残差块处理后的特征映射到最终的Q值。

# ReplayBuffer类已移至buffer.py文件

class DDPGAgent:
    """DDPG智能体，支持在线和离线学习"""
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=HIDDEN_DIM,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        noise_scale=NOISE_SCALE,
        offline_ratio=OFFLINE_RATIO,
        use_offline_data=USE_OFFLINE_DATA
    ):
        # 初始化奖励统计信息
        self.rewards = []#初始化一个空列表，用于存储每个episode（阶段）的累计奖励。
        self.max_reward = float('-inf')#初始化最大奖励为负无穷，用于跟踪训练过程中获得的最高episode奖励。
        self.total_reward = 0.0#初始化总奖励为0.0，用于累加所有episode的奖励。
        self.num_episodes = 0#初始化episode计数器为0，用于记录已完成的episode数量。
        
        # 初始化网络并移动到指定设备
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)#创建主Actor网络（在线网络），将网络移动到指定设备（CPU或GPU）
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(device)#创建目标Actor网络，提供稳定的目标值计算，通过软更新(tau)缓慢跟踪在线网络
        self.actor_target.load_state_dict(self.actor.state_dict())#将在线网络的参数完全复制到目标网络，确保两个网络初始参数完全一致，是DDPG算法中目标网络初始化的标准做法
        
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)#同上
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)#Adam:一种优化算法，self.actor.parameters()：Actor网络的所有可训练参数，lr=actor_lr：学习率（见164行）
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)#创建在线经验回放池，存储与环境交互的转移数据
        
        # 离线数据缓冲区
        self.offline_buffer = None#预留离线数据集指针，需通过 load_offline_data() 方法加载外部数据
        self.use_offline_data = use_offline_data
        self.offline_ratio = offline_ratio
        
        # 超参数
        self.batch_size = batch_size
        self.gamma = gamma  # 折扣因子
        self.tau = tau      # 目标网络软更新系数
        self.noise_scale = noise_scale  # 探索噪声的标准差
        
        # 状态和动作维度
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(self, state, noise_scale=None):#确保输入状态为32位浮点数组
        """选择动作
        
        Args:
            state (numpy.ndarray): 环境状态，一维数组，维度为96
            noise_scale (float, optional): 探索噪声的标准差，如果为None则使用初始化时设置的值
            
        Returns:
            numpy.ndarray: 选择的动作，一维数组，维度为6
        """
        # 确保状态是numpy数组并转换为正确的数据类型
        state = np.array(state, dtype=np.float32)
        if state.ndim > 1:
            state = state.reshape(-1)  # 重塑为一维数组，兼容多维观测空间（DDPG通常用于低维连续状态）
        
        # torch.FloatTensor:转换为PyTorch张量。to(device):移动到GPU
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 通过Actor网络生成动作
        #torch.no_grad():禁用梯度计算，节省内存和计算资源。self.actor(state)：通过策略网络生成动作。.cpu()：将数据移回CPU（避免GPU内存泄漏）.squeeze(0)：去除batch维度。.numpy()：转换为numpy数组供环境使用
        with torch.no_grad():
            action = self.actor(state).cpu().squeeze(0).numpy()
        
        # 添加探索噪声，如果未指定noise_scale则使用默认值
        noise_std = noise_scale if noise_scale is not None else self.noise_scale
        action += np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action, -1, 1)#防止噪声导致动作越界
    
    def update(self):
        """更新网络参数，支持从在线和离线数据中学习"""
        # 检查在线缓冲区是否有足够的数据
        if len(self.replay_buffer) < self.batch_size:#确保有足够样本训练，​条件: buffer_size > batch_size
            return
        
        # 更新奖励统计信息
        _, _, rewards, _, _ = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))#从回放缓冲区（replay buffer）中采样一个批次（batch）的数据。这里只关心第三个返回值，即rewards（奖励），其他返回值用下划线忽略。
        episode_reward = rewards.mean().item()#计算采样得到的批次中所有奖励（rewards）的平均值，并转换为Python标量（.item()）
        self.rewards.append(episode_reward)#将上面计算得到的平均奖励episode_reward追加到self.rewards列表中
        self.max_reward = max(self.max_reward, episode_reward)#更新当前记录的最大奖励。将self.max_reward与当前批次的平均奖励比较，取较大者，以跟踪训练过程中出现的最大平均批次奖励。
        self.total_reward += episode_reward#累加当前批次的平均奖励到self.total_reward，用于后续计算平均奖励
        self.num_episodes += 1#统计已经处理了多少个批次（每个批次计算一次平均奖励，然后计数增加1）
        
        # 确定是否使用离线数据
        use_offline = self.use_offline_data and self.offline_buffer is not None and len(self.offline_buffer) > 0
        
        if use_offline:
            # 计算在线和离线的批次大小
            online_batch_size = int(self.batch_size * (1 - self.offline_ratio))
            offline_batch_size = self.batch_size - online_batch_size
            
            # 确保批次大小至少为1，避免空张量导致的运行时错误
            online_batch_size = max(1, online_batch_size)
            offline_batch_size = max(1, offline_batch_size)
            
            # 从在线缓冲区采样
            online_state, online_action, online_reward, online_next_state, online_done = \
                self.replay_buffer.sample(min(online_batch_size, len(self.replay_buffer)))#采样数量取online_batch_size和当前缓冲区大小的最小值，防止缓冲区未填满时采样超过实际存量
            
            try:
                # 从离线经验回放缓冲区中采样一批数据用于训练
                offline_state, offline_action, offline_reward, offline_next_state, offline_done = \
                    self.offline_buffer.sample(offline_batch_size)
                
                # 合并在线和离线数据
                state = torch.cat([online_state, offline_state])#将在线状态（online_state）和离线状态（offline_state）在第一个维度（通常是批处理维度）上拼接起来。
                action = torch.cat([online_action, offline_action])#同上，拼接在线动作和离线动作。
                reward = torch.cat([online_reward, offline_reward])#拼接在线奖励和离线奖励。
                next_state = torch.cat([online_next_state, offline_next_state])#拼接在线下一个状态和离线下一个状态。
                done = torch.cat([online_done, offline_done])#拼接在线终止标志和离线终止标志。
            except (ValueError, RuntimeError) as e:#捕获在拼接过程中可能出现的两种异常：ValueError（例如，当两个张量的维度不匹配时）和RuntimeError（例如，当两个张量不在同一个设备上，或者类型不匹配时）。
                print(f"离线数据采样失败: {e}，仅使用在线数据")
                state, action, reward, next_state, done = online_state, online_action, online_reward, online_next_state, online_done#在发生异常时，将整个批次的数据设置为在线数据
        else:
            # 仅从在线缓冲区采样
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # 更新Critic
        with torch.no_grad():#创建无梯度计算上下文,确保目标值不被反向传播
            next_action = self.actor_target(next_state)#使用目标Actor网络预测下一状态的最佳动作
            target_Q = self.critic_target(next_state, next_action)#使用目标Critic网络评估状态-动作对的Q值
            target_Q = reward + (1 - done) * self.gamma * target_Q#计算贝尔曼目标值(在给定状态下，执行某个动作后的期望回报)
        
        current_Q = self.critic(state, action)#使用主Critic网络预测当前状态-动作对的Q值
        critic_loss = nn.MSELoss()(current_Q, target_Q)#计算时序差分损失
        
        self.critic_optimizer.zero_grad()#清空Critic网络的梯度缓存,防止梯度累计导致错误更新,PyTorch默认会累加梯度，必须在每次更新前重置
        critic_loss.backward()#执行反向传播计算梯度
        self.critic_optimizer.step()#应用优化器更新网络权重
        
        # 更新Actor（类似292-294行）
        actor_loss = -self.critic(state, self.actor(state)).mean()#计算Actor的损失函数(​数学本质​：最大化期望回报)
        
        self.actor_optimizer.zero_grad()#清空Actor网络的梯度缓存
        actor_loss.backward()#反向传播计算梯度
        self.actor_optimizer.step()#执行参数更新
        
        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):#并行遍历主网络和目标网络的对应参数，要求两个网络结构完全相同
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)#直接在张量数据上操作（不涉及梯度计算）
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):#对actor网络做同样操作
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def get_state_dict(self):
        """获取智能体的网络状态字典
        
        Returns:
            dict: 包含actor和critic网络参数的状态字典
        """
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
    
    def save_experience(self, directory="checkpoints", filename=None):
        """保存经验回放缓冲区到文件
        
        Args:
            directory: 保存目录
            filename: 文件名，如果为None则使用时间戳生成
            
        Returns:
            str: 保存的文件路径
        """
        # 确保目录存在
        os.makedirs(directory, exist_ok=True)
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
            filename = f"experience_{timestamp}.buffer"
        
        # 构建完整路径
        filepath = os.path.join(directory, filename)
        
        # 保存经验回放缓冲区
        self.replay_buffer.save(filepath)
        
        return filepath
    
    def load_experience(self, filepath):
        """从文件加载经验回放缓冲区
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否成功加载
        """
        return self.replay_buffer.load(filepath)
    
    def set_offline_buffer(self, offline_buffer):
        """设置离线数据缓冲区
        
        Args:
            offline_buffer: OfflineBuffer对象
        """
        if isinstance(offline_buffer, OfflineBuffer):
            self.offline_buffer = offline_buffer
            self.use_offline_data = True
            print(f"已设置离线数据缓冲区，包含 {len(offline_buffer)} 条经验")
        else:
            raise TypeError("离线缓冲区必须是OfflineBuffer类型")
    
    def load_offline_data(self, directory="checkpoints", pattern="*.buffer"):
        """从目录加载所有离线数据
        
        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            
        Returns:
            int: 加载的缓冲区数量
        """
        # 创建离线缓冲区（如果尚未创建）
        if self.offline_buffer is None:
            self.offline_buffer = OfflineBuffer()
        
        # 加载数据
        count = self.offline_buffer.load_from_directory(directory, pattern)
        
        # 如果成功加载了数据，启用离线学习
        if count > 0:
            self.use_offline_data = True
        
        return count
    
    def load_state_dict(self, state_dict):
        """加载智能体的网络状态字典
        
        Args:
            state_dict (dict): 包含网络参数的状态字典
        """
        self.actor.load_state_dict(state_dict['actor'])#加载Actor网络参数（后面的一样）
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
    
    def get_reward_stats(self):
        """获取奖励统计信息
        
        Returns:
            dict: 包含平均奖励值和最大奖励值的字典
        """
        if self.num_episodes == 0:
            return {
                'mean_reward': 0.0,
                'max_reward': self.max_reward
            }
        
        return {
            'mean_reward': self.total_reward / self.num_episodes,
            'max_reward': self.max_reward
        }