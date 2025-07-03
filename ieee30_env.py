import numpy as np # 导入numpy库，用于数学计算
import gym # 导入gym库，用于强化学习环境
from gym import spaces # 导入gym库中的spaces模块，用于定义动作和状态空间
from pypower.api import case30, runpf # 导入pypower库中的runpf函数，用于执行潮流计算
import matplotlib.pyplot as plt # 导入matplotlib库，用于绘图
from datetime import datetime # 导入datetime库，用于获取当前时间
import os # 导入os库，用于处理文件路径

class IEEE30Env(gym.Env):
    """IEEE 30节点电力系统环境
    
    使用pypower实现的IEEE 30节点电力系统仿真环境，用于强化学习训练。
    状态空间包含节点电压、相角、发电机出力和负荷功率等信息。
    动作空间为6台发电机的有功出力调整。
    """
    
    def __init__(self):
        super(IEEE30Env, self).__init__()
        
        # 加载IEEE 30节点案例
        self.case = case30()
        
        # 系统参数
        self.n_gen = 6   # 发电机数量
        self.n_bus = 30  # 节点数量
        self.n_branch = 41  # 支路数量
        
        # 定义动作空间（6个发电机的有功出力）
        # 根据case30.py中的发电机限制设置
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),  # 发电机最小出力
            high=np.array([80, 80, 50, 55, 30, 40]),  # 发电机最大出力
            dtype=np.float32
        )
        
        # 定义状态空间
        # 状态包含: 节点电压幅值(30) + 节点相角(30) + 发电机有功出力(6) + 节点有功负荷(30)
        # 根据case30.py中的数据设置限制
        self.observation_space = spaces.Box(
            low=np.array([0.95] * self.n_bus +  # 最小电压
                        [-np.pi] * self.n_bus +  # 最小相角
                        [0] * self.n_gen +  # 最小发电机出力
                        [0] * self.n_bus),  # 最小负荷功率
            high=np.array([1.1] * self.n_bus +  # 最大电压
                         [np.pi] * self.n_bus +  # 最大相角
                         [100] * self.n_gen +  # 最大发电机出力
                         [100] * self.n_bus),  # 最大负荷功率
            dtype=np.float32
        )
        
        # 惩罚系数参数
        self.voltage_k = 0.5  # 电压惩罚系数
        self.voltage_alpha = 3.0  # 电压惩罚指数
        self.branch_k = 0.5   # 支路惩罚基础系数（增加惩罚强度）
        
        # 奖励权重参数
        self.gen_balance_k = 0.3  # 发电机出力均衡奖励系数
        self.load_balance_k = 0.3  # 支路负载均衡奖励系数
        self.loss_penalty_k = 0.4  # 系统损耗惩罚系数
        self.voltage_quality_k = 0.4  # 电压质量奖励系数
        
        # 初始化监控数据存储
        self.line_loading_history = []  # 存储线路负载率历史数据
        self.line_loss_history = []     # 存储线路损耗历史数据
        
        # 设置关键支路
        self.critical_branches = [0, 1, 5, 8]  # 关键线路索引
        
        # 支路重要性权重（可根据实际情况调整）
        self.branch_importance = np.ones(self.n_branch)  # 默认权重为1
        self.branch_importance[self.critical_branches] = 2.0  # 关键支路权重加倍
        
        # 初始化状态
        self.state = None
        
    def step(self, action):
        """执行一步仿真
        
        Args:
            action: numpy数组，包含6个发电机的有功出力
            
        Returns:
            observation: 新的系统状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 更新发电机出力
        self.case['gen'][:, 1] = action  # 设置有功出力
        
        # 执行潮流计算
        results, success = runpf(self.case) 
        
        if not success:
            return self.state, -1000,True, {'success': False}    # 潮流计算失败，返回惩罚（-1000）
        
        # 获取新状态
        self.state = self._get_state(results)
        
        # 计算奖励
        reward = self._calculate_reward(results, action)
        
        # 检查是否需要终止
        done = self._check_termination(results)
        
        # 计算线路负载率和损耗
        line_loading = np.abs(results['branch'][:, 13]) / 100  # 线路负载率（标幺值），
        line_loss = results['branch'][:, 15]  # 有功损耗（MW）
        
        # 更新历史数据
        self.line_loading_history.append(line_loading)  # 存储线路负载率历史数据
        self.line_loss_history.append(line_loss) # 存储线路损耗历史数据
        
        return self.state, reward, done, {
            'success': True,
            'line_loading': line_loading,
            'line_loss': line_loss
        }
    
    def reset(self, seed=None):
        """重置环境状态
        
        Returns:
            observation: 初始状态
        """
        super().reset(seed=seed)
        
        # 重新加载案例
        self.case = case30()
        
        # 重置发电机出力为初始值
        self.case['gen'][:, 1] = np.array([23.54, 60.97, 21.59, 26.91, 19.2, 37.0]) 
        
        # 执行初始潮流计算
        results, _ = runpf(self.case)
        
        # 获取初始状态
        self.state = self._get_state(results)
        
        # 仅返回状态值，不返回额外信息
        return self.state
    
    def _get_state(self, ppc):
        """获取环境状态
        
        Returns:
            numpy.ndarray: 包含系统状态信息的一维数组，维度为96
                - 节点电压幅值(30)
                - 节点相角(30)
                - 发电机有功出力(6)
                - 节点有功负荷(30)
        """
        # 提取状态信息并确保是一维数组
        vm = np.array(ppc['bus'][:, 7], dtype=np.float32)  # 节点电压幅值 (30,)
        va = np.array(ppc['bus'][:, 8], dtype=np.float32) * np.pi / 180  # 节点相角（转换为弧度）(30,)
        pg = np.array(ppc['gen'][:, 1], dtype=np.float32)  # 发电机有功出力 (6,)
        pd = np.array(ppc['bus'][:, 2], dtype=np.float32)  # 节点有功负荷 (30,)
        
        # 组合状态向量
        state = np.concatenate([vm.reshape(-1), va.reshape(-1), pg.reshape(-1), pd.reshape(-1)])
        return state
    

        
    def get_line_capacity(self):
        """获取所有线路的容量信息
        
        Returns:
            numpy.ndarray: 包含所有线路容量的数组 (MW)
        """
        return np.array(self.case['branch'][:, 5])  # branch矩阵第6列（索引5）为线路容量
        
    def plot_line_loading(self, ax=None):
        """绘制线路负载率分布
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新图形
        """
        if not self.line_loading_history:
            print("没有可用的线路负载率数据")
            return
            
        # 获取最新的线路负载率数据和容量信息
        line_loading_ratios = self.line_loading_history[-1] 
        line_capacities = self.get_line_capacity() 
        
        # 如果没有提供ax，创建新的图形
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
        
        # 创建双轴图
        ax2 = ax.twinx()
        
        # 绘制负载率（左轴）
        bars = ax.bar(range(self.n_branch), line_loading_ratios,
                     color=[('red' if ratio > 0.9 else 'blue') for ratio in line_loading_ratios])
        ax.axhline(0.9, color='red', linestyle='--', label='90% 负载阈值')
        ax.set_title('线路负载率和容量分布')
        ax.set_xlabel('线路编号')
        ax.set_ylabel('负载率 (p.u.)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 绘制容量（右轴）
        line = ax2.plot(range(self.n_branch), line_capacities, 'g--', label='线路容量')
        ax2.set_ylabel('容量 (MW)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 合并图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        
        ax.grid(True)
        
        # 获取当前时间并格式化
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
        
        # 构建保存路径
        save_dir = os.path.join(os.path.dirname(__file__), 'result')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建保存目录: {save_dir}")
        save_path = os.path.join(save_dir, f'{timestamp}_line.png')
        
        # 保存图像并添加错误处理
        try:
            plt.savefig(save_path)
            print(f"成功保存线路负载率图像到: {save_path}")
        except Exception as e:
            print(f"保存线路负载率图像时出错: {e}")
        finally:
            plt.show()
        
    def plot_line_losses(self, ax=None):
        """绘制线路损耗分布
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新图形
        """
        if not self.line_loss_history:
            print("没有可用的线路损耗数据")
            return
            
        # 获取最新的线路损耗数据
        line_losses = self.line_loss_history[-1]
        
        # 如果没有提供ax，创建新的图形
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
        
        ax.bar(range(self.n_branch), line_losses,
               color=[('red' if i in self.critical_branches else 'blue') for i in range(self.n_branch)])
        ax.set_title('线路损耗分布')
        ax.set_xlabel('线路编号')
        ax.set_ylabel('损耗 (MW)')
        ax.grid(True)
        
        # 定义动作空间和状态空间（保持不变）
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([80, 80, 50, 55, 30, 40]),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=np.array([0.95] * self.n_bus +
                        [-np.pi] * self.n_bus +
                        [0] * self.n_gen +
                        [0] * self.n_bus),
            high=np.array([1.1] * self.n_bus +
                         [np.pi] * self.n_bus +
                         [100] * self.n_gen +
                         [100] * self.n_bus),
            dtype=np.float32
        )
        
        # 初始化状态
        self.state = None

    def _calculate_reward(self, ppc, action):
        """计算奖励值
        
        基于系统运行状态计算奖励，考虑多个方面：
        1. 基础约束：发电机负出力惩罚、发电成本
        2. 电压质量：电压偏差惩罚和优质电压奖励
        3. 发电均衡：发电机出力均衡性奖励
        4. 网络状态：支路负载均衡奖励、过载惩罚和损耗惩罚
        """
        reward = 0
        
        # 1. 基础约束
        # 1.1 发电机负出力惩罚
        negative_output_penalty = 0 # 初始化负出力惩罚
        for pg in action:
            if pg < 0:
                negative_output_penalty += -100000 * abs(pg) # 负出力惩罚系数100000*有功出力
        reward += negative_output_penalty # 累加负出力惩罚
        
        # 1.2 发电成本
        gen_cost = 0
        for i, pg in enumerate(action):
            c2, c1, c0 = self.case['gencost'][i, 4:7] # 获取成本参数c2, c1, c0
            gen_cost += c2 * pg * pg + c1 * pg + c0 # 计算成本=c2*有功出力^2 + c1*有功出力 + c0
        reward -= gen_cost
        
        # 2. 电压质量
        v = ppc['bus'][:, 7]
        # 2.1 电压越限惩罚
        voltage_dev = np.maximum(0, np.abs(v - 1.0) - 0.05)  # 电压越限程度
        v_penalty = -self.voltage_k * np.sum(np.exp(self.voltage_alpha * voltage_dev))  # 电压越限惩罚=电压惩罚系数*电压越限程度的指数函数
        reward += v_penalty
        
        # 2.2 电压质量奖励（电压越接近1.0越好）
        voltage_quality = np.exp(-10 * np.abs(v - 1.0))  # 使用指数函数奖励接近1.0的电压
        voltage_quality_reward = self.voltage_quality_k * np.mean(voltage_quality)
        reward += voltage_quality_reward
        
        # 3. 发电均衡
        gen_capacity = self.action_space.high 
        gen_output_ratios = action / gen_capacity  # 计算出力比例
        
        # 3.1 发电机出力均衡奖励（基于出力比例的标准差）
        output_std = np.std(gen_output_ratios)  # 计算出力比例标准差
        gen_balance_reward = self.gen_balance_k * np.exp(-5 * output_std)  # 出力越均衡，标准差越小，奖励越大
        reward += gen_balance_reward  # 累加出力均衡奖励
        
        # 3.2 轻载惩罚
        light_load_threshold = 0.2
        for ratio in gen_output_ratios:
            if ratio < light_load_threshold:  # 若出力比例小于阈值（0.2*最大出力）
                reward += -100000 * (light_load_threshold - ratio)  # 惩罚系数100000*（阈值-出力比例）
        
        # 4. 网络状态
        s_from = np.abs(ppc['branch'][:, 13])  # 支路起始端视在功率
        s_to = np.abs(ppc['branch'][:, 15])    # 支路终止端视在功率
        s_max = ppc['branch'][:, 5]            # 支路容量限制
        
        # 4.1 支路负载率
        line_loading_ratios = np.maximum(s_from / s_max, s_to / s_max)  # 计算线路负载率=max(起始端视在功率/支路容量限制, 终止端视在功率/支路容量限制)
        self.line_loading_history.append(line_loading_ratios)  # 存储线路负载率历史数据
        
        # 4.2 支路负载均衡奖励
        load_std = np.std(line_loading_ratios) 
        load_balance_reward = self.load_balance_k * np.exp(-5 * load_std)  # 负载越均衡，标准差越小，奖励越大，=负载均衡奖励系数*负载标准差的指数函数
        reward += load_balance_reward 
        
        # 4.3 过载惩罚
        overload_from = np.maximum(0, s_from / s_max - 1.0)  # 计算支路起始端过载程度=max(起始端视在功率/支路容量限制-1, 0)
        overload_to = np.maximum(0, s_to / s_max - 1.0)  # 计算支路终止端过载程度=max(终止端视在功率/支路容量限制-1, 0)
        branch_penalty_from = -self.branch_k * self.branch_importance * (overload_from ** 3)  # 支路起始端过载惩罚=支路惩罚系数*支路重要性*起始端过载程度的立方
        branch_penalty_to = -self.branch_k * self.branch_importance * (overload_to ** 3)  # 支路终止端过载惩罚=支路惩罚系数*支路重要性*终止端过载程度的立方
        branch_penalty = np.sum(branch_penalty_from) + np.sum(branch_penalty_to)  # 累加支路起始端和终止端过载惩罚
        reward += branch_penalty  # 累加过载惩罚
        
        # 4.4 系统损耗惩罚
        i_sqr = (s_from ** 2 + s_to ** 2) / (ppc['branch'][:, 5] ** 2)  # 计算支路的功率损耗率=(起始端视在功率^2 + 终止端视在功率^2) / 支路容量限制^2
        r = ppc['branch'][:, 2]  # 支路电阻
        line_losses = i_sqr * r  # 计算支路损耗=功率损耗率 * 支路电阻
        self.line_loss_history.append(line_losses)  # 存储线路损耗历史数据
        total_loss = np.sum(line_losses)  # 计算总损耗=支路损耗的总和
        loss_penalty = -self.loss_penalty_k * total_loss  # 损耗越大，惩罚越大 = 损耗惩罚系数*总损耗
        reward += loss_penalty
        
        return reward
    
    def _check_termination(self, ppc):
        """检查是否需要终止仿真"""
        # 检查电压是否严重越限
        v = ppc['bus'][:, 7]
        if np.any(v < 0.9) or np.any(v > 1.1):
            return True
        
        # 检查支路是否严重过载
        s_from = np.abs(ppc['branch'][:, 13]) 
        s_to = np.abs(ppc['branch'][:, 15]) 
        s_max = ppc['branch'][:, 5] 
        if np.any(s_from > 1.2 * s_max) or np.any(s_to > 1.2 * s_max):
            return True
        
        return False
        
    def plot_line_loading(self, ax=None):
        """绘制线路负载率分布
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新图形
        """
        if not self.line_loading_history:
            print("没有可用的线路负载率数据")
            return
            
        # 获取最新的线路负载率数据
        line_loading_ratios = self.line_loading_history[-1]
        
        # 如果没有提供ax，创建新的图形
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
        
        ax.bar(range(self.n_branch), line_loading_ratios,
               color=[('red' if ratio > 0.9 else 'blue') for ratio in line_loading_ratios])
        ax.axhline(0.9, color='red', linestyle='--', label='90% 负载阈值')
        ax.set_title('线路负载率分布')
        ax.set_xlabel('线路编号')
        ax.set_ylabel('负载率')
        ax.legend()
        ax.grid(True)
        
        # 获取当前时间并格式化
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
        
        # 构建保存路径
        save_dir = os.path.join(os.path.dirname(__file__), 'result')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建保存目录: {save_dir}")
        save_path = os.path.join(save_dir, f'{timestamp}_line.png')
        
        # 保存图像并添加错误处理
        try:
            plt.savefig(save_path)
            print(f"成功保存线路负载率图像到: {save_path}")
        except Exception as e:
            print(f"保存线路负载率图像时出错: {e}")
        finally:
            plt.show()
        
    def plot_line_losses(self, ax=None):
        """绘制线路损耗分布
        
        Args:
            ax: matplotlib轴对象，如果为None则创建新图形
        """
        if not self.line_loss_history:
            print("没有可用的线路损耗数据")
            return
            
        # 获取最新的线路损耗数据
        line_losses = self.line_loss_history[-1]
        
        # 如果没有提供ax，创建新的图形
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()
        
        ax.bar(range(self.n_branch), line_losses,
               color=[('red' if i in self.critical_branches else 'blue') for i in range(self.n_branch)])
        ax.set_title('线路损耗分布')
        ax.set_xlabel('线路编号')
        ax.set_ylabel('损耗 (MW)')
        ax.grid(True)
        
        # 获取当前时间并格式化
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
        
        # 构建保存路径
        save_dir = os.path.join(os.path.dirname(__file__), 'result')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"创建保存目录: {save_dir}")
        save_path = os.path.join(save_dir, f'{timestamp}_loss.png')
        
        # 保存图像并添加错误处理
        try:
            plt.savefig(save_path)
            print(f"成功保存线路损耗图像到: {save_path}")
        except Exception as e:
            print(f"保存线路损耗图像时出错: {e}")
        finally:
            plt.show()