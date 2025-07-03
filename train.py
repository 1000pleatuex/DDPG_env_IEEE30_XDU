# 训练相关常量定义
EPISODES = 1        # 设置默认训练轮数
MAX_STEPS = 300      # 每轮最大步数
PATIENCE = 100       # 早停耐心值
USE_OFFLINE = True   # 是否使用离线数据
OFFLINE_RATIO = 0.3  # 离线数据采样比例
STEP_INTERVAL = 300  # 设置数据获取和图表更新的间隔步数

# 导入必要的库
import numpy as np#numpy：用于矩阵和三角函数等数值计算，常与负责绘图的matplotlib.pyplot结合使用
import torch  #PyTorch：深度学习框架，用于构建和训练神经网络
import matplotlib.pyplot as plt  #matplotlib：数据可视化库，用于绘制图表
from ieee30_env import IEEE30Env  #导入环境类
from ddpg_agent import DDPGAgent  #导入DDPG智能体类
from buffer import OfflineBuffer  #导入经验回放缓冲区类
from datetime import datetime  #datetime：用于获取当前时间
import os  #os：用于处理文件和目录的操作
import glob  #glob：用于获取匹配特定模式的文件路径

def train(episodes=EPISODES, max_steps=MAX_STEPS, patience=PATIENCE, use_offline_data=USE_OFFLINE, offline_ratio=OFFLINE_RATIO): 
    """训练DDPG智能体
    
    Args:
        episodes: 训练轮次
        max_steps: 每轮最大步数
        patience: 早停耐心值，若超过该轮数没有改善则停止训练
        use_offline_data: 是否使用离线数据
        offline_ratio: 离线数据采样比例
    
    Returns:
        episode_rewards: 每轮的奖励值列表
        best_agent: 训练得到的最佳智能体
    """
    # 设置信号处理
    import signal  #signal：用于处理信号的模块
    training_interrupted = False
    
    def signal_handler(signum, frame):
        print('\nInterrupting training...')
        nonlocal training_interrupted
        training_interrupted = True
    
    # 注册SIGINT信号处理器（Ctrl+C以强制停止训练过程）
    signal.signal(signal.SIGINT, signal_handler)
    # 创建环境和智能体
    env = IEEE30Env()
    agent = DDPGAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        use_offline_data=use_offline_data,
        offline_ratio=offline_ratio
    )
    
    # 如果使用离线数据，加载历史经验
    if use_offline_data:
        # 确保checkpoints目录存在
        os.makedirs("checkpoints", exist_ok=True)
        
        # 加载离线数据
        loaded_count = agent.load_offline_data("checkpoints", "*.buffer")
        
        if loaded_count == 0:
            print("未找到离线数据，将仅使用在线学习")
            use_offline_data = False
        else:
            print(f"已加载 {loaded_count} 个离线数据缓冲区")
            
            # 打印离线数据总量
            if agent.offline_buffer is not None:
                print(f"离线数据总量: {len(agent.offline_buffer)} 条经验")
                print(f"离线数据采样比例: {offline_ratio * 100:.1f}%")
    
    # 记录训练过程
    episode_rewards = []
    best_reward = float('-inf')
    best_agent_state = None
    no_improve_count = 0
    
    # 记录训练状态
    line_loadings = []  # 记录线路负载率
    line_losses = []  # 记录线路损耗
    voltage_violations = []  # 记录电压违规次数
    generator_outputs = []  # 记录发电机组出力
    
    # 创建实时绘图窗口
    plt.ion()  # 开启交互模式
    
    # 创建包含两个子图的图表
    fig_live, (ax_reward, ax_gen) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 配置奖励值子图
    ax_reward.set_title('Real time training rewards') #设置子图标题
    ax_reward.set_xlabel('Episodes') #设置x轴标签
    ax_reward.set_ylabel('Rewards') #设置y轴标签
    ax_reward.grid(True) #显示网格线
    reward_line, = ax_reward.plot([], [], 'b-', label='Rewards')  #绘制奖励曲线
    ax_reward.legend() #显示图例
    
    # 配置发电机输出子图
    ax_gen.set_title('Real time Gen output')
    ax_gen.set_xlabel('Episodes')
    ax_gen.set_ylabel('P (MW)')
    ax_gen.grid(True)
    
    # 初始化数据列表
    reward_data = []
    gen_data = [[] for _ in range(env.action_space.shape[0])]
    lines = [ax_gen.plot([], [], label=f'Generator {i+1}')[0] for i in range(env.action_space.shape[0])]
    ax_gen.legend()
    
    # 调整子图布局
    plt.tight_layout()
    
    # 记录总步数
    total_steps = 0
    
    for episode in range(episodes): #遍历训练轮次
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_line_loading = []
        episode_line_loss = []
        episode_voltage_violation = []
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新网络
            agent.update()
            
            # 记录状态
            if 'line_loading' in info:
                episode_line_loading.append(info['line_loading'])
            if 'line_loss' in info:
                episode_line_loss.append(info['line_loss'])
            if 'voltage_violation' in info:
                episode_voltage_violation.append(info['voltage_violation'])
            
            # 记录总步数
            total_steps += 1
            
            # 按STEP_INTERVAL间隔记录数据并更新实时图表
            if total_steps % STEP_INTERVAL == 0:
                # 记录并更新奖励数据
                reward_data.append(episode_reward)
                reward_line.set_data(range(len(reward_data)), reward_data)
                ax_reward.relim()
                ax_reward.autoscale_view()
                
                # 获取发电机实际出力范围
                gen_ranges = list(zip(env.action_space.low, env.action_space.high))
                
                # 记录发电机组出力（映射到实际范围）
                real_outputs = [low + (high - low) * act for act, (low, high) in zip(action, gen_ranges)] #将动作映射到实际出力范围
                generator_outputs.append(real_outputs) #记录每个时间步的发电机组出力
                
                # 更新每个发电机的数据
                for i, output in enumerate(real_outputs): #遍历每个发电机
                    gen_data[i].append(output) #将每个时间步的出力添加到对应列表
                
                # 更新发电机输出曲线
                for i, line in enumerate(lines): #遍历每个发电机
                    line.set_data(range(len(gen_data[i])), gen_data[i]) #更新每个发电机的出力曲线
                ax_gen.relim()
                ax_gen.autoscale_view()
                
                # 刷新图表
                fig_live.canvas.draw()
                fig_live.canvas.flush_events()
            
            episode_reward += reward #累计奖励值
            state = next_state #更新状态
            
            if done:
                break
        
        episode_rewards.append(episode_reward) #记录每轮的奖励值
        
        # 记录每轮的平均状态
        if episode_line_loading:
            line_loadings.append(np.mean(episode_line_loading, axis=0)) #计算每轮的平均线路负载率
        if episode_line_loss:
            line_losses.append(np.mean(episode_line_loss, axis=0)) #计算每轮的平均线路损耗
        if episode_voltage_violation:
            voltage_violations.append(np.mean(episode_voltage_violation)) #计算每轮的平均电压违规次数
        if len(generator_outputs) > 0:
            generator_outputs.append(np.mean(generator_outputs, axis=0)) #计算每轮的平均发电机组出力
        
        # 更新最佳智能体
        current_avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward #计算当前轮次的平均奖励值
        if current_avg_reward > best_reward: #如果当前轮次的平均奖励值大于之前的最佳奖励值
            best_reward = current_avg_reward #更新最佳奖励值
            best_agent_state = agent.get_state_dict() #保存当前最佳智能体状态
            no_improve_count = 0 #重置早停计数器
        else:
            no_improve_count += 1 #早停计数器加1
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            print(f'Episode {episode + 1}, Average Reward: {current_avg_reward:.2f}')
            if line_loadings:
                print(f'Average Line Loading: {np.mean(line_loadings[-1]):.2f}')
            if line_losses:
                print(f'Average Line Loss: {np.mean(line_losses[-1]):.2f}')
            if voltage_violations:
                print(f'Average Voltage Violation: {voltage_violations[-1]:.2f}')
            if generator_outputs:
                print('\n发电机组出力:')
                for i, output in enumerate(generator_outputs[-1]):
                    print(f'Generator {i+1}: {output:.2f} MW')
        
        # 检查是否手动中断或早停
        if training_interrupted or no_improve_count >= patience: 
            # 获取奖励统计信息
            reward_stats = agent.get_reward_stats()
            stop_reason = 'manual interruption' if training_interrupted else 'no improvement' #获取停止原因【手动Ctrl+C】或【长期没有进展触发早停】
            print(f'\nStopping at episode {episode + 1} due to {stop_reason}')
            print(f'Mean Reward: {reward_stats["mean_reward"]:.2f}')
            print(f'Max Reward: {reward_stats["max_reward"]:.2f}')
            break
    
    # 恢复最佳智能体状态
    if best_agent_state is not None:
        agent.load_state_dict(best_agent_state)
    
    # 关闭实时绘图窗口
    plt.close(fig_live)
    plt.ioff()  # 关闭交互模式
    
    # 保存训练状态数据
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # 确保checkpoints目录存在
    os.makedirs("checkpoints", exist_ok=True)
    
    # 保存训练状态数据
    training_data = {
        'rewards': episode_rewards,
        'line_loadings': line_loadings,
        'line_losses': line_losses,
        'voltage_violations': voltage_violations,
        'generator_outputs': generator_outputs
    }
    np.save(f'checkpoints/training_history_{timestamp}.npy', training_data)
    
    # 保存经验回放缓冲区作为离线数据供下次训练使用
    experience_path = agent.save_experience()
    print(f"经验回放缓冲区已保存到: {experience_path}")
    
    return episode_rewards, agent





def plot_training_results(rewards, env):
    """绘制训练结果，包括奖励曲线、线路负载率、损耗分布和发电机组出力波动"""
    try:
        # 创建一个1行2列的子图布局
        fig = plt.figure(figsize=(15, 6))
        
        # 子图1：训练奖励曲线
        ax1 = fig.add_subplot(121)
        ax1.plot(rewards)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # 子图2：发电机组出力波动
        ax2 = fig.add_subplot(122)
        try:
            # 获取最新的训练历史数据文件
            history_files = glob.glob('checkpoints/training_history_*.npy')
            if history_files:
                # 按时间戳排序，获取最新的文件
                latest_file = max(history_files)
                # 加载训练历史数据
                training_data = np.load(latest_file, allow_pickle=True).item()
                generator_outputs = training_data.get('generator_outputs', [])
                
                if generator_outputs:
                    generator_outputs = np.array(generator_outputs)
                    episodes = range(len(generator_outputs))
                    
                    # 获取发电机实际出力范围
                    gen_ranges = list(zip(env.action_space.low, env.action_space.high))
                    
                    # 绘制每个发电机的出力曲线
                    for i in range(6):  # 6个发电机
                        ax2.plot(episodes, generator_outputs[:, i],
                                label=f'Generator {i+1} ({gen_ranges[i][0]:.1f}-{gen_ranges[i][1]:.1f} MW)')
                    
                    ax2.set_title('Generator Outputs Over Episodes')
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Power Output (MW)')
                    ax2.grid(True)
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax2.set_ylim(bottom=0)  # 设置y轴从0开始
            else:
                ax2.text(0.5, 0.5, 'Generator Output Data Not Available',
                         ha='center', va='center')
        except Exception as e:
            print(f'Warning: Failed to plot generator outputs: {e}')
            ax4.text(0.5, 0.5, 'Generator Output Data Not Available',
                     ha='center', va='center')
        
        # 设置总标题
        fig.suptitle('Training Results Analysis', fontsize=16, y=1.02)
        
        # 调整子图布局
        plt.tight_layout()
        
        # 确保result目录存在
        save_dir = os.path.join(os.path.dirname(__file__), 'result')
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取当前时间并格式化
        current_time = datetime.now()
        timestamp = current_time.strftime('%Y-%m-%d-%H-%M')
        
        # 构建保存路径
        save_path = os.path.join(save_dir, f'{timestamp}.png')
        
        # 保存图像
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
        
    except Exception as e:
        print(f'Error in plot_training_results: {e}')
        plt.close('all')  # 清理所有图形

def evaluate(agent, env, episodes=10):
    """评估智能体性能"""
    total_rewards = []
    
    for episode in range(episodes): 
        state = env.reset() 
        episode_reward = 0 
        done = False 
        
        while not done:
            action = agent.select_action(state, noise_scale=0.0)  # 评估时不使用噪声
            next_state, reward, done, _ = env.step(action) 
            episode_reward += reward # 累加奖励
            state = next_state # 更新状态
        
        total_rewards.append(episode_reward) # 记录总奖励
    
    avg_reward = np.mean(total_rewards) # 计算平均奖励
    std_reward = np.std(total_rewards) # 计算奖励标准差
    print(f'Evaluation over {episodes} episodes:')
    print(f'Average Reward: {avg_reward:.2f} ± {std_reward:.2f}')

def main():
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 训练参数
    train_params = {
        'episodes': EPISODES,          # 使用常量定义的训练轮数
        'max_steps': MAX_STEPS,       # 每轮最大步数
        'patience': PATIENCE,         # 早停耐心值
        'use_offline_data': USE_OFFLINE,  # 是否使用离线数据
        'offline_ratio': OFFLINE_RATIO    # 离线数据采样比例
    }
    
    # 训练智能体
    print('Starting training...')
    print(f"使用离线数据: {'是' if train_params['use_offline_data'] else '否'}")
    rewards, best_agent = train(**train_params)
    


    """
    以下评估和保存智能体功能尚未实现
    """

    # 创建环境用于评估
    env = IEEE30Env()
    
    # 绘制训练结果
    print('\nPlotting training results...')
    plot_training_results(rewards, env)
    
    # 评估智能体性能
    print('\nEvaluating agent...') 
    evaluate(best_agent, env) 
    
    # 保存最佳智能体
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M') # 确保checkpoints目录存在
    os.makedirs("checkpoints", exist_ok=True) 
    torch.save(best_agent.state_dict(), f'checkpoints/best_agent_{timestamp}.pth') # 保存最佳智能体

if __name__ == '__main__':
    main()