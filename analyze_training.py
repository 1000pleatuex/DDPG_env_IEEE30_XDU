import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def analyze_training_history():
    """分析训练历史数据，包括奖励值变化趋势和训练效果"""
    # 加载训练历史数据
    training_data = np.load('training_history.npy', allow_pickle=True).item()
    rewards = training_data['rewards']
    
    # 计算统计指标
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    # 计算滑动平均
    window_size = 10
    if len(rewards) >= window_size:
        rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        valid_range = range(window_size-1, len(rewards))
    else:
        rolling_mean = []
        valid_range = []
    
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 子图1：奖励值变化趋势
    plt.subplot(2, 1, 1)
    plt.plot(rewards, label='Episode Rewards', alpha=0.6)
    if len(rolling_mean) > 0:
        plt.plot(valid_range, rolling_mean, label=f'{window_size}-Episode Moving Average', linewidth=2)
    plt.axhline(y=mean_reward, color='r', linestyle='--', label='Mean Reward')
    plt.fill_between(range(len(rewards)), 
                     [mean_reward - std_reward] * len(rewards),
                     [mean_reward + std_reward] * len(rewards),
                     alpha=0.2, color='r', label='±1 Std Dev')
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # 子图2：奖励分布直方图
    plt.subplot(2, 1, 2)
    plt.hist(rewards, bins=30, density=True, alpha=0.7)
    plt.axvline(x=mean_reward, color='r', linestyle='--', label='Mean')
    plt.axvline(x=mean_reward + std_reward, color='g', linestyle=':', label='+1 Std Dev')
    plt.axvline(x=mean_reward - std_reward, color='g', linestyle=':', label='-1 Std Dev')
    plt.title('Reward Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    # 添加统计信息文本框
    stats_text = f'Statistical Summary:\n'\
                f'Mean Reward: {mean_reward:.2f}\n'\
                f'Std Dev: {std_reward:.2f}\n'\
                f'Max Reward: {max_reward:.2f}\n'\
                f'Min Reward: {min_reward:.2f}'
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图表
    save_dir = os.path.join(os.path.dirname(__file__), 'result')
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    save_path = os.path.join(save_dir, f'training_analysis_{timestamp}.png')
    plt.savefig(save_path)
    plt.show()
    
    # 打印分析结果
    print('\nTraining Analysis Results:')
    print(f'Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}')
    print(f'Maximum Reward: {max_reward:.2f}')
    print(f'Minimum Reward: {min_reward:.2f}')
    
    # 分析收敛性
    last_window = rewards[-window_size:]
    last_mean = np.mean(last_window)
    last_std = np.std(last_window)
    print(f'\nConvergence Analysis:')
    print(f'Final {window_size} Episodes Mean Reward: {last_mean:.2f} ± {last_std:.2f}')
    
    # 计算收敛速度（达到最终奖励90%的时间）
    convergence_threshold = 0.9 * last_mean
    for i, r in enumerate(rolling_mean):
        if r >= convergence_threshold:
            print(f'Reached 90% of final performance at episode: {i+window_size}')
            break

if __name__ == '__main__':
    analyze_training_history()