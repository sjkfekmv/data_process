import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 设置文件夹路径
folder_path = '/Users/aurora/Downloads/DATA/10G_data_new'

def enhanced_user_analysis(folder_path):
    """
    增强版用户数据分析:
    1. 用户年龄分布(饼图)
    2. 活跃用户收入分布(柱状图)
    3. 用户注册时间趋势(折线图)
    """
    # 初始化DataFrame来存储所有数据
    all_data = pd.DataFrame()
    parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    
    if not parquet_files:
        print("未找到Parquet文件，请检查文件夹路径")
        return
    
    print(f"找到 {len(parquet_files)} 个Parquet文件，开始读取...")
    
    # 逐个读取Parquet文件
    for file in tqdm(parquet_files):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_parquet(file_path)
            
            # 选择需要的列(假设列名为'age', 'is_active', 'income', 'registration_date')
            cols_to_keep = []
            if 'age' in df.columns:
                cols_to_keep.append('age')
            if 'is_active' in df.columns:
                cols_to_keep.append('is_active')
            if 'income' in df.columns:
                cols_to_keep.append('income')
            if 'registration_date' in df.columns:
                cols_to_keep.append('registration_date')
            
            if cols_to_keep:
                all_data = pd.concat([all_data, df[cols_to_keep]], ignore_index=True)
        except Exception as e:
            print(f"读取文件 {file} 时出错: {e}")
    
    if all_data.empty:
        print("没有找到有效数据")
        return
    
    # 创建结果目录
    results_dir = os.path.join(folder_path, 'analysis_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. 年龄分布分析(饼图)
    if 'age' in all_data.columns:
        analyze_age_distribution(all_data, results_dir)
    
    # 2. 活跃用户收入分布(柱状图)
    if 'is_active' in all_data.columns and 'income' in all_data.columns:
        analyze_active_user_income(all_data, results_dir)
    
    # 3. 用户注册时间趋势(折线图)
    if 'registration_date' in all_data.columns:
        analyze_registration_trend(all_data, results_dir)

def analyze_age_distribution(data, save_dir):
    """分析年龄分布并绘制饼图"""
    print("\n正在分析年龄分布...")
    
    # 统计年龄分布
    age_distribution = data['age'].value_counts(normalize=True) * 100
    
    # 如果年龄值过多，进行分组
    if len(age_distribution) > 15:
        bins = [0, 18, 25, 35, 45, 55, 65, 100]
        labels = ['0-18', '19-25', '26-35', '36-45', '46-55', '56-65', '65+']
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
        age_distribution = data['age_group'].value_counts(normalize=True) * 100
        age_distribution = age_distribution.sort_index()
        
        # 绘制分组后的饼图
        plt.figure(figsize=(12, 8))
        age_distribution.plot.pie(autopct='%1.1f%%', startangle=90, 
                                textprops={'fontsize': 12}, pctdistance=0.85)
        plt.title('user_age_dis', fontsize=15, pad=20)
    else:
        # 直接绘制原始年龄的饼图
        plt.figure(figsize=(12, 8))
        age_distribution.plot.pie(autopct='%1.1f%%', startangle=90, 
                                textprops={'fontsize': 12}, pctdistance=0.85)
        plt.title('user_age_dis', fontsize=15, pad=20)
    
    plt.ylabel('')
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'age_distribution_pie_chart.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"年龄分布饼图已保存至: {save_path}")
    plt.close()

def analyze_active_user_income(data, save_dir):
    """分析活跃用户收入分布并绘制柱状图"""
    print("\n正在分析活跃用户收入分布...")
    
    # 筛选活跃用户
    active_users = data[data['is_active'] == True]
    
    if active_users.empty:
        print("没有活跃用户数据")
        return
    
    # 定义收入区间(可根据实际数据调整)
    income_bins = [0, 30000, 50000, 75000, 100000, 150000, np.inf]
    income_labels = [
        '<30k', '30k-50k', '50k-75k', 
        '75k-100k', '100k-150k', '>150k'
    ]
    
    # 分类收入
    active_users['income_group'] = pd.cut(
        active_users['income'], 
        bins=income_bins, 
        labels=income_labels,
        right=False
    )
    
    # 统计各收入区间占比
    income_dist = active_users['income_group'].value_counts(normalize=True) * 100
    income_dist = income_dist.sort_index()
    
    # 绘制柱状图
    plt.figure(figsize=(12, 6))
    bars = income_dist.plot.bar(color='#4c72b0', edgecolor='black')
    
    # 添加数值标签
    for bar in bars.patches:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%',
                 ha='center', va='bottom', fontsize=10)
    
    plt.title('active_user_dis', fontsize=15, pad=15)
    plt.xlabel('Income range', fontsize=12)
    plt.ylabel('Proportion(%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'active_users_income_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"活跃用户收入分布图已保存至: {save_path}")
    plt.close()

def analyze_registration_trend(data, save_dir):
    """分析用户注册时间趋势并绘制折线图"""
    print("\n正在分析用户注册时间趋势...")
    
    # 转换日期格式(假设是时间戳或字符串)
    try:
        if np.issubdtype(data['registration_date'].dtype, np.number):
            # 如果是数字时间戳(假设是秒级)
            data['registration_date'] = pd.to_datetime(data['registration_date'], unit='s')
        else:
            # 尝试自动解析日期字符串
            data['registration_date'] = pd.to_datetime(data['registration_date'])
    except Exception as e:
        print(f"日期转换出错: {e}")
        return
    
    # 按日期统计注册量
    reg_daily = data['registration_date'].dt.floor('D').value_counts().sort_index()
    
    # 按月统计(备选方案)
    reg_monthly = data['registration_date'].dt.to_period('M').value_counts().sort_index()
    reg_monthly.index = reg_monthly.index.to_timestamp()
    
    # 绘制折线图
    plt.figure(figsize=(14, 6))
    
    # 根据数据量决定使用日数据还是月数据
    if len(reg_daily) > 60:  # 如果超过60天，使用月数据
        reg_monthly.plot(linewidth=2, marker='o', color='#d62728')
        plt.title('User registration quantity trend (monthly)', fontsize=15, pad=15)
        x_label = 'month'
    else:
        reg_daily.plot(linewidth=2, marker='o', color='#d62728')
        plt.title('User registration quantity trend (day)', fontsize=15, pad=15)
        x_label = 'date'
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('registration num', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 格式化x轴日期显示
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'user_registration_trend.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"用户注册趋势图已保存至: {save_path}")
    plt.close()

# 执行分析
enhanced_user_analysis(folder_path)
