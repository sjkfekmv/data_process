import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

# 指定Parquet文件目录
parquet_dir = '/Users/aurora/Downloads/DATA/10G_data_new'

# 设置Pandas显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10)

# 输出文件
problem_file = 'problem.txt'
need_delete_file = 'need_delete.txt'

# 清空或创建输出文件
with open(problem_file, 'w') as f:
    f.write("常规异常值记录:\n")
with open(need_delete_file, 'w') as f:
    f.write("严重异常值记录(建议删除):\n")

def detect_anomalies(df):
    """检测数据中的异常值"""
    problem_records = []
    delete_records = []
    
    # 1. 严重异常值检测
    # 检查ID或邮箱为空
    null_id_or_email = df[df['id'].isna() | df['email'].isna()]
    if not null_id_or_email.empty:
        delete_records.append("ID或邮箱为空的记录:\n" + null_id_or_email.to_string() + "\n")
    
    # 检查注册日期大于最后登录时间
    try:
        df['last_login_dt'] = pd.to_datetime(df['last_login'])
        df['registration_dt'] = pd.to_datetime(df['registration_date'])
        invalid_dates = df[df['registration_dt'] > df['last_login_dt']]
        if not invalid_dates.empty:
            delete_records.append("注册日期晚于最后登录时间的记录:\n" + invalid_dates[['id', 'last_login', 'registration_date']].to_string() + "\n")
    except Exception as e:
        problem_records.append(f"日期转换错误: {e}\n")
    
    # 检查收入异常(3σ原则)
    try:
        income_mean = df['income'].mean()
        income_std = df['income'].std()
        threshold = income_mean + 3 * income_std
        abnormal_income = df[df['income'] > threshold]
        if not abnormal_income.empty:
            delete_records.append(f"收入超过3σ原则(>{threshold:.2f})的记录:\n" + abnormal_income[['id', 'income']].to_string() + "\n")
    except Exception as e:
        problem_records.append(f"收入异常检测错误: {e}\n")
    
    # 2. 常规异常值检测
    # 检查年龄异常
    abnormal_age = df[(df['age'] < 0) | (df['age'] > 120)]
    if not abnormal_age.empty:
        problem_records.append("年龄异常(<0或>120)的记录:\n" + abnormal_age[['id', 'age']].to_string() + "\n")
    
    # 检查性别字段
    abnormal_gender = df[~df['gender'].isin(['男', '女'])]
    if not abnormal_gender.empty:
        problem_records.append("性别字段异常(非'男'/'女')的记录:\n" + abnormal_gender[['id', 'gender']].to_string() + "\n")
    
    # 检查邮箱格式
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    invalid_emails = df[~df['email'].str.contains(email_pattern, na=False)]
    if not invalid_emails.empty:
        problem_records.append("邮箱格式不正确的记录:\n" + invalid_emails[['id', 'email']].to_string() + "\n")
    
    # 检查is_active是否为布尔值
    if df['is_active'].dtype != 'bool':
        problem_records.append("is_active列包含非布尔值\n")
    
    return problem_records, delete_records

# 记录总开始时间
total_start_time = time.time()

# 1. 列出目录中的所有Parquet文件
parquet_files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet')]

if not parquet_files:
    print("该目录中没有找到Parquet文件")
else:
    print(f"找到 {len(parquet_files)} 个Parquet文件:")
    for i, f in enumerate(parquet_files[:5]):
        print(f"{i+1}. {f}")
    if len(parquet_files) > 5:
        print(f"...以及另外 {len(parquet_files)-5} 个文件")
    
    # 初始化统计信息
    total_problems = 0
    total_deletes = 0
    total_files = 0
    
    # 2. 处理每个文件
    for file in parquet_files:
        file_start_time = time.time()
        file_path = os.path.join(parquet_dir, file)
        print(f"\n开始处理文件: {file}")
        
        try:
            df = pd.read_parquet(file_path)
            
            # 检测异常值
            problems, deletes = detect_anomalies(df)
            
            # 写入结果文件
            with open(problem_file, 'a') as f:
                f.write(f"\n=== 文件 {file} 的常规异常 ===\n")
                f.write("\n".join(problems))
                total_problems += len(problems)
            
            with open(need_delete_file, 'a') as f:
                f.write(f"\n=== 文件 {file} 的严重异常 ===\n")
                f.write("\n".join(deletes))
                total_deletes += len(deletes)
            
            # 计算文件处理时间
            file_time = time.time() - file_start_time
            total_files += 1
            
            # 显示当前文件处理信息
            print(f"处理完成: 发现 {len(problems)} 条常规异常, {len(deletes)} 条严重异常")
            print(f"处理时间: {file_time:.2f} 秒")
            
        except Exception as e:
            file_time = time.time() - file_start_time
            print(f"处理文件 {file} 时出错: {e}")
            print(f"错误处理时间: {file_time:.2f} 秒")
            with open(problem_file, 'a') as f:
                f.write(f"\n处理文件 {file} 时出错: {e}\n")

    # 计算总处理时间
    total_time = time.time() - total_start_time
    
    # 打印汇总统计
    print("\n" + "="*50)
    print("处理完成! 汇总统计:")
    print(f"- 处理文件总数: {total_files}/{len(parquet_files)}")
    print(f"- 发现常规异常总数: {total_problems}")
    print(f"- 发现严重异常总数: {total_deletes}")
    print(f"- 总处理时间: {total_time:.2f} 秒")
    print(f"- 平均每个文件处理时间: {total_time/max(1, total_files):.2f} 秒")
    print("结果已保存到:")
    print(f"- 常规异常: {problem_file}")
    print(f"- 严重异常: {need_delete_file}")
    print("="*50)
