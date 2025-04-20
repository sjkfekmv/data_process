import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import express as px
import plotly.graph_objects as go
import warnings
from dateutil.parser import parse
warnings.filterwarnings('ignore')

# 配置可视化风格
plt.style.use('ggplot')
sns.set_palette("husl")

# 文件路径配置
parquet_dir = '/work/share/acf6pa03fy/liyanjie/data/10G_data_new'
output_dir = 'user_profiles'
os.makedirs(output_dir, exist_ok=True)

# 添加在 build_user_profile() 函数之前
def get_age_segment(age):
    """将年龄转换为分段标签"""
    if age < 0: return "未知"
    elif age < 18: return "未成年"
    elif age <= 25: return "18-25"
    elif age <= 35: return "26-35"
    elif age <= 50: return "36-50"
    else: return "50+"

def parse_purchase_history(history):
    """解析购买历史（增强版）"""
    data = safe_json_parse(history)
    if not data:
        return {"avg_price": 0, "main_category": "无记录"}
    
    try:
        # 处理多种可能的数据结构
        items = data.get('items', [])
        if isinstance(items, str):
            items = safe_json_parse(items) or []
        
        return {
            "avg_price": float(data.get('avg_price', 0)),
            "main_category": str(data.get('categories', '未知')).split(',')[0],
            "payment_method": str(data.get('payment_method', '未知')),
            "refund_rate": 1 if str(data.get('payment_status', '')).find('退款') >=0 else 0,
            "purchase_count": len(items) if isinstance(items, (list, dict)) else 0
        }
    except Exception as e:
        print(f"解析购买历史出错: {e}")
        return {"error": str(e)}
# 辅助函数增强健壮性
def safe_json_parse(json_str):
    """更安全的JSON解析"""
    if pd.isna(json_str) or not json_str:
        return {}
    try:
        if isinstance(json_str, str):
            # 处理单引号和非标准JSON
            json_str = json_str.replace("'", '"').replace("None", "null")
            return json.loads(json_str)
        elif isinstance(json_str, dict):
            return json_str
        return {}
    except json.JSONDecodeError:
        return {}

def parse_datetime(dt_str):
    """更鲁棒的日期解析（统一时区处理）"""
    try:
        if pd.isna(dt_str):
            return pd.NaT
        
        dt = parse(dt_str)
        # 如果已经是带时区的，转换为UTC；否则标记为UTC
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc)
        else:
            return dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return pd.NaT
def parse_city(address):
    """
    从地址字符串中解析城市信息
    参数:
        address (str): 完整地址字符串
    返回:
        str: 解析出的城市名称或"未知"
    """
    if not isinstance(address, str):
        return "未知"
    
    # 常见城市标识符（可根据实际数据调整）
    city_indicators = ['市', '区', '州', '县']
    for indicator in city_indicators:
        if indicator in address:
            # 提取市名（如"北京市朝阳区" -> "北京"）
            parts = address.split(indicator)
            if parts and parts[0]:
                return parts[0] + indicator
    return "未知"

def ensure_tz_aware(dt):
    """确保时间对象带有时区（统一为UTC）"""
    if pd.isna(dt):
        return pd.NaT
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

# 画像构建函数（修正时区问题）
def build_user_profile(row):
    """构建单用户画像（增强健壮性）"""
    try:
        # 基础属性（增加空值处理）
        age = int(row['age']) if pd.notna(row['age']) else -1
        income = float(row['income']) if pd.notna(row['income']) else 0
        
        # 处理时区敏感字段
        last_login = ensure_tz_aware(parse_datetime(row['last_login']))
        registration_date = ensure_tz_aware(parse_datetime(row['registration_date']))
        
        profile = {
            "basic": {
                "age_segment": get_age_segment(age),
                "income_level": "高" if income > 500000 else ("中" if income > 100000 else "低"),
                "geo_group": f"{row.get('country', '未知')}-{parse_city(row.get('address', ''))}",
                "gender": str(row.get('gender', '未知')).replace("'", "")
            },
            "consumption": parse_purchase_history(row.get('purchase_history')),
            "activity": parse_login_history(row.get('login_history'), last_login),
            "value": calculate_user_value(row, last_login),
            "_raw_login_history": row.get('login_history', ''),  # 保存原始数据用于可视化
            "_raw_purchase_history": row.get('purchase_history', '')
        }
        return profile
    except Exception as e:
        print(f"构建用户画像出错 (ID: {row.get('id', '未知')}): {str(e)}")
        return None

def calculate_user_value(row, last_login):
    """计算用户价值（修正时区问题）"""
    purchase = parse_purchase_history(row.get('purchase_history'))
    login = parse_login_history(row.get('login_history'), last_login)
    
    # 处理可能的None值
    avg_price = purchase.get('avg_price', 0) or 0
    purchase_count = purchase.get('purchase_count', 0) or 0
    login_count = login.get('login_count', 0) or 0
    
    monetary = avg_price * purchase_count
    frequency = login_count
    
    # 统一时区处理
    now = datetime.now(timezone.utc)
    last_login_dt = ensure_tz_aware(last_login)
    recency = (now - last_login_dt).days if pd.notna(last_login_dt) else -1
    
    # RFM模型计算
    r_score = 5 if recency <=30 else (3 if recency <=90 else 1) if recency >=0 else 1
    f_score = 5 if frequency >=20 else (3 if frequency >=5 else 1)
    m_score = 5 if monetary >=10000 else (3 if monetary >=1000 else 1)
    
    return {
        "rfm_score": round(r_score * 0.5 + f_score * 0.3 + m_score * 0.2, 2),
        "monetary": monetary,
        "recency": recency if recency >=0 else "未知",
        "frequency": frequency
    }

def parse_login_history(history, last_login):
    """解析登录历史（修正时区问题）"""
    data = safe_json_parse(history)
    if not data:
        return {"login_count": 0}
    
    try:
        # 处理时间戳数据
        timestamps = data.get('timestamps', [])
        if isinstance(timestamps, str):
            timestamps = safe_json_parse(timestamps) or []
        
        valid_logins = []
        for ts in timestamps:
            dt = ensure_tz_aware(parse_datetime(ts))
            if pd.notna(dt):
                valid_logins.append(dt)
        
        # 计算最近30天活跃（使用UTC时间）
        last_30d = 0
        if pd.notna(last_login):
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=30)
            last_30d = sum(1 for d in valid_logins if d >= cutoff)
        
        return {
            "login_count": len(valid_logins),
            "devices": list(set(data.get('devices', []))),
            "last_30d_logins": last_30d,
            "avg_session_duration": float(data.get('avg_session_duration', 0))
        }
    except Exception as e:
        print(f"解析登录历史出错: {e}")
        return {"error": str(e)}

# 可视化函数（保持不变）
def generate_visualizations(user_id, profile, save_path):
    """生成三种可视化方案（完整修正版）"""
    if not profile:
        print(f"警告: 用户 {user_id} 的画像数据为空")
        return

    try:
        # 1. 基础属性雷达图
        fig1 = go.Figure()
        
        categories = ['年龄', '收入', '活跃度', '消费力', '忠诚度']
        
        # 计算雷达图数值（增强容错）
        age_value = 30  # 默认值
        if 'age_segment' in profile['basic']:
            try:
                age_seg = profile['basic']['age_segment']
                age_value = min(100, int(age_seg.split('-')[0][:2]) if '-' in age_seg else 30)
            except:
                age_value = 30
        
        income_value = 30  # 默认值
        if 'income_level' in profile['basic']:
            income_value = 90 if profile['basic']['income_level'] == '高' else (
                60 if profile['basic']['income_level'] == '中' else 30)
        
        activity_value = min(100, profile['activity'].get('last_30d_logins', 0) * 5)
        consumption_value = min(100, profile['consumption'].get('avg_price', 0) / 10)
        loyalty_value = min(100, profile['value'].get('rfm_score', 0) * 20)
        
        values = [
            age_value,
            income_value,
            activity_value,
            consumption_value,
            loyalty_value
        ]
        
        fig1.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='用户指标',
            line=dict(color='rgb(26, 118, 255)'),
            fillcolor='rgba(26, 118, 255, 0.2)'
        ))
        
        fig1.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    gridcolor='lightgray'
                )
            ),
            title=f'{user_id} 用户属性雷达图',
            font=dict(size=12),
            margin=dict(l=50, r=50, b=50, t=50)
        )
        fig1.write_html(f"{save_path}/radar_{user_id}.html")
        
        # 2. 消费-活跃度散点图（增强版）
        fig2 = px.scatter(
            x=[profile['consumption'].get('avg_price', 0)],
            y=[profile['activity'].get('login_count', 0)],
            color=[profile['basic'].get('income_level', '未知')],
            size=[np.log1p(profile['value'].get('monetary', 0))],  # 对数变换避免极端值
            labels={
                'x': '平均消费金额 (元)',
                'y': '总登录次数',
                'color': '收入等级',
                'size': '消费规模'
            },
            title=f'{user_id} 消费行为分析',
            color_discrete_map={
                '高': 'rgb(214, 39, 40)',
                '中': 'rgb(255, 127, 14)',
                '低': 'rgb(44, 160, 44)'
            },
            hover_name=[f"用户 {user_id}"],
            hover_data={
                '最近登录': [profile['value'].get('recency', '未知')],
                '退款率': [profile['consumption'].get('refund_rate', 0)]
            }
        )
        
        # 添加参考线
        fig2.update_layout(
            shapes=[
                dict(
                    type='line',
                    x0=0, x1=0,
                    y0=0, y1=max(profile['activity'].get('login_count', 0)*1.2, 10),
                    line=dict(color='gray', dash='dot')
                ),
                dict(
                    type='line',
                    y0=0, y1=0,
                    x0=0, x1=max(profile['consumption'].get('avg_price', 0)*1.2, 1000),
                    line=dict(color='gray', dash='dot')
                )
            ],
            hovermode='closest'
        )
        fig2.write_html(f"{save_path}/scatter_{user_id}.html")
        
        # 3. 时间序列热力图（完整修正版）
        if '_raw_login_history' in profile:
            login_data = safe_json_parse(profile['_raw_login_history'])
            if login_data and 'timestamps' in login_data:
                timestamps = []
                for ts in login_data['timestamps']:
                    dt = parse_datetime(ts)
                    if pd.notna(dt):
                        timestamps.append(dt)
                
                if timestamps:
                    # 创建包含所有可能时间点的完整矩阵
                    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
                    
                    # 初始化7x24的零矩阵
                    heatmap_matrix = np.zeros((7, 24))
                    
                    # 填充实际数据
                    login_df = pd.DataFrame({
                        'weekday': [d.weekday() for d in timestamps],
                        'hour': [d.hour for d in timestamps]
                    })
                    counts = login_df.groupby(['weekday', 'hour']).size()
                    
                    # 将计数填充到矩阵中
                    for (weekday, hour), count in counts.items():
                        heatmap_matrix[weekday, hour] = count
                    
                    # 生成热力图
                    fig3 = px.imshow(
                        heatmap_matrix,
                        labels=dict(x="小时", y="星期", color="登录次数"),
                        x=[f"{h:02d}:00" for h in range(24)],
                        y=weekday_names,
                        title=f'{user_id} 活跃时间分布',
                        color_continuous_scale='Blues',
                        aspect="auto",
                        zmin=0,
                        zmax=max(1, heatmap_matrix.max())  # 避免全零时的显示问题
                    )
                    
                    # 添加色条和调整布局
                    fig3.update_layout(
                        coloraxis_colorbar=dict(
                            title="登录次数",
                            thicknessmode="pixels",
                            thickness=15,
                            lenmode="pixels",
                            len=300,
                            yanchor="top",
                            y=1,
                            ticks="outside"
                        ),
                        xaxis_nticks=24,
                        yaxis_nticks=7,
                        margin=dict(l=60, r=30, b=60, t=60)
                    )
                    
                    # 添加单元格注释
                    if len(timestamps) < 200:  # 数据点较少时才显示数字
                        annotations = []
                        for y in range(7):
                            for x in range(24):
                                annotations.append(
                                    dict(
                                        x=x, y=y,
                                        text=str(int(heatmap_matrix[y, x])),
                                        showarrow=False,
                                        font=dict(color='white' if heatmap_matrix[y, x] > heatmap_matrix.max()/2 else 'black')
                                    )
                        )
                        fig3.update_layout(annotations=annotations)
                    
                    fig3.write_html(f"{save_path}/heatmap_{user_id}.html")

    except Exception as e:
        print(f"生成可视化失败 ({user_id}): {str(e)}")
        # 调试信息
        if 'heatmap_matrix' in locals():
            print(f"热力图矩阵形状: {heatmap_matrix.shape}, 最大值: {heatmap_matrix.max()}")

def process_file(file_path):
    """处理单个文件并生成画像"""
    df = pd.read_parquet(file_path)
    profiles = []
    
    for _, row in df.iterrows():
        try:
            profile = build_user_profile(row)
            profiles.append({
                "user_id": row['id'],
                "profile": profile
            })
            
            # 为前5个用户生成可视化
            if len(profiles) <= 5:
                generate_visualizations(
                    row['id'], 
                    profile,
                    output_dir
                )
                
        except Exception as e:
            print(f"处理用户 {row.get('id', 'unknown')} 时出错: {e}")
    
    # 保存所有画像数据
    filename = os.path.basename(file_path)
    pd.DataFrame(profiles).to_json(
        f"{output_dir}/{filename}_profiles.json",
        orient='records',
        force_ascii=False
    )
    
    return len(profiles)


if __name__ == "__main__":
    print("=== 用户画像生成系统 ===")
    print(f"输入目录: {parquet_dir}")
    print(f"输出目录: {output_dir}")
    
    total_start = time.time()
    parquet_files = [f for f in os.listdir(parquet_dir) 
                    if f.endswith('.parquet')][:3]  # 限制文件数用于测试
    
    if not parquet_files:
        print("错误: 未找到Parquet文件")
    else:
        total_profiles = 0
        for file in parquet_files:
            file_start = time.time()
            file_path = os.path.join(parquet_dir, file)
            print(f"\n▶ 正在处理: {file}")
            
            count = process_file(file_path)
            total_profiles += count
            
            print(f"✓ 生成 {count} 个画像 | 耗时: {time.time()-file_start:.1f}s")
    
    print(f"\n处理完成! 总生成 {total_profiles} 个用户画像")
    print(f"总耗时: {time.time()-total_start:.1f}秒")
    print(f"结果保存在: {os.path.abspath(output_dir)}")
