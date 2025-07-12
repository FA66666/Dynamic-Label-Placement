
# 可调节的变量
params_Adj = {
    'c_collision': 700,     # 300-700
    'c_pull': 20,           # 20-40
    'c_static':1,           # 0-6
    'm_collision': 10,      # 随机
    'm_feature': 10,        # 随机
    'm_max':10              # 随机
    }

param_NoAdj={
    'c_weak_collision': 0.05*params_Adj['c_collision'],
    'c_feature': params_Adj['c_collision'],
    'c_weak_feature': 0.05*params_Adj['c_collision'],
    'c_label_predict':6,
    'c_point_predict':6,
    'c_friction':6,
    'm_label_predict':1.5,
    'm_point_predict':1.5,
    'm_pull':params_Adj['m_feature'],
    's_low':1,
    's_high':3,
    's_max':6,
    's_recover':5,
    'm_weak_collision': None,  # 弱标签-标签碰撞距离参数
    'm_weak_feature': None,    # 弱标签-特征点距离参数
    
}
# 坐标可视化参数
global_params = {
    'max_x': 1000,  
    'max_y': 1000,  
    'min_x': 0,   
    'min_y': 0,   
}