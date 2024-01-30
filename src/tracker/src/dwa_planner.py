import numpy as np
from math import *
import rospy

#参数设置
class DWAPlanner():

    def __init__(self, param, radius=1, predict_time=2.0) -> None:
        # 设置机器人的最大线速度和最大角速度
        self.V_max = param['V_max']
        self.V_min = param['V_min']
        self.W_max = param['W_max']
        self.W_min = param['W_min']

        # 设置DWA算法的参数
        self.V_acc = param['V_acc']
        self.W_acc = param['W_acc']
        self.linear_resolution = param['linear_resolution']
        self.angular_resolution = param['angular_resolution']
        self.dt = param['dt'] # 时间间隔
        self.alpha = param['alpha'] #距离目标点的评价函数的权重系数
        self.beta = param['beta'] #速度评价函数的权重系数
        self.gamma = param['gamma'] #距离障碍物距离的评价函数的权重系数
        self.radius = radius #机器人模型半径
        self.predict_time = predict_time #模拟轨迹的持续时间

    @staticmethod
    #距离目标点的评价函数
    def Goal_Cost(Goal,Pos):
        return sqrt((Pos[-1,0]-Goal[0])**2+(Pos[-1,1]-Goal[1])**2)
 
    #速度评价函数
    def Velocity_Cost(self, Pos):
        return self.V_max - Pos[-1,3]
 
    #距离障碍物距离的评价函数
    # 代表机器人在当前轨迹上与最近的障碍物之间的距离。如果这条轨迹上没有障碍物，那就将其设定为一个常数。
    def Obstacle_Cost(self, Pos, Obstacle):
        MinDistance = float('Inf')          #初始化时候机器人周围无障碍物所以最小距离设为无穷
        for i in range(len(Pos)):           #对每一个位置点循环
            for j in range(len(Obstacle)):  #对每一个障碍物循环
                Current_Distance = sqrt((Pos[i,0]-Obstacle[j][0])**2+(Pos[i,1]-Obstacle[j][1])**2)  #求出每个点和每个障碍物距离
                if Current_Distance < self.radius + 0.5:            #如果小于机器人自身的半径那肯定撞到障碍物了返回的评价值自然为无穷
                    return float('Inf')
                if Current_Distance < MinDistance:
                    MinDistance=Current_Distance         #得到点和障碍物距离的最小
        return 1 / MinDistance
 
    #速度采用
    def V_Range(self, X):
        # 由于电机扭矩先限制，存在最大加减速，在动态窗口内的速度就是机器人可以达到的最大速度
        Vmin_Actual = X[3] - self.V_acc * self.dt          #实际在dt时间间隔内的最小速度
        Vmax_actual = X[3] + self.V_acc * self.dt          #实际载dt时间间隔内的最大速度
        Wmin_Actual = X[4] - self.W_acc * self.dt         #实际在dt时间间隔内的最小角速度
        Wmax_Actual = X[4] + self.W_acc * self.dt         #实际在dt时间间隔内的最大角速度
        VW = [max(self.V_min,Vmin_Actual),min(self.V_max,Vmax_actual),max(self.W_min,Wmin_Actual),min(self.W_max,Wmax_Actual)]  #因为前面本身定义了机器人最小最大速度所以这里取交集
        return VW
    
    #一条模拟轨迹路线中的位置，速度计算
    def Motion(self, X, u):
        X[0] += u[0] * self.dt*cos(X[2])           #x方向上位置
        X[1] += u[0] * self.dt*sin(X[2])           #y方向上位置
        X[2] += u[1] * self.dt                     #角度变换
        X[3] = u[0]                         #速度
        X[4] = u[1]                         #角速度
        return X
 
    #一条模拟轨迹的完整计算
    def Calculate_Traj(self, X, u):
        Traj = np.array(X)
        Xnew = np.array(X)
        time = 0
        while time <= self.predict_time:        #一条模拟轨迹时间
            Xnew = self.Motion(Xnew, u)
            Traj = np.vstack((Traj, Xnew))   #一条完整模拟轨迹中所有信息集合成一个矩阵
            time = time + self.dt
        return Traj
 
    #DWA核心计算
    def dwa_Core(self, X, u, obstacles):
        vw=self.V_Range(X)
        best_traj=np.array(X)
        min_score=10000.0                 #随便设置一下初始的最小评价分数
        for v in np.arange(vw[0], vw[1], self.linear_resolution):         #对每一个线速度循环
            for w in np.arange(vw[2], vw[3], self.angular_resolution):     #对每一个角速度循环
                traj = self.Calculate_Traj(X,[v,w])
                # goal_score = self.Goal_Cost(goal,traj)
                vel_score = self.Velocity_Cost(traj)
                obs_score = self.Obstacle_Cost(traj,obstacles)
                # score = self.alpha * goal_score+ self.beta * vel_score + self.gamma * obs_score
                score = self.beta * vel_score + self.gamma * obs_score
                if min_score >= score:                    #得出最优评分和轨迹
                    min_score = score
                    u=np.array([v,w])
                    best_traj=traj
        return u, best_traj
 
    def dwa_plan(self, X, u, Obstacles):
        # X 设定初始位置，角速度，线速度
        # u 设定初始速度
        # goal 设定目标位置
        global_tarj=np.array(X)
        for i in range(1000): #循环1000次，这里可以直接改成while的直到循环到目标位置
            u, current = self.dwa_Core(X, u, Obstacles)
            X = self.Motion(X, u)
            global_traj = np.vstack((global_tarj, X))
            if sqrt((X[0]-current[-1, 0])**2+(X[1]-current[-1, 1])**2) <= self.radius:  #判断是否到达目标点
                break
        return u
