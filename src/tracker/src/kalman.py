from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanPredictor:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        self.kf.F = np.array([[1, 0, 0.5, 0],
                    [0, 1, 0, 0.5],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
        
        # 设置测量矩阵
        self.kf.H = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0]])

        # 设置过程噪声和测量噪声的协方差矩阵
        self.kf.Q = np.array([[0.01, 0, 0, 0],
                        [0, 0.01, 0, 0],
                        [0, 0, 0.01, 0],
                        [0, 0, 0, 0.01]])
        self.kf.R = np.array([[0.1, 0],
                        [0, 0.1]])
            
    # 定义一个函数用来预测未来位置
    def predict_next_position(self, num_steps):
        predicted_positions = []
        for _ in range(num_steps):
            self.kf.predict()  # 使用系统模型进行状态预测
            predicted_positions.append(self.kf.x[:2])  # 将预测的位置存储到列表中
        return predicted_positions

    def update_kalman_filter(self, measurement):
        self.kf.update(measurement)
        
    