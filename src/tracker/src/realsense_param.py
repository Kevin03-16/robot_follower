import pyrealsense2 as rs
import numpy as np

class Realsense(object):
    def __init__(self):
        self.pipeline = rs.pipeline()  # 定义流程pipeline
        self.config = rs.config()  # 定义配置config
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)  # 流程开始
        align_to = rs.stream.color  # 与color流对齐
        self.align = rs.align(align_to)
        self.intr, self.depth_intrin, self.aligned_depth_frame = self.get_aligned_images()
    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        ############### 相机参数的获取 #######################
        intr = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
        '''camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                            'ppx': intr.ppx, 'ppy': intr.ppy,
                            'height': intr.height, 'width': intr.width,
                            'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                            }'''

        # # 保存内参到本地
        # # with open('./intrinsics.json', 'w') as fp:
        # #json.dump(camera_parameters, fp)
        # #######################################################

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        # depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)  # 深度图（8位）
        # depth_image_3d = np.dstack(
        #     (depth_image_8bit, depth_image_8bit, depth_image_8bit))  # 3通道深度图
        # color_image = np.asanyarray(color_frame.get_data())  # RGB图

        # 返回相机内参、深度参数、彩色图、深度图、齐帧中的depth帧
        return intr, depth_intrin, aligned_depth_frame
    
    def get_xyz(self,ux, uy):
        # ux = int((bboxes[0]+bboxes[2])/2)  # 计算像素坐标系的x
        # uy = int((bboxes[1]+bboxes[3])/2)  # 计算像素坐标系的y
        dis = self.aligned_depth_frame.get_distance(ux, uy)  
        camera_xyz = rs.rs2_deproject_pixel_to_point(self.depth_intrin, (ux, uy), dis)  # 计算相机坐标系xyz
        camera_xyz = np.round(np.array(camera_xyz), 3)  # 转成3位小数
        camera_xyz = camera_xyz.tolist()
        return camera_xyz

