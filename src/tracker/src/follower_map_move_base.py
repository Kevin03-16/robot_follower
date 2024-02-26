#!/usr/bin/env python3

import rospy
import threading
import actionlib
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3, PointStamped, PoseStamped, Quaternion, Point, Pose
from dwa_planner import DWAPlanner
import tf2_geometry_msgs
from ros_tracker.msg import position as PositionMsg
from ros_tracker.msg import delta as DeltaMsg
from std_msgs.msg import String as StringMsg
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from kalman import KalmanPredictor
from collections import deque
from nav_msgs.srv import GetMap
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros



class Follower:
	def __init__(self):
		# as soon as we stop receiving Joy messages from the ps3 controller we stop all movement:
		# self.predict_thread = threading.Thread(target=self.predict_path)
		self.kalman = KalmanPredictor()
		self.history_list = deque(maxlen=2)
		self.switchMode= rospy.get_param('~switchMode') # if this is set to False the O button has to be kept pressed in order for it to move
		self.max_speed = rospy.get_param('~maxSpeed') 
		self.max_gimbal_speed = rospy.get_param('~maxGimbalSpeed')
		self.controllButtonIndex = rospy.get_param('~controllButtonIndex')
		self.radius = rospy.get_param('~car_radius')
		self.k = np.array([[1206.8897719532354, 0.0, 960.5], [0.0, 1206.8897719532354, 540.5], [0.0, 0.0, 1.0]])
		self.active=False
		self.goal_in_map = PoseStamped()
		self.goal_in_camera = PointStamped()
		self.robot_pos = None
		self.goal_pos = None
		self.person_loss = False
		self.delta_x, self.delta_y = 0, 0
		self.Target_arrived = False
		self.avoid_obs = False
		self.inflation = 4 * self.radius
		self.tf_buffer = tf2_ros.Buffer()
		tf_listener = tf2_ros.TransformListener(self.tf_buffer)
		# self.tf_buffer_odom = tf2_ros.Buffer()
		# tf_listener_2 = tf2_ros.TransformListener(self.tf_buffer_odom)

		self.cmdVelPublisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		# self.gimbalPublisher = rospy.Publisher("/joint_states", JointState, queue_size=10)
		self.positionSubscriber = rospy.Subscriber('current_position', PositionMsg, self.positionUpdateCallback)
		self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
		self.trackerInfoSubscriber = rospy.Subscriber('tracker_info', StringMsg, self.trackerInfoCallback)
		self.predictorInfoPublisher = rospy.Publisher('predictor_info', StringMsg, queue_size=10)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
		self.target_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=3)
		self.targetDist = rospy.get_param('~targetDist')
		# 获取地图边界
		rospy.wait_for_service('/static_map')
		get_map = rospy.ServiceProxy('/static_map', GetMap)
		map_resp = get_map()
		self.map_info = map_resp.map.info

		# the first parameter is the angular target (0 degrees always) the second is the target distance (say 1 meter)
		# self.joint_state = JointState()
		# self.joint_state.name = 'camera_joint'
		self.searching_velocity = Twist()	
		self.searching_velocity.linear = Vector3(0,0,0.)
		self.searching_velocity.angular= Vector3(0., 0.,0.8)
		self.min_distance_threshold = 1 # 设置最小距离阈值，低于该值时执行避障动作

		self.tracking_velocity = Twist()
		PID_param = rospy.get_param('~PID_controller')	
		self.PID_controller = simplePID([0, self.targetDist], PID_param['P'], PID_param['I'], PID_param['D'])
		# 发布目标位置到move_base节点
		self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		self.move_base_goal = MoveBaseGoal()
		self.move_base_client.wait_for_server()
		# self.predict_thread.start()
	
	
	def odom_callback(self, data):
        # 从里程计信息中提取机器人当前位置和方向
		odom = PointStamped()
		odom.header.frame_id = 'odom'
		odom.header.stamp = rospy.Time.now()
		odom.point.x = data.pose.pose.position.x
		odom.point.y = data.pose.pose.position.y
		odom.point.z = data.pose.pose.position.z
		try:
			transform = self.tf_buffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(1.0))
			self.robot_pos = tf2_geometry_msgs.do_transform_point(odom, transform)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logerr(f"Failed to transform the goal: {e}")
			exit()
		return

	def scan_callback(self, scan_msg):
		return
		# if min(scan_msg.ranges) <= self.inflation:
		# 	self.active = False
		# 	self.stopMoving()

	def send_move_base_goal(self, kalman=True, step=3):
		target_pos_x = []
		target_pos_y = []
		if kalman:
			self.kalman.kf.x[:2] = self.history_list[-1][:2].reshape(2,1) # 初始位置
			if len(self.history_list) > 1:
				self.kalman.kf.x[2:] = (self.history_list[-1][:2] - self.history_list[-2][:2]).reshape(2,1)  # 初始速度
			else:
				self.kalman.kf.x[2:] = np.array([[1],[1]])
			predict_position = self.kalman.predict_next_position(num_steps=step)
			for pos in predict_position:
				target_pos_x.append(pos[0])
				target_pos_y.append(pos[1])
		else:
			for i in range(step):
				target_pos_x.append(self.history_list[-1][0] + i * self.delta_x)
				target_pos_y.append(self.history_list[-1][1] + i * self.delta_y)
		target_pos_z = self.history_list[-1][-1]

		# for j in range(step):
		# 	x, y = self.check_out_boundary(target_pos_x[j], target_pos_y[j])
		# 	self.move_base_goal.target_pose.header.frame_id = 'map'
		# 	self.move_base_goal.target_pose.header.stamp = rospy.Time.now()
		# 	self.move_base_goal.target_pose.pose.orientation.w = 1
		# 	self.move_base_goal.target_pose.pose.position.x = x
		# 	self.move_base_goal.target_pose.pose.position.y = y
		# 	self.move_base_goal.target_pose.pose.position.z = target_pos_z
		# 	self.move_base_client.send_goal(self.move_base_goal)
		# 	self.move_base_client.wait_for_result()

		x, y = self.check_out_boundary(target_pos_x[-1], target_pos_y[-1])
		self.move_base_goal.target_pose.header.frame_id = 'map'
		self.move_base_goal.target_pose.header.stamp = rospy.Time.now()
		self.move_base_goal.target_pose.pose.orientation.w = 1
		self.move_base_goal.target_pose.pose.position.x = x
		self.move_base_goal.target_pose.pose.position.y = y
		self.move_base_goal.target_pose.pose.position.z = target_pos_z
		self.move_base_client.send_goal(self.move_base_goal)
		self.move_base_client.wait_for_result()
		if self.move_base_client.get_state() == 3:
			self.predictorInfoPublisher.publish('predictor_finished')
		
	def pid_control(self, distance, angleX):
		rospy.loginfo('Angle: {}, Distance: {}, '.format(angleX, distance))
		
		# call the PID controller to update it and get new speeds
		[uncliped_ang_speed, uncliped_lin_speed] = self.PID_controller.update([angleX, distance])
		# [uncliped_y_ang_speed, _] = self.PID_controller.update([angleY, distance])
		
		# clip these speeds to be less then the maximal speed specified above
		angularSpeed = np.clip(-uncliped_ang_speed, -self.max_speed, self.max_speed)
		linearSpeed  = np.clip(-uncliped_lin_speed, -self.max_speed, self.max_speed)
		return [linearSpeed, angularSpeed]

	def check_out_boundary(self, x, y):
		# width = 187
		# height = 425
		# resolution = 0.05
		# position = (-5.18, -11.31, 0)
		if x > (self.map_info.origin.position.x + (self.map_info.width * self.map_info.resolution - self.min_distance_threshold)):
			x = self.map_info.origin.position.x + (self.map_info.width * self.map_info.resolution - self.min_distance_threshold)

		if y > (self.map_info.origin.position.y + (self.map_info.height * self.map_info.resolution- self.min_distance_threshold)):
			y = self.map_info.origin.position.y + (self.map_info.height * self.map_info.resolution- self.min_distance_threshold)
		return x, y
	
	# def positionUpdateCallback(self, position):
	# 	uv_camera = position.distance * np.linalg.inv(self.k).dot(np.array([[position.goalX],[position.goalY],[1]]))
	# 	# self.goal_in_camera.header.frame_id = 'camera_rgb_optical_frame'
	# 	self.goal_in_camera.point.x = uv_camera[0]
	# 	self.goal_in_camera.point.y = uv_camera[1]
	# 	self.goal_in_camera.point.z = uv_camera[2]
	# 	try:
	# 		transform = self.tf_buffer.lookup_transform('map', 'camera_rgb_optical_frame', rospy.Time(0), rospy.Duration(1.0))
	# 		map_point = tf2_geometry_msgs.do_transform_point(self.goal_in_camera, transform)
	# 	except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
	# 		rospy.logerr(f"Failed to transform the goal: {e}")
	# 		exit()
	# 	theta = math.atan2((map_point.point.y - self.robot_pos.point.y), (map_point.point.x - self.robot_pos.point.x))	
	# 	self.goal_in_map.header.frame_id = "map"  # 目标点的坐标系为map
	# 	self.goal_in_map.header.stamp = rospy.Time.now()
	# 	self.goal_in_map.pose.position.x = map_point.point.x - self.targetDist * math.cos(theta)
	# 	self.goal_in_map.pose.position.y = map_point.point.y - self.targetDist * math.sin(theta)
	# 	self.goal_in_map.pose.position.z = map_point.point.z
	# 	self.goal_in_map.pose.orientation.w = 1
	# 	if self.goal_pos is not None:
	# 		self.delta_x = map_point.point.x - self.goal_pos.point.x
	# 		self.delta_y = map_point.point.y - self.goal_pos.point.y
	# 	self.goal_pos = map_point
	# 	self.target_pub.publish(self.goal_in_map)

	def positionUpdateCallback(self, position):
		uv_camera = position.distance * np.linalg.inv(self.k).dot(np.array([[position.goalX],[position.goalY],[1]]))
		# self.goal_in_camera.header.frame_id = 'camera_rgb_optical_frame'
		self.goal_in_camera.point.x = uv_camera[0]
		self.goal_in_camera.point.y = uv_camera[1]
		self.goal_in_camera.point.z = uv_camera[2]
		if self.goal_in_camera.point.z > self.targetDist:
			try:
				transform = self.tf_buffer.lookup_transform('map', 'camera_rgb_optical_frame', rospy.Time(0), rospy.Duration(1.0))
				map_point = tf2_geometry_msgs.do_transform_point(self.goal_in_camera, transform)
				# x, y = self.check_out_boundary(map_point.point.x, map_point.point.y)
				self.move_base_goal.target_pose.header.frame_id='map'
				self.move_base_goal.target_pose.header.stamp=rospy.Time.now()
				# self.move_base_goal.target_pose.pose.position.x = x
				# self.move_base_goal.target_pose.pose.position.y = y
				# self.move_base_goal.target_pose.pose.position.z = map_point.point.z
				self.move_base_goal.target_pose.pose.position = map_point.point
				self.move_base_goal.target_pose.pose.orientation.w = 1
				self.history_list.append(np.array([map_point.point.x, map_point.point.y, map_point.point.z]))
				self.kalman.update_kalman_filter(self.history_list[-1][:2].reshape(2,1))
				if len(self.history_list) > 1:
					[self.delta_x, self.delta_y] = self.history_list[-1][:2] - self.history_list[-2][:2]
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				rospy.logerr(f"Failed to transform the goal: {e}")
				exit()
			self.move_base_client.send_goal(self.move_base_goal)
		else:
			self.move_base_client.cancel_all_goals()
			cmd_vel = self.pid_control(position.distance, position.angleX)
			self.tracking_velocity.angular= Vector3(0., 0.,cmd_vel[1])
			self.cmdVelPublisher.publish(self.tracking_velocity)

	def trackerInfoCallback(self, info):
		if info.data == 'No_people':
			rospy.logwarn(info.data)
			if self.person_loss:
				self.move_base_client.cancel_all_goals()
				self.person_loss = False
			self.cmdVelPublisher.publish(self.searching_velocity)
		elif info.data == 'Loss_target':
			if not self.person_loss:
				self.person_loss = True 
				rospy.logwarn(info.data)
				self.move_base_client.cancel_all_goals()
				self.send_move_base_goal()
		elif info.data == 'People_detected':
			rospy.logwarn(info.data)
			if self.person_loss:
				self.move_base_client.cancel_all_goals()
				self.person_loss = False

	def stopMoving(self):
		velocity = Twist()
		velocity.linear = Vector3(0.,0.,0.)
		velocity.angular= Vector3(0.,0.,0.)
		self.cmdVelPublisher.publish(velocity)


		
class simplePID:
	'''very simple discrete PID controller'''
	def __init__(self, target, P, I, D):
		'''Create a discrete PID controller
		each of the parameters may be a vector if they have the same length
		
		Args:
		target (double) -- the target value(s)
		P, I, D (double)-- the PID parameter

		'''

		# check if parameter shapes are compatabile. 
		if(not(np.size(P)==np.size(I)==np.size(D)) or ((np.size(target)==1) and np.size(P)!=1) or (np.size(target )!=1 and (np.size(P) != np.size(target) and (np.size(P) != 1)))):
			raise TypeError('input parameters shape is not compatable')
		rospy.loginfo('PID initialised with P:{}, I:{}, D:{}'.format(P,I,D))
		self.Kp		=np.array(P)
		self.Ki		=np.array(I)
		self.Kd		=np.array(D)
		self.setPoint   =np.array(target)
		
		self.last_error=0
		self.integrator = 0
		self.integrator_max = float('inf')
		self.timeOfLastCall = None 
		
		
	def update(self, current_value):
		'''Updates the PID controller. 

		Args:
			current_value (double): vector/number of same legth as the target given in the constructor

		Returns:
			controll signal (double): vector of same length as the target

		'''
		current_value=np.array(current_value)
		if(np.size(current_value) != np.size(self.setPoint)):
			raise TypeError('current_value and target do not have the same shape')
		if(self.timeOfLastCall is None):
			# the PID was called for the first time. we don't know the deltaT yet
			# no controll signal is applied
			self.timeOfLastCall = time.process_time()
			return np.zeros(np.size(current_value))

		
		error = self.setPoint - current_value
		P =  error
		
		currentTime = time.process_time()
		deltaT      = (currentTime-self.timeOfLastCall)

		# integral of the error is current error * time since last update
		self.integrator = self.integrator + (error*deltaT)
		I = self.integrator
		
		# derivative is difference in error / time since last update
		D = (error-self.last_error)/deltaT
		
		self.last_error = error
		self.timeOfLastCall = currentTime
		
		# return controll signal
		return self.Kp*P + self.Ki*I + self.Kd*D
		

if __name__ == '__main__':
	print('starting')
	rospy.init_node('follower')
	follower = Follower()
	try:
		rospy.spin()
	except rospy.ROSInterruptException:
		print('exception')