#!/usr/bin/env python3

import rospy
import threading
import actionlib
import time
import numpy as np
from sensor_msgs.msg import LaserScan, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3, PoseStamped, PointStamped, Point
from dwa_planner import DWAPlanner
from ros_tracker.msg import position as PositionMsg
from ros_tracker.msg import LastMoveAction, LastMoveResult
from std_msgs.msg import String as StringMsg
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math


class Follower:
	def __init__(self):
		# as soon as we stop receiving Joy messages from the ps3 controller we stop all movement:
		self.path_detection_thread = threading.Thread(target=self.path_detection)
		self.switchMode= rospy.get_param('~switchMode') # if this is set to False the O button has to be kept pressed in order for it to move
		self.max_speed = rospy.get_param('~maxSpeed') 
		self.max_gimbal_speed = rospy.get_param('~maxGimbalSpeed')
		self.controllButtonIndex = rospy.get_param('~controllButtonIndex')
		self.radius = rospy.get_param('~car_radius')
		self.active=False
		self.inflation = 4 * self.radius
		self.tf_listener = tf.TransformListener()
		self.pos_in_camera = PoseStamped()
		self.obs = PointStamped()
		self.tf_listener.waitForTransform("base_link", "base_scan", rospy.Time(), rospy.Duration(4.0))
		self.cmdVelPublisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.gimbalPublisher = rospy.Publisher("/joint_states", JointState, queue_size=10)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
		self.action_server = actionlib.SimpleActionServer('last_position', LastMoveAction, self.action_execute, False)
		self.action_server.start()

		self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
		self.positionSubscriber = rospy.Subscriber('/object_tracker/current_position', PositionMsg, self.positionUpdateCallback)
		self.trackerInfoSubscriber = rospy.Subscriber('/object_tracker/info', StringMsg, self.trackerInfoCallback)

		# PID parameters first is angular, dist
		self.targetDist = rospy.get_param('~targetDist')
		PID_param = rospy.get_param('~PID_controller')	
		self.dwa = DWAPlanner(param=rospy.get_param('~dwa_param'), radius=self.radius)
		self.goal_pos = None

		# the first parameter is the angular target (0 degrees always) the second is the target distance (say 1 meter)
		self.PID_controller = simplePID([0, self.targetDist], PID_param['P'], PID_param['I'], PID_param['D'])
		self.Person_detection = False
		self.joint_state = JointState()
		self.joint_state.name = 'camera_joint'
		self.start_detection = False
		self.searching_velocity = Twist()	
		self.searching_velocity.linear = Vector3(0,0,0.)
		self.searching_velocity.angular= Vector3(0., 0.,1)
		self.min_distance_threshold = 1 # 设置最小距离阈值，低于该值时执行避障动作

		self.robot_x = 0.0
		self.robot_y = 0.0
		self.robot_yaw = 0.0
		self.tracking_velocity = Twist()	
		self.obstacles = [] 
		self.path_datection_terminate = False
		self.path_detection_thread.start()
		rospy.on_shutdown(self.on_shutdown_callback)


	def on_shutdown_callback(self):
		if not self.path_datection_terminate:
			self.path_datection_terminate = True

	def action_execute(self, goal):
		cmd_vel = self.pid_control(goal.distance + self.targetDist + 1, 2 * goal.angleX)
		self.tracking_velocity.linear = Vector3(cmd_vel[0], 0, 0.0)
		self.tracking_velocity.angular= Vector3(0., 0.,cmd_vel[1])
		self.cmdVelPublisher.publish(self.tracking_velocity)

	def odom_callback(self, data): # frame_id odom
        # 从里程计信息中提取机器人当前位置和方向
		self.robot_x = data.pose.pose.position.x
		self.robot_y = data.pose.pose.position.y
		orientation_q = data.pose.pose.orientation
		_, _, self.robot_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])


	def scan_callback(self, scan_msg):# frame_id base_scan
		self.scan_data = np.array(scan_msg.ranges)
		self.obstacle_position_detection(scan_msg)


	def obstacle_position_detection(self, scan_msg):
		self.obstacles = []
		self.start_detection = False
		# 处理激光雷达的数据，提取障碍物信息
		if min(scan_msg.ranges) <= self.inflation:
			self.active = False
			self.stopMoving()
		else:
			for i, r in enumerate(scan_msg.ranges):
				if self.inflation < r < scan_msg.range_max:
					rospy.logwarn_once('start detecting')
					# 通过范围判断是否是障碍物
					# rospy.logwarn(r)
					angle = scan_msg.angle_min + i * scan_msg.angle_increment
					self.obs.header.stamp = rospy.Time.now()
					self.obs.header.frame_id = "base_scan"  # 相机坐标系
					self.obs.point = Point(x=(r - self.inflation) * math.cos(angle), y=(r - self.inflation) * math.sin(angle), z=0.0) 
					obs_in_base = self.tf_listener.transformPoint("base_link", self.obs)
					# rospy.logwarn(obs_in_base)
					self.obstacles.append([obs_in_base.point.x, obs_in_base.point.y, obs_in_base.point.z])
			self.start_detection = True


	def path_detection(self):
		while not self.path_datection_terminate:
			if self.start_detection:
				rospy.logwarn_once('detecting collision~~~~~')
				# X_cur = [self.robot_x, self.robot_y, self.robot_yaw, self.tracking_velocity.linear.x, self.tracking_velocity.angular.z]
				X_cur = [0, 0, self.robot_yaw, self.tracking_velocity.linear.x, self.tracking_velocity.angular.z]
				u = [self.tracking_velocity.linear.x, self.tracking_velocity.angular.z]
				traj_pred = self.dwa.Calculate_Traj(X_cur, u)
				# self.car_collision_object.setTransform(fcl.Transform())
				# for i in range(0, len(traj_pred)):
				# 	translation = np.array([traj_pred[i, 0], traj_pred[i,1], 0])
				# 	rotation = np.array(quaternion_from_euler(0, 0, self.tracking_velocity.angular.z))
					# pose_transform = fcl.Transform(rotation, translation)
					# self.car_collision_object.setTransform(pose_transform)
					# fcl.collide(self.car_collision_object, self.laser_collision_object, self.request, self.result)
					# 检测机器人路径
					# if self.result.is_collision:
				if self.path_collision(traj_pred):
					# 如果有障碍物，采用DWA算法避障
					cmd_vel = self.dwa.dwa_plan(X_cur, u, self.obstacles)
					self.tracking_velocity.linear = Vector3(cmd_vel[0],0,0.)
					self.tracking_velocity.angular= Vector3(0., 0.,cmd_vel[1])
					self.cmdVelPublisher.publish(self.tracking_velocity)
				time.sleep(1)  # 等待一秒钟再进行下一次检测

	def path_collision(self, traj_pred):
		minDist = self.dwa.Obstacle_Cost(traj_pred,self.obstacles)
		if minDist == float('Inf'):
			self.active = False
			return True
		else:
			self.active = True
			return False
	
	def trackerInfoCallback(self, info):
		if info.data == 'No_people':
			rospy.logwarn_once(info.data)
			self.Person_detection = False
			self.cmdVelPublisher.publish(self.searching_velocity)
		elif info.data == 'Loss_target':
			rospy.logwarn_once(info.data)
			self.Person_detection = False
		elif info.data == 'People_detected':
			if not self.Person_detection:
				rospy.logwarn_once(info.data)
				self.Person_detection = True
				self.active = True
				self.stopMoving()

	def predict_target_position(self, time_interval):
		# 预测目标在未来的位置
		angleX = self.goal_pos.angleX + self.tracking_velocity.angular.z * time_interval
		distance = self.goal_pos.distance + self.tracking_velocity.linear.x * time_interval
		self.goal_pos = PositionMsg(angleX, 0, 0, 0, distance)
		# we do not handle any info from the object tracker specifically at the moment. just ignore that we lost the object for example
	
	def pid_control(self, distance, angleX):
		rospy.loginfo('Angle: {}, Distance: {}, '.format(angleX, distance))
		
		# call the PID controller to update it and get new speeds
		[uncliped_ang_speed, uncliped_lin_speed] = self.PID_controller.update([angleX, distance])
		# [uncliped_y_ang_speed, _] = self.PID_controller.update([angleY, distance])
		
		# clip these speeds to be less then the maximal speed specified above
		angularSpeed = np.clip(-uncliped_ang_speed, -self.max_speed, self.max_speed)
		linearSpeed  = np.clip(-uncliped_lin_speed, -self.max_speed, self.max_speed)
		return [linearSpeed, angularSpeed]
		
	
	def positionUpdateCallback(self, position):# position is in camera_rgb_optical_frame
		self.goal_pos = position
		# gets called whenever we receive a new position. It will then update the motorcomand
		if(not(self.active)):
			return #if we are not active we will return imediatly without doing anything
		cmd_vel = self.pid_control(position.distance, position.angleX)
		self.tracking_velocity.linear = Vector3(cmd_vel[0],0,0.)
		self.tracking_velocity.angular= Vector3(0., 0.,cmd_vel[1])
		self.cmdVelPublisher.publish(self.tracking_velocity)
		self.joint_state.position = [position.angleY]
		self.gimbalPublisher.publish(self.joint_state)
		

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