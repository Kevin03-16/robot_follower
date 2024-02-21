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
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import tf
import math


class Follower:
	def __init__(self):
		# as soon as we stop receiving Joy messages from the ps3 controller we stop all movement:
		# self.predict_thread = threading.Thread(target=self.predict_path)
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
		self.Person_detection = False
		self.avoid_obs = False
		self.inflation = 4 * self.radius
		self.tf_buffer = tf2_ros.Buffer()
		tf_listener = tf2_ros.TransformListener(self.tf_buffer)
		self.tf_buffer_odom = tf2_ros.Buffer()
		tf_listener_2 = tf2_ros.TransformListener(self.tf_buffer_odom)

		self.cmdVelPublisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		# self.gimbalPublisher = rospy.Publisher("/joint_states", JointState, queue_size=10)
		self.positionSubscriber = rospy.Subscriber('/object_tracker/current_position', PositionMsg, self.positionUpdateCallback)
		self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
		self.trackerInfoSubscriber = rospy.Subscriber('/object_tracker/info', StringMsg, self.trackerInfoCallback)
		self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
		self.target_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=3)
		self.targetDist = rospy.get_param('~targetDist')
		self.dwa = DWAPlanner(param=rospy.get_param('~dwa_param'))

		# the first parameter is the angular target (0 degrees always) the second is the target distance (say 1 meter)
		# self.joint_state = JointState()
		# self.joint_state.name = 'camera_joint'
		self.searching_velocity = Twist()	
		self.searching_velocity.linear = Vector3(0,0,0.)
		self.searching_velocity.angular= Vector3(0., 0.,1)
		self.min_distance_threshold = 1 # 设置最小距离阈值，低于该值时执行避障动作

		self.tracking_velocity = Twist()
		PID_param = rospy.get_param('~PID_controller')	
		self.PID_controller = simplePID([0, self.targetDist], PID_param['P'], PID_param['I'], PID_param['D'])
		# 发布目标位置到move_base节点
		self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		self.move_base_goal = MoveBaseGoal()
		self.move_base_client.wait_for_server()
		# self.predict_thread.start()
	
	# def predict_path(self):
	# 	while not rospy.is_shutdown():
	# 		if self.person_loss:
	# 			self.send_move_base_goal(step=5)

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
		if min(scan_msg.ranges) <= self.inflation:
			self.active = False
			self.stopMoving()

	def send_move_base_goal(self):
		self.move_base_goal.target_pose.header.frame_id = 'map'
		self.move_base_goal.target_pose.header.stamp = rospy.Time.now()
		self.move_base_goal.target_pose.pose.orientation.w = 1
		self.move_base_goal.target_pose.pose.position.x = self.goal_pos.position.x + self.delta_x
		self.move_base_goal.target_pose.pose.position.y = self.goal_pos.position.y + self.delta_y
		self.move_base_goal.target_pose.pose.position.z = self.goal_pos.position.z
		self.move_base_client.send_goal(self.move_base_goal)
		# self.move_base_client.wait_for_result()
		

	def pid_control(self, distance, angleX):
		rospy.loginfo('Angle: {}, Distance: {}, '.format(angleX, distance))
		
		# call the PID controller to update it and get new speeds
		[uncliped_ang_speed, uncliped_lin_speed] = self.PID_controller.update([angleX, distance])
		# [uncliped_y_ang_speed, _] = self.PID_controller.update([angleY, distance])
		
		# clip these speeds to be less then the maximal speed specified above
		angularSpeed = np.clip(-uncliped_ang_speed, -self.max_speed, self.max_speed)
		linearSpeed  = np.clip(-uncliped_lin_speed, -self.max_speed, self.max_speed)
		return [linearSpeed, angularSpeed]

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
				self.move_base_goal.target_pose.header.frame_id='map'
				self.move_base_goal.target_pose.header.stamp=rospy.Time.now()
				self.move_base_goal.target_pose.pose.position = map_point.point
				self.move_base_goal.target_pose.pose.orientation.w = 1
				if self.goal_pos is not None:
					self.delta_x = self.move_base_goal.target_pose.pose.position.x - self.goal_pos.position.x
					self.delta_y = self.move_base_goal.target_pose.pose.position.y - self.goal_pos.position.y
				self.goal_pos = self.move_base_goal.target_pose.pose
			except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
				rospy.logerr(f"Failed to transform the goal: {e}")
				exit()
			if self.Person_detection:
				self.move_base_client.send_goal(self.move_base_goal)
		else:
			self.move_base_client.cancel_all_goals()
			cmd_vel = self.pid_control(position.distance, position.angleX)
			self.tracking_velocity.angular= Vector3(0., 0.,cmd_vel[1])
			self.cmdVelPublisher.publish(self.tracking_velocity)

	def trackerInfoCallback(self, info):
		if info.data == 'No_people':
			rospy.logwarn(info.data)
			self.Person_detection = False
			self.person_loss = False
			self.cmdVelPublisher.publish(self.searching_velocity)
		elif info.data == 'Loss_target':
			rospy.logwarn(info.data)
			self.person_loss = True
			self.Person_detection = False
			self.send_move_base_goal()
		elif info.data == 'People_detected':
			rospy.logwarn(info.data)
			self.Person_detection = True
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