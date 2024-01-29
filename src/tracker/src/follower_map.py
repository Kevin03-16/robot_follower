#!/usr/bin/env python3

import rospy
import threading
import actionlib
import time
import numpy as np
from sensor_msgs.msg import LaserScan, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from dwa_planner import DWAPlanner
import tf
import tf2_geometry_msgs
from ros_tracker.msg import position as PositionMsg
from std_msgs.msg import String as StringMsg
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf2_ros
import math


class Follower:
	def __init__(self):
		# as soon as we stop receiving Joy messages from the ps3 controller we stop all movement:
		self.switchMode= rospy.get_param('~switchMode') # if this is set to False the O button has to be kept pressed in order for it to move
		self.max_speed = rospy.get_param('~maxSpeed') 
		self.max_gimbal_speed = rospy.get_param('~maxGimbalSpeed')
		self.controllButtonIndex = rospy.get_param('~controllButtonIndex')
		self.active=False
		self.cmdVelPublisher = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
		self.gimbalPublisher = rospy.Publisher("/joint_states", JointState, queue_size=10)
		self.positionSubscriber = rospy.Subscriber('/object_tracker/current_position', PositionMsg, self.positionUpdateCallback)
		self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
		self.trackerInfoSubscriber = rospy.Subscriber('/object_tracker/info', StringMsg, self.trackerInfoCallback)

		# PID parameters first is angular, dist
		self.targetDist = rospy.get_param('~targetDist')
		self.dwa = DWAPlanner(param=rospy.get_param('~dwa_param'))
		self.goal_pos = None

		# the first parameter is the angular target (0 degrees always) the second is the target distance (say 1 meter)
		self.Person_detection = False
		self.joint_state = JointState()
		self.joint_state.name = 'camera_joint'
		self.searching_velocity = Twist()	
		self.searching_velocity.linear = Vector3(0,0,0.)
		self.searching_velocity.angular= Vector3(0., 0.,1)
		self.min_distance_threshold = 1 # 设置最小距离阈值，低于该值时执行避障动作

		self.obstacles = np.zeros([0, 2])
		self.robot_x = 0.0
		self.robot_y = 0.0
		self.robot_yaw = 0.0
		self.tracking_velocity = Twist()	

		self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
		self.move_base_goal = MoveBaseGoal()
		self.move_base_client.wait_for_server()
		self.tf_buffer = tf2_ros.Buffer()
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
		self.target = PoseStamped()

	def odom_callback(self, data):
        # 从里程计信息中提取机器人当前位置和方向
		self.robot_x = data.pose.pose.position.x
		self.robot_y = data.pose.pose.position.y
		orientation_q = data.pose.pose.orientation
		_, _, self.robot_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

	def positionUpdateCallback(self, position):
		self.goal_pos = position
		self.target.header.frame_id = 'base_link'
		self.target.pose.position.x = position.distance 
		self.target.pose.position.y = (position.distance) * math.tan(-position.angleX)
		self.target.pose.position.z = 0
		q = quaternion_from_euler(0.0, 0.0, position.angleX)
		self.target.pose.orientation.x = q[0]
		self.target.pose.orientation.y = q[1]
		self.target.pose.orientation.z = q[2]
		self.target.pose.orientation.w = q[3]
		try:
			transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
			target_in_map = tf2_geometry_msgs.do_transform_pose(self.target, transform)
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
			rospy.logerr(f"Failed to transform the goal: {e}")
			exit()
		self.move_base_goal.target_pose.header.frame_id = 'map'
		self.move_base_goal.target_pose.pose = target_in_map.pose
		self.move_base_client.cancel_goal()
		self.move_base_client.send_goal(self.move_base_goal)
		self.move_base_client.wait_for_result()
	
	def trackerInfoCallback(self, info):
		if info.data == 'No_people':
			rospy.logwarn_once(info.data)
			self.Person_detection = False
			self.cmdVelPublisher.publish(self.searching_velocity)
		elif info.data == 'Loss_target':
			rospy.logwarn_once(info.data)
			self.active = False
			self.Person_detection = False
			self.goal_pos = self.predict_target_position(time_interval=2)
			self.positionUpdateCallback(self.goal_pos)
		elif info.data == 'People_detected':
			if not self.Person_detection:
				rospy.logwarn_once(info.data)
				self.Person_detection = True
				self.active = True
				self.stopMoving()

	def predict_target_position(self, time_interval):
		# 预测目标在未来的位置
		angleX = self.tracking_velocity.angular.z * time_interval
		distance = self.tracking_velocity.linear.x * time_interval
		predicted_position = PositionMsg(angleX, 0, 0, 0, distance)
		return predicted_position
		# we do not handle any info from the object tracker specifically at the moment. just ignore that we lost the object for example
		
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