import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc import depth_fill
import collections
import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import SetBool, SetBoolResponse


class RosWrapper:
    def __init__(self, config):
        self.config = config
        self.start = False
        self.start_motion = False
        self.nmpc_flag = False

        ## init node
        rospy.init_node('nmpc_colpred')

        ## message queues
        self.depth_queue = collections.deque(maxlen=1)
        self.depth_state_queue = collections.deque(maxlen=1)
        self.state_queue = collections.deque(maxlen=10)
        self.last_state = None

        ## topics
        topics = self.config.mission['ros_topics']
        self.depth_subscriber = rospy.Subscriber(topics['depth_input'], Image, self.depth_callback, queue_size=1)
        self.state_subscriber = rospy.Subscriber(topics['odom'], Odometry, self.state_callback, queue_size=1)
        self.cmd_vel_publisher = rospy.Publisher(topics['vel_cmd'], Twist, tcp_nodelay=False, queue_size=1)
        self.cmd_vel_stamped_publisher = rospy.Publisher(topics['vel_cmd_stamped'], TwistStamped, queue_size=1)
        self.cmd_traj_publisher = rospy.Publisher(topics['traj_horizon'], Path, queue_size=1)
        self.depth_publisher = rospy.Publisher(topics['depth_output'], Image, queue_size=1)

        ## service
        services = self.config.mission['ros_srv']
        rospy.Service(services['start'], SetBool, self.startstop_srv)
        rospy.Service(services['goto'], SetBool, self.goto_srv)
        rospy.Service(services['flag'], SetBool, self.colpred_srv)

    ## utils
    def msg_to_img(self, msg):
        if self.config.mission['simulation']:
            img = np.ndarray((msg.height, msg.width), np.float32, msg.data, 0)  # from Gazebo: raw data is float and in meter
        else:
            img = np.ndarray((msg.height, msg.width), np.uint16, msg.data, 0).astype(np.float32)/1000  # # from Gazebo: raw data is uint16 and in millimeter
        img = img/self.config.depth_max
        img = img.clip(0,1)
        if self.config.mission['use_hole_filling']:
            img = depth_fill.fill_in_fast(img, extrapolate=True)

        return img


    def img_to_msg(self, img):
        img_filtered_uint8 = (img * 255).astype('uint8')
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = '8UC1'
        msg.is_bigendian = 0
        msg.step = msg.width  # 1 bpp
        msg.data = np.reshape(img_filtered_uint8, (msg.height * msg.width,)).tolist()
        return msg


    ## interface
    def get_image(self):
        """If a new image has been retrieved, returns it (processed) along with the corresponding pose. Otherwise, returns None."""
        if len(self.depth_queue) and len(self.depth_state_queue):
            img = self.msg_to_img(self.depth_queue.popleft())
            pose = self.depth_state_queue.popleft().pose.pose
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            orientation = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            return img, position, orientation
        else:
            return None


    def get_state(self):
        """If a new state has been retrieved, returns it. Otherwise, returns None."""
        if len(self.state_queue):
            state = self.state_queue.popleft()
            pose = state.pose.pose
            vel = state.twist.twist.linear
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            orientation = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
            velocity = np.array([vel.x, vel.y, vel.z])
            return position, orientation, velocity
        else:
            return None


    def send_img(self, img):
        """Publish processed image to ROS message."""
        msg = self.img_to_msg(img)
        self.depth_publisher.publish(msg)


    def send_commands(self, commands):
        """Publish commands to ROS message."""
        ## send commands to controller
        msg = Twist()
        msg.linear.x = commands[0]
        msg.linear.y = 0.
        msg.linear.z = commands[1]
        msg.angular.x = 0.
        msg.angular.y = 0.
        msg.angular.z = commands[2]
        self.cmd_vel_publisher.publish(msg)

        ## send stamped cmds for rviz
        msg_stamped = TwistStamped()
        msg_stamped.header = Header(stamp=rospy.Time.now(), frame_id='state')
        msg_stamped.twist = msg
        self.cmd_vel_stamped_publisher.publish(msg_stamped)


    def send_path(self, path):
        """Publish commands to ROS message."""
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id='map')
        for point in path:
            pose = PoseStamped()
            # pose.header = msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.w = point[3]
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = point[4]
            msg.poses.append(pose)
        self.cmd_traj_publisher.publish(msg)


    ## callbacks
    def depth_callback(self, depth_image):
        if self.last_state is not None:
            self.depth_queue.appendleft(depth_image)
            self.depth_state_queue.appendleft(self.last_state)  # store latest state as the one the depth image was captured in

    def state_callback(self, state):
        self.state_queue.appendleft(state)
        self.last_state = state


    ## services
    def startstop_srv(self, bool):
        self.start = bool.data
        return SetBoolResponse(success=True, message='')

    def goto_srv(self, bool):
        self.start_motion = bool.data
        return SetBoolResponse(success=True, message='')

    def colpred_srv(self, bool):
        self.nmpc_flag = bool.data
        return SetBoolResponse(success=True, message='')
