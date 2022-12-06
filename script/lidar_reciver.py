#!/usr/bin/env python
from functools import partial
import rospy
import pcl
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistStamped

open_utf8 = partial(open, encoding='UTF-8')


class LidarListener:
    def __init__(self):

        self.sub_blue = rospy.Subscriber("/ground_segmentation/obstacle_cloud",
                                         PointCloud2, self.lidar_callback,
                                         queue_size=1, buff_size=2**24)
        self.pub_marker = rospy.Publisher("/cones_clusters", MarkerArray,
                                          queue_size=20)

        # get data from params
        self.min_dist = rospy.get_param("perception/cones_postion_constraints/min_dist")
        self.max_dist = rospy.get_param("perception/cones_postion_constraints/max_dist")
        self.z_treshold = rospy.get_param("perception/cones_postion_constraints/z_treshold")
        self.angle_threshold = rospy.get_param("perception/cones_postion_constraints/angle_threshold")

        self.min_cluster_size = rospy.get_param("perception/clustering/min_cluster_size")
        self.max_cluster_size = rospy.get_param("perception/clustering/max_cluster_size")
        self.cluster_tolerance = rospy.get_param("perception/clustering/cluster_tolerance")

        self.logging = rospy.get_param("perception/logs/logging")
        self.log_path = rospy.get_param("perception/logs/log_path")

        if self.logging:
            rospy.Subscriber("/fsds/gss", TwistStamped, self.gss_callback)
            rospy.Subscriber("/fsds/imu", Imu, self.imu_callback)

            self.write_logs = open_utf8(self.log_path, "w")

    def gss_callback(self, message):
        self.velocity_x = message.twist.linear.x
        self.velocity_y = message.twist.linear.y

    def imu_callback(self, message):
        self.angular_velocity_z = message.angular_velocity.z

    def lidar_callback(self, message):
        """ Callback function for the lidar subscriber.
        Transforms the lidar data (PonitCloud2) into a PointCloud_PointXYZI.
        Clusters the data and publishes the clusters centers as a
        visualization_msgs.msg MarkerArray.
        """
        rospy.loginfo("Start lidar callback")
        start_time = rospy.Time.now()

        points_xyz = self.ros_to_pcl(message, self.min_dist,
                                     self.max_dist, self.z_treshold,
                                     self.angle_threshold)

        cluster_indices = self.euclid_cluster(points_xyz,
                                                   self.min_cluster_size,
                                                   self.max_cluster_size,
                                                   self.cluster_tolerance)

        # visualization
        marker_array = MarkerArray()

        cones_list_s = ""

        for j, indices in enumerate(cluster_indices):

            sum_pos = np.zeros(3)

            for i, indice in enumerate(indices):
                # cloud reconstruction
                sum_pos += points_xyz[indice]

            avg_pos = sum_pos / len(indices)

            # visualization and initalize quaternions
            marker = Marker()
            marker.header.frame_id = "fsds/Lidar"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "basic_cloud_cluster"
            marker.id = j
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = avg_pos[0]
            marker.pose.position.y = avg_pos[1]
            marker.pose.position.z = avg_pos[2]
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.4
            marker.color.r = 0
            marker.color.g = 0
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.1)
            marker.frame_locked = True
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            cones_list_s += ('[' + str(avg_pos[0]) + ',' +
                             str(avg_pos[1]) + '],')

            marker_array.markers.append(marker)

        # chech if marker_array is empty
        if len(marker_array.markers) > 0:
            self.pub_marker.publish(marker_array)

        rospy.loginfo("Number of cones detected: %s",
                      len(marker_array.markers))
                      
        rospy.loginfo("End lidar callback in %s ms",
                      (rospy.Time.now() - start_time).to_sec() * 1000)

        # save data
        if self.logging:
            self.save_cluster2txt(cones_list_s)

    def euclid_cluster(self, cloud, min_size, max_size, tolerance):
        """ Performs euclidian clustering on a given cloud.
        Args:
            cloud: PointCloud_PointXYZI
            tolerance: float
            min_size: int
            max_size: int
        Returns:
            cloud_clusters: list of PointCloud_PointXYZI
        """
        tree = cloud.make_kdtree()
        ec = cloud.make_EuclideanClusterExtraction()
        ec.set_ClusterTolerance(tolerance)
        ec.set_MinClusterSize(min_size)
        ec.set_MaxClusterSize(max_size)
        ec.set_SearchMethod(tree)
        cluster_indices = ec.Extract()

        return cluster_indices

    def ros_to_pcl(self, ros_cloud, min_dist, max_dist, angle_threshold,
                   z_treshold):
        """ Converts a ROS PointCloud2 message to a pcl PointXYZ.
        Filters a given cloud by removing points with a distance < min_dist,
        angle > angle_threshold and points with a distance > max_dist.

            Args:
                ros_cloud (PointCloud2): ROS PointCloud2 message
                min_dist: float
                max_dist: float
                angle_threshold: float

            Returns:
                pcl.PointCloud_PointXYZ
        """
        points_list = []

        for data in point_cloud2.read_points(ros_cloud, skip_nans=True):
            distance = np.linalg.norm(data[:3])
            angle = np.arctan2(data[1], data[0])

            if (distance > min_dist and distance < max_dist and
                    angle < angle_threshold and data[2] < z_treshold):
                points_list.append([data[0], data[1], data[2]])

        pcl_data = pcl.PointCloud()
        pcl_data.from_list(points_list)

        return pcl_data

    def save_cluster2txt(self, cones_list_s):
        '''
        Saves cluster_list to cones.txt in folder ./data
        '''
        self.write_logs.write(str(self.velocity_x) + ',')
        self.write_logs.write(str(self.velocity_y) + ',')
        self.write_logs.write(str(self.angular_velocity_z) + ',')
        self.write_logs.write(cones_list_s + '\n')


if __name__ == '__main__':
    rospy.init_node('lidar_reciver')
    lidar_listener = LidarListener()
    rospy.spin()
