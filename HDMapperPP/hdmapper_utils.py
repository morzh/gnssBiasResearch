import numpy as np
# import quaternion
from geopy import distance
import pandas as pd
import sophuspy as sp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymap3d as pm


def bbox3D(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def calc_trajectory_length_segments(points):
    segments_num = points.shape[1]
    length_segments = np.zeros(segments_num)
    diffs = points[:, 1:] - points[:, :-1]
    abs_diffs = np.linalg.norm(diffs, axis=0)
    for i in range(1, segments_num):
        length_segments[i] = length_segments[i-1] + abs_diffs[i-1]
    length_debug = np.sum(abs_diffs)
    # length = length_segments[-1]
    return length_segments

def pc_normalize_by_length(points):
    shift = np.min(points, axis=1).reshape(3, 1)
    points -= shift
    diffs = points[:, 1:] - points[:, :-1]
    abs_diffs = np.linalg.norm(diffs, axis=0)
    scale = np.sum(abs_diffs)
    return points / scale, shift, scale

def pc_normalize(pc, axis='xy'):
    # pc_norms = np.linalg.norm(pc, axis=0)
    # pc_min_idx = np.argmin(pc_norms)
    # pc_max_idx = np.argmax(pc_norms)
    shift = np.min(pc, axis=1).reshape(3, 1)
    pc -= shift
    if axis == 'xy':
        pc_max = np.max(pc[0:2, :], axis=1).reshape(2, 1)
    elif axis == 'xz':
        pc_max = np.max(np.vstack((pc[0, :], pc[2, :])), axis=1)
    elif axis == 'zx':
        pc_max = np.max(np.vstack((pc[2, :], pc[0, :])), axis=1)
    elif axis == 'xyz':
        pc_max = np.max(pc, axis=1).reshape(3, 1)
    scale = max(pc_max)
    pc /= scale
    return pc, shift, scale

def showPointClouds(pc_1, pc_2, clamp=True, show_info=False):
    pc_1, shift_1, scale_1 = pc_normalize(pc_1, axis='xyz')
    pc_2, shift_2, scale_2 = pc_normalize(pc_2, axis='xyz')
    # pc_1, shift_1, scale_1 = pc_normalize_by_length(pc_1)
    # pc_2, shift_2, scale_2 = pc_normalize_by_length(pc_2)

    if show_info:
        print('point cloud 1 shift, scale:', shift_1.T, scale_1)
        print('point cloud 2 shift, scale:', shift_2.T, scale_2)

    if clamp:
        pc_1 = np.clip(pc_1, 0.0, 1.0)
        pc_2 = np.clip(pc_2, 0.0, 1.0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(pc_1[0], pc_1[1], pc_1[2])
    ax.plot(pc_2[0], pc_2[1], pc_2[2])
    plt.show()

    return pc_1, pc_2, shift_1, shift_2, scale_1, scale_2

def remove_duplicate_columns(array):
    num_rows = array.shape[0]
    new_array = np.empty((num_rows, 0))
    new_array = np.hstack((new_array, array[:, 0].reshape(num_rows, -1)))
    duplicate_indices_array = []
    for i in range(1, array.shape[1]):
        if np.linalg.norm(array[:, i-1] - array[:, i]) < 1e-8:
            duplicate_indices_array.append(i)
            continue
        new_array = np.hstack((new_array, array[:, i].reshape(num_rows, -1)))

    return new_array, duplicate_indices_array

def get_gps_translations(data):
    array = data.lat.to_numpy()
    array = np.vstack((array, data.lon.to_numpy()))
    array = np.vstack((array, data.altitude.to_numpy()))
    return array

def get_gps_accuracy(data):
    array = data.accuracy.to_numpy()
    array = np.vstack((array, array, data.altitude_accuracy.to_numpy()))
    return array

def read_tokens(f):
   for line in f:
       for token in line.split():
           yield token


class Track:
    def __init__(self):
        self.frames = list()

    def pushFrame(self, frame):
        self.frames.append(frame)

    def framesNumber(self):
        return len(self.frames)

    def findClosestToTimestamp(self, timestamp):
        time_min_delta = np.inf
        closest_frame = None

        for frame in self.frames:
            time_delta = np.abs(frame.timestamp - timestamp)
            if time_delta < 1e-18:
                return frame, time_delta
            elif time_delta < time_min_delta:
                time_min_delta = time_delta
                closest_frame = frame

        return closest_frame, time_min_delta



class AssociatedTrack(Track):
    def __init__(self):
        Track.__init__(self)
        self.odometry_connections = dict()

    def associateOdometryGPSTracks(self, gps_track, odometry_track, kMaxDiffTime):
        max_poses_to_associate = min(gps_track.framesNumber(), odometry_track.framesNumber())

        track_primary = None
        track_secondary = None

        if gps_track.framesNumber() == max_poses_to_associate:
            track_primary = gps_track
            track_secondary = odometry_track
        elif odometry_track.framesNumber() == max_poses_to_associate:
            track_primary = odometry_track
            track_secondary = gps_track
        else:
            print('something went wrong')


        for frame in track_primary.frames:
            timestamp = frame.timestamp
            frame_secondary, time_delta = track_secondary.findClosestToTimestamp(timestamp)
            if time_delta < kMaxDiffTime:
                self.pushFrame(AssociatedFrame(frame, frame_secondary))


    def getConnectionsIDsDict(self, source_frames_number):
        max_id = source_frames_number
        connections_dict = dict()
        for source_frame_id in range(source_frames_number):
            if source_frame_id >= self.framesNumber():
                continue
            source_frame, s_idx = self.findOdometryFrameWithId(self.frames[source_frame_id].odometry_frame.id)
            if source_frame is None:
                continue
            connections_to = []
            for target_id in self.odometry_connections[source_frame.id]:
                target_frame, t_idx = self.findOdometryFrameWithId(target_id)
                if target_frame is None:
                    continue
                connections_to.append(t_idx)

            connections_dict[s_idx] = connections_to

        return connections_dict



    def getNumberOfConnections(self, source_frames_number):
        connections_num = 0
        frames_number = self.framesNumber()
        for source_frame_id in range(source_frames_number):
            source_frame,_ = self.findOdometryFrameWithId(source_frame_id)
            if source_frame is None:
                continue
            connections_num += 1
            for target_id in self.odometry_connections[source_frame.id]:
                target_frame, _ = self.findOdometryFrameWithId(target_id)
                if target_frame is None:
                    continue
                connections_num += 1

        return connections_num


    def odometryFrameExists(self, ref_id):
        for frame in self.frames:
            if frame.odometry_frame.id == ref_id:
                return True
        return False

    def timestampsStatistics(self):
        timestamps_diffs_list = []
        for frame in self.frames:
            timestamps_diff = abs(frame.gps_frame.timestamp - frame.odometry_frame.timestamp)
            timestamps_diffs_list.append(timestamps_diff)

        timestamps_diffs = np.array(timestamps_diffs_list)
        print('frames number:', len(self.frames))
        print('timestamp diff mean:', np.mean(timestamps_diffs))
        print('timestamp diff deviation:', np.std(timestamps_diffs))

    def getOdometryTranslates(self):
        frames_num = self.framesNumber()
        translates = np.zeros((3, frames_num))
        for frame, idx in zip(self.frames, range(frames_num)):
            translates[:, idx] = frame.odometry_frame.t_w_c.translation()#.reshape(3, 1)
        return translates

    def getOdometryOptimizedTranslates(self):
        frames_num = self.framesNumber()
        translates = np.zeros((3, frames_num))
        for frame, idx in zip(self.frames, range(frames_num)):
            translates[:, idx] = frame.odometry_frame.optimized_t_w_c.translation()#.reshape(3, 1)
        return translates

    def getOdometryOptimizedTranslates(self):
        frames_num = self.framesNumber()
        translates = np.zeros((3, frames_num))
        for frame, idx in zip(self.frames, range(frames_num)):
            translates[:, idx] = frame.odometry_frame.optimized_t_w_c.translation()#.reshape(3, 1)
        return translates

    def getGPSMetersTranslates(self):
        frames_num = self.framesNumber()
        translates = np.zeros((3, frames_num))
        for frame, idx in zip(self.frames, range(frames_num)):
            translates[0, idx] = frame.gps_frame.latitude_meters
            translates[1, idx] = frame.gps_frame.longtitude_meters
            translates[2, idx] = frame.gps_frame.altitude

        return translates

    def getGPSAccuracyStacked(self):
        accuracy = np.ones((3, self.framesNumber()))
        for idx in range(self.framesNumber()):
            accuracy[0, idx] = self.frames[idx].gps_frame.tangent_accuracy
            accuracy[1, idx] = self.frames[idx].gps_frame.tangent_accuracy
            accuracy[2, idx] = self.frames[idx].gps_frame.altitude_accuracy
        return accuracy

    def getENUTranslates(self):
        frames_num = self.framesNumber()
        translates = np.zeros((3, frames_num))
        for frame, idx in zip(self.frames, range(frames_num)):
            translates[0, idx] = frame.gps_frame.ENU_coords[0]
            translates[1, idx] = frame.gps_frame.ENU_coords[1]
            translates[2, idx] = frame.gps_frame.ENU_coords[2]

        return translates

    def setOdometryOptimizedTransform(self, sR, t):
        for frame in self.frames:
            # frame.odometry_frame.optimized_t_w_c = sp.SIM3()
            frame.odometry_frame.optimized_t_w_c.setRotationMatrix(sR)
            frame.odometry_frame.optimized_t_w_c.setTranslation(t)

    def updateOdometryOptimizedTransform(self, s, R, t):
        transform = sp.SIM3()
        transform.setRotationMatrix(R)
        transform.setScale(s)
        transform.setTranslation(t)
        '''
        print("-------------------------------------------------------------")
        print(s*R)
        print(t)
        print(transform.matrix())
        print("-------------------------------------------------------------")
        '''
        for frame in self.frames:
            '''
            T = np.eye(4)
            T[0:3, 0:4] = np.hstack((s*R, t))
            pose = frame.odometry_frame.optimized_t_w_c.matrix()
            newT = T @ pose
            '''
            frame.odometry_frame.optimized_t_w_c = transform * frame.odometry_frame.optimized_t_w_c

    def updateENUTranslates(self, R_normlaizer):
        for frame in self.frames:
            frame.gps_frame.ENU_coords = R_normlaizer @ frame.gps_frame.ENU_coords

    def visualizeOdometryFrameConnections(self):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.gca(projection='3d')
        constraints_items = self.odometry_connections.items()
        number_of_colors = self.framesNumber()
        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]
        for constraint in constraints_items:
            source_frame_id = constraint[0]
            source_frame, _ = self.findOdometryFrameWithId(source_frame_id)
            if source_frame is None:
                continue
            source_translate = source_frame.optimized_t_w_c.translation().reshape(3, -1)
            for target_id in constraint[1]:
                target_frame, _ = self.findOdometryFrameWithId(target_id)
                if target_frame is None:
                    continue
                target_translate = target_frame.optimized_t_w_c.translation().reshape(3, -1)
                link_coordinates = np.hstack((source_translate, target_translate))
                ax.plot(link_coordinates[0], link_coordinates[1], link_coordinates[2])#, colors[source_frame_id])
        odometry_translates = self.getOdometryOptimizedTranslates()
        ax.plot(odometry_translates[0], odometry_translates[1], odometry_translates[2])
        plt.show()

    def findOdometryFrameWithId(self, id):
        for frame_idx in range(len(self.frames)):
            if self.frames[frame_idx].odometry_frame.id == id:
                return self.frames[frame_idx].odometry_frame, frame_idx
        return None, -1

class OdometryTrack(Track):
    def __init__(self):
        Track.__init__(self)
        self.active_constraints = dict()

    def pushConstraint(self, key_frame_iter, connection):
        self.active_constraints[key_frame_iter] = connection

    def getActiveConstraintsSIM3(self):
        active_constraints_sim3 = dict()
        for key in self.active_constraints.keys():
            frame_from = self.frames[key]
            frames_to = self.active_constraints[key]
            frames_to_sim3 = dict()
            for frame_idx in frames_to:
                transform_sim3 = sp.SIM3()
                transform_se3 = frame_from.t_w_c.inverse() * self.frames[frame_idx].t_w_c
                transform_sim3.setParameters(transform_se3.getParameters())
                frames_to_sim3[frame_idx] = transform_sim3
            active_constraints_sim3[key] = frames_to_sim3

        return active_constraints_sim3

    def getActiveConstraintsSIM3__(self):
        active_constraints_sim3 = dict()
        for key in self.active_constraints.keys():
            frame_from = self.frames[key]
            frames_to = self.active_constraints[key]
            frames_to_sim3 = dict()
            for frame_idx in frames_to:
                transform_sim3 = sp.SIM3()
                transform_se3 = frame_from.t_w_c.inverse() * self.frames[frame_idx].t_w_c
                transform_sim3.setParameters(transform_se3.getParameters())
                frames_to_sim3[frame_idx] = transform_sim3
            active_constraints_sim3[key] = frames_to_sim3

        return active_constraints_sim3


    def getActiveConstraintsSE3(self):
        active_constraints_se3 = dict()
        for key in self.active_constraints.keys():
            frame_from = self.frames[key]
            frames_to = self.active_constraints[key]
            frames_to_se3 = []
            for frame_idx in frames_to:
                frames_to_se3.append(frame_from.t_w_c.inverse() * self.frames[frame_idx].t_w_c)
            active_constraints_se3[key] = frames_to_se3

        return active_constraints_se3


class GPSTrack(Track):
    def __init__(self):
        Track.__init__(self)

    def convertLatLonToMeters(self):
        earth_radius_km = 6371.0
        for frame in self.frames:
            frame.latitude_meters = 1e3 * np.deg2rad(frame.latitude) * earth_radius_km
            frame.longtitude_meters = 1e3 * np.deg2rad(frame.longtitude) * earth_radius_km


    def convertToENU(self):
        zero_point = self.frames[0]
        for frame in self.frames:
            enu_ccords = pm.enu.geodetic2enu(frame.longtitude, frame.latitude, frame.altitude, zero_point.longtitude, zero_point.latitude, zero_point.altitude)
            frame.ENU_coords = np.array(enu_ccords)




class FeaturePoint:
    def __init__(self, position, color, rel_bs, segmentation_label):
        self.position = position
        self.color = color
        self.rel_bs = rel_bs
        self.segmentation_label = segmentation_label


class Frame:
    def __init__(self, timestamp, id):
        self.timestamp = timestamp
        self.id = id

class GPSFrame(Frame):
    def __init__(self, timestamp, latitude, longtitude, tangent_accuracy, altitude, altitude_accuracy, id):
        Frame.__init__(self, timestamp, id)
        # self.timestamp = timestamp
        self.latitude = latitude
        self.longtitude = longtitude
        self.tangent_accuracy = tangent_accuracy
        self.altitude = altitude
        self.altitude_accuracy = altitude_accuracy
        self.latitude_meters = None
        self.longtitude_meters = None
        self.ENU_coords = np.zeros((3, 1))

class OdometryFrame(Frame):
    def __init__(self, timestamp, odometry_t_w_c, id):
        Frame.__init__(self, timestamp, id)
        # self.timestamp = timestamp
        self.t_w_c = odometry_t_w_c
        self.optimized_t_w_c = sp.SIM3(odometry_t_w_c.matrix())
        self.elevation = None
        self.cumulative_distance = None
        self.feature_points = list()

    def pushPoint(self, point):
        self.feature_points.append(point)

class AssociatedFrame:
    def __init__(self, frame_1, frame_2):
        if isinstance(frame_1, GPSFrame) and isinstance(frame_2, OdometryFrame):
            self.gps_frame = frame_1
            self.odometry_frame = frame_2
        elif isinstance(frame_2, GPSFrame) and isinstance(frame_1, OdometryFrame):
            self.gps_frame = frame_2
            self.odometry_frame = frame_1


class TrackLoader:
    @staticmethod
    def parse_gps(gps_filename):
        gps_track = GPSTrack()
        gps_data = pd.read_csv(gps_filename)
        for idx in range(len(gps_data)):
            timestamp = gps_data['time'][idx]
            latitude = gps_data['lat'][idx]
            longtitude = gps_data['lon'][idx]
            tangent_accuracy = gps_data['accuracy'][idx]
            altitude = gps_data['altitude'][idx]
            altitude_accuracy = gps_data['altitude_accuracy'][idx]
            gps_track.pushFrame(GPSFrame(timestamp, latitude, longtitude, tangent_accuracy, altitude, altitude_accuracy, idx))
        return gps_track

    @staticmethod
    def parse_odometry(filename):
        odometry_track = OdometryTrack()
        connections_num = dict() #contains how many frames connected to  particular frame

        file = open(filename, 'r')
        tokens = read_tokens(file)
        # has_segmentation = bool(int(next(tokens)))
        num_key_frames = int(next(tokens))

        for key_frame_iter in range(num_key_frames):
            connections_num[key_frame_iter] = 0

        for key_frame_iter in range(num_key_frames):
            timestamp = np.float64(next(tokens))
            num_points = int(next(tokens))
            t = np.array([next(tokens), next(tokens), next(tokens)], dtype=np.float64)
            # T_w_c.setTranslation(t)
            q = np.array([next(tokens), next(tokens), next(tokens), next(tokens)], dtype=np.float64)
            r = R.from_quat([q[0], q[1], q[2], q[3]])
            # q = np.quaternion(q[0], q[1], q[2], q[3])
            T_w_c = sp.SE3(r.as_matrix(), t)


            frame = OdometryFrame(timestamp, T_w_c, key_frame_iter)

            num_connections = int(next(tokens))
            connections_list = list()
            for connection_iter in range(num_connections):
                connection = int(next(tokens))
                connections_list.append(connection)
                connections_num[connection] += 1
            odometry_track.pushConstraint(key_frame_iter, connections_list)


            for point_iter in range(num_points):
                position = np.array([next(tokens), next(tokens), next(tokens)], dtype=np.float64)
                color = np.array([next(tokens), next(tokens), next(tokens)], dtype=np.uint8) #in HDMapper color elements are oermuted: color[2], color[1], color[0]
                rel_bs = np.float64(next(tokens))
                segmentation_label = -1

                # if has_segmentation:
                #     segmentation_label = next(tokens)

                point = FeaturePoint(position, color, rel_bs, segmentation_label)
                frame.pushPoint(point)

            odometry_track.pushFrame(frame)

        return odometry_track
    '''
    def getFramesTranslates(self):
        translates = np.empty((3, 0))

        for frame in self.odometry_track.frames:
            translates = np.hstack((translates, frame.odometry_t_w_c.translation().reshape(3, 1)))

        return translates

    def findClosestByTimestamp(self, timestamp):
        time_delta = np.inf
        return_frame = None
        for frame in self.odometry_track.frames:
            time_diff = np.abs(frame.timestamp - timestamp)
            if time_diff < time_delta:
                time_delta = time_diff
                return_frame = frame

        return return_frame, time_delta
    '''

def save_houdini_csv(P, path2save):
    import pandas as pd
    df = pd.DataFrame(P.T)
    df.columns = ['P[x]', 'P[y]', 'P[z]']
    df.to_csv(path2save, index=False)


def save_houdini_accuracy_csv(P, accuracy, path2save):
    accuracy_lonlat_max = np.max(accuracy[0])
    accuracy_altitude_max = np.max(accuracy[2])

    accuracy[0] /= accuracy_lonlat_max
    accuracy[1] /= accuracy_lonlat_max
    accuracy[2] /= accuracy_altitude_max

    df = pd.DataFrame(np.hstack((P.T, accuracy.T)))
    df.columns = ['P[x]', 'P[y]', 'P[z]', 'Cd[r]', 'Cd[g]', 'Cd[b]']
    df.to_csv(path2save, index=False)

def convert_lat_lon_to_km(gps_poses):
    gps_poses_km = lat_lon_to_meters(gps_poses[:2, :])
    gps_poses_km = np.vstack((gps_poses_km, gps_poses[2]))
    return gps_poses_km

def linear_interpolate_poses(poses, number_poses, show_plot=False):
    interpolated_poses = poses[:, 0].reshape(3, 1)
    length_segments = calc_trajectory_length_segments(poses)
    length = length_segments[-1]
    rrr = np.linspace(0, length_segments.shape[0]-1, length_segments.shape[0])
    plt.plot(rrr, length_segments)
    plt.show()
    for idx in range(1, number_poses):
        length_fraction = idx / (number_poses - 1) * length
        cond_array = np.where(length_segments >= length_fraction)
        idx_max = np.min(cond_array)
        idx_min = max(idx_max-1, 0)
        alpha = (length_fraction - length_segments[idx_min]) / (length_segments[idx_max] - length_segments[idx_min])
        interpolated_poses = np.hstack((interpolated_poses,  ((1 - alpha) * poses[:, idx_min] + alpha * poses[:, idx_max]).reshape(3, 1)))

        if show_plot:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(interpolated_poses[0], interpolated_poses[1], interpolated_poses[2])
            plt.show()

    return interpolated_poses


def lat_lon_to_meters(latlon_array):
    earth_radius_km = 6371.0
    km = np.deg2rad(latlon_array) * earth_radius_km
    return 1e3 * km



def add_piecewise_bias_to_gps_data(gps_data):
    pass

import random

def viz_frames_connections(odometry_track):
    fig = plt.figure(figsize=(20, 10))
    ax = fig.gca(projection='3d')
    # ax.axis('equal')
    constraints_items = odometry_track.active_constraints.items()
    number_of_colors = odometry_track.framesNumber()
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])  for i in range(number_of_colors)]
    for constraint in constraints_items:
        source_frame_idx = constraint[0]
        for idx in constraint[1]:
            source_t = odometry_track.frames[source_frame_idx].t_w_c.translation().reshape(3, -1)
            linked_t = odometry_track.frames[idx].t_w_c.translation().reshape(3, -1)
            current_coordinates = np.hstack((source_t, linked_t))
            ax.plot(current_coordinates[0], current_coordinates[1], current_coordinates[2], colors[source_frame_idx])

    plt.show()
