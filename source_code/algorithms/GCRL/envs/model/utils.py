import numpy as np

np.seterr(invalid='ignore')

from algorithms.GCRL.configs.config import BaseEnvConfig
from algorithms.GCRL.envs.model.agent import *
from algorithms.GCRL.envs.model.mdp import JointState
from shapely.geometry import *

tmp_config = BaseEnvConfig()


def tensor_to_joint_state(state):  # 恢复原先尺度
    robot_states, human_states = state

    robot_states = robot_states.cpu().squeeze(0).data.numpy()
    robot_states = [RobotState(robot_state[0] * tmp_config.env.nlon,
                               robot_state[1] * tmp_config.env.nlat,
                               robot_state[2] * tmp_config.env.rotation_limit,
                               robot_state[3] * tmp_config.env.max_uav_energy) for robot_state in robot_states]

    human_states = human_states.cpu().squeeze(0).data.numpy()
    human_states = [HumanState(human_state[0] * tmp_config.env.nlon,
                               human_state[1] * tmp_config.env.nlat,
                               human_state[2] * tmp_config.env.rotation_limit,
                               human_state[3] * tmp_config.env.num_timestep) for human_state in human_states]

    return JointState(robot_states, human_states)


def tensor_to_robot_states(robot_state_tensor):
    robot_states = robot_state_tensor.cpu().squeeze(0).data.numpy()
    robot_states = [RobotState(robot_state[0] * tmp_config.env.nlon,
                               robot_state[1] * tmp_config.env.nlat,
                               robot_state[2] * tmp_config.env.rotation_limit,
                               robot_state[3] * tmp_config.env.max_uav_energy) for robot_state in robot_states]
    return robot_states


def get_human_position_list(selected_timestep, human_df):
    selected_timestamp = tmp_config.env.start_timestamp + selected_timestep * tmp_config.env.step_time
    selected_data = human_df[human_df.timestamp == selected_timestamp]
    selected_data = selected_data.set_index("id")

    if selected_timestep < tmp_config.env.num_timestep:
        selected_next_data = human_df[human_df.timestamp == (selected_timestamp + tmp_config.env.step_time)]
        selected_next_data = selected_next_data.set_index("id")
    else:
        selected_next_data = None

    return selected_data, selected_next_data


def get_human_position_from_list(selected_timestep, human_id, selected_data, selected_next_data):
    px, py = selected_data.loc[human_id, ["x", "y"]]

    if selected_timestep < tmp_config.env.num_timestep:
        npx, npy = selected_next_data.loc[human_id, ["x", "y"]]
        theta = get_theta(0, 0, npx - px, npy - py)
        # print(px, py, npx, npy, theta)
    else:
        theta = 0

    return px, py, theta


def judge_aoi_update(human_position, robot_position):
    '''
    根据pair(m位置,u位置)，判断当前时刻用户m的数据能否被无人机u成功接收
    其实就是二维欧式距离小于sensing_range~
    '''
    should_reset = False
    for robot_id in range(tmp_config.env.robot_num):
        unit_distance = np.sqrt(np.power(robot_position[robot_id][0] - human_position[0], 2)
                                + np.power(robot_position[robot_id][1] - human_position[1], 2))
        if unit_distance <= tmp_config.env.sensing_range:  # sensing_range唯一被读的地方
            should_reset = True
            break

    return should_reset


def inPoly(polygon, x, y):
    pt = (x, y)
    line = LineString(polygon)
    point = Point(pt)
    polygon = Polygon(line)
    return polygon.contains(point)


def iscrosses(line1, line2):
    if LineString(line1).crosses(LineString(line2)):
        return True
    return False


def crossPoly(square, x1, y1, x2, y2):
    our_line = LineString([[x1, y1], [x2, y2]])
    line1 = LineString([square[0], square[2]])
    line2 = LineString([square[1], square[3]])
    if our_line.crosses(line1) or our_line.crosses(line2):
        return True
    else:
        return False


def judge_collision(new_robot_px, new_robot_py, old_robot_px, old_robot_py):
    if tmp_config.env.no_fly_zone is None:
        return False

    for square in tmp_config.env.no_fly_zone:
        if inPoly(square, new_robot_px, new_robot_py):
            return True
        if crossPoly(square, new_robot_px, new_robot_py, old_robot_px, old_robot_py):
            return True
    return False


def get_theta(x1, y1, x2, y2):
    ang1 = np.arctan2(y1, x1)
    ang2 = np.arctan2(y2, x2)
    theta = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    return theta


def consume_uav_energy(fly_time, hover_time):
    tmp_config = BaseEnvConfig()

    # configs
    # Pu = 0.5  # the average transmitted power of each user, W,  e.g. mobile phone
    P0 = 79.8563  # blade profile power, W
    P1 = 88.6279  # derived power, W
    U_tips = 120  # tip speed of the rotor blade of the UAV,m/s
    v0 = 4.03  # the mean rotor induced velocity in the hovering state,m/s
    d0 = 0.6  # fuselage drag ratio
    rho = 1.225  # density of air,kg/m^3
    s0 = 0.05  # the rotor solidity
    A = 0.503  # the area of the rotor disk, m^2
    Vt = tmp_config.env.velocity  # velocity of the UAV,m/s ???

    Power_flying = P0 * (1 + 3 * Vt ** 2 / U_tips ** 2) + \
                   P1 * np.sqrt((np.sqrt(1 + Vt ** 4 / (4 * v0 ** 4)) - Vt ** 2 / (2 * v0 ** 2))) + \
                   0.5 * d0 * rho * s0 * A * Vt ** 3

    Power_hovering = P0 + P1

    return fly_time * Power_flying + hover_time * Power_hovering


def get_border(ur, lf):
    upper_left = [lf[0], ur[1]]
    upper_right = [ur[0], ur[1]]
    lower_right = [ur[0], lf[1]]
    lower_left = [lf[0], lf[1]]

    coordinates = [
        upper_left,
        upper_right,
        lower_right,
        lower_left,
        upper_left
    ]

    geo_json = {"type": "FeatureCollection",
                "properties": {
                    "lower_left": lower_left,
                    "upper_right": upper_right
                },
                "features": []}

    grid_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates],
        }
    }

    geo_json["features"].append(grid_feature)

    return geo_json


import numpy as np
import math


def rgb2hsi(rgb, di=None, seti=None):  # rgb=(255,0,0)  每个分量取值范围[0,255]
    B, G, R = rgb[0], rgb[1], rgb[2]
    # 归一化到[0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0

    H, S, I = B, G, R

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = float(np.arccos(num / den))
    if den == 0:
        H = 0
    elif B <= G:
        H = theta
    else:
        H = 2 * 3.14169265 - theta

    min_RGB = min(min(B, G), R)
    sum = B + G + R
    if sum == 0:
        S = 0
    else:
        S = 1 - 3 * min_RGB / sum

    H = H / (2 * 3.14159265)
    I = sum / 3.0
    if di is not None:
        I += di
        # I += 0.05
    if seti is not None:
        I = seti
        S = seti
    # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间 可删 只是为了调亮度
    # H = H * 255
    # S = S * 255
    # I = I * 255

    hsi = (H, S, I)
    return hsi


def hsi2rgb(hsi):

    H, S, I = hsi[0], hsi[1], hsi[2]
    # 可删
    # H = H / 255.0
    # S = S / 255.0
    # I = I / 255.0

    B, G, R = H, S, I

    if S < 1e-6:
        R = I
        G = I
        B = I
    else:
        H *= 360
        if H > 0 and H <= 120:
            B = I * (1 - S)
            R = I * (1 + (S * math.cos(H * math.pi / 180)) / math.cos(
                (60 - H) * math.pi / 180))
            G = 3 * I - (R + B)
        elif H > 120 and H <= 240:
            H = H - 120
            R = I * (1 - S)
            G = I * (1 + (S * math.cos(H * math.pi / 180)) / math.cos(
                (60 - H) * math.pi / 180))
            B = 3 * I - (R + G)
        elif H > 240 and H <= 360:
            H = H - 240
            G = I * (1 - S)
            B = I * (1 + (S * math.cos(H * math.pi / 180)) / math.cos(
                (60 - H) * math.pi / 180))
            R = 3 * I - (G + B)

    B = int(B * 255)
    G = int(G * 255)
    R = int(R * 255)

    # 强度调整后可能超出范围，clip到[0,255]
    B = 0 if B < 0 else B
    B = 255 if B > 255 else B
    G = 0 if G < 0 else G
    G = 255 if G > 255 else G
    R = 0 if R < 0 else R
    R = 255 if R > 255 else R

    bgr = (B, G, R)
    return bgr




def traj_to_timestamped_geojson(index, trajectory, robot_num, color_base, args):  # 画第index条traj，index是enum值
    point_gdf = trajectory.df.copy()
    point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
    point_gdf["time"] = point_gdf.index
    point_gdf["previous_time"] = point_gdf["time"].shift()


    features = []

    # for Point in GeoJSON type
    s = 0
    for _, row in point_gdf.iterrows():  # 一条轨迹中的第row个点
        corrent_point_coordinates = [
            row["geometry"].xy[0][0],
            row["geometry"].xy[1][0]
        ]
        current_time = [row["time"].isoformat()]

        # color渐变
        color = '#%02X%02X%02X' % color_base  # method1:强度不变
        # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=(s+1)/point_gdf.shape[0]))  # method2:强度从0到1 (有点浮夸)
        # color = '#%02X%02X%02X' % hsi2rgb(rgb2hsi(color_base, seti=0.333+(s+1)/(1.5*point_gdf.shape[0])))  # method3:强度从0.333到1

        if index < robot_num:  # 绘制无人机的圈圈
            radius = 12 if args.dataset == 'Purdue' else 8   # kaist8NCSU8purdue12
            opacity = 1
            radius = 20  # sensing range
            opacity = 0.5  # sensing range
            # opacity = 0.333 + (s + 1) / (1.5 * point_gdf.shape[0])  # 无人机点也渐变~
            popup_html = f'<h4>UAV {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>energy: {row["energy"]}J </p>'
        else:  # 绘制user，打点
            radius = 3 if args.dataset == 'Purdue' else 2  # kaist2NCSU2Purdue3
            opacity = 1
            # opacity = 0.333 + (s + 1) / (1.5 * point_gdf.shape[0])  # user点也渐变~
            popup_html = f'<h4>Human {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
                         + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
                         + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
                         + f'<p>aoi: {int(row["aoi"])} </p>'

        # for Point in GeoJSON type  (Temporally Deprecated)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": corrent_point_coordinates,
                },
                "properties": {
                    "times": current_time,
                    'tooltip': popup_html,
                    'popup': popup_html,  # 悬停时弹出的信息！包括user id，据此确定要删除的user
                    "icon": 'circle',  # circle
                    "iconstyle": {
                        'fillColor': color,
                        'fillOpacity': opacity,  # 透明度
                        'stroke': 'true',
                        'radius': radius,
                        'weight': 1,
                    },

                    "style": {  # line
                        "color": color,  # color在外部随机生成~
                    },
                    "code": 11,

                },
            }
        )

        s += 1
    return features

# def traj_to_timestamped_geojson2(index, trajectory, robot_num, color):  # 画第index条traj，index是enum值
#     point_gdf = trajectory.df.copy()
#     point_gdf["previous_geometry"] = point_gdf["geometry"].shift()
#     point_gdf["time"] = point_gdf.index
#     point_gdf["previous_time"] = point_gdf["time"].shift()
#
#     features = []
#
#     # for Point in GeoJSON type
#     last = [-78.680795, 35.777355]
#     for _, row in point_gdf.iterrows():
#
#         corrent_point_coordinates = [
#             row["geometry"].xy[0][0],
#             row["geometry"].xy[1][0]
#         ]
#         current_time = [row["time"].isoformat()]
#
#         if index < robot_num:  # 绘制无人机的圈圈
#             radius = 2  # 8
#             opacity = 0.05   # 透明度
#             popup_html = f'<h4>UAV {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
#                          + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
#                          + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
#                          + f'<p>energy: {row["energy"]}J </p>'
#         else:  # 绘制user，打点
#             radius = 2  # 2
#             opacity = 1  # 1
#             popup_html = f'<h4>Human {int(row["id"])}</h4>' + f'<p>raw coord: {corrent_point_coordinates}</p>' \
#                          + f'<p>grid coord: ({row["x"]},{row["y"]})</p>' \
#                          + f'<p>dist coord: ({row["x_distance"]}m, {row["y_distance"]}m)</p>' \
#                          + f'<p>aoi: {int(row["aoi"])} </p>'
#
#         # for Point in GeoJSON type  (Temporally Deprecated)
#
#         features.append(
#             {
#                 "type": "Feature",
#                 "geometry": {
#                     "type": "PolyLine",
#                     "coordinates": (last, corrent_point_coordinates),
#                 },
#                 "properties": {
#                     "times": current_time,
#                     'popup': popup_html,  # 悬停时弹出的信息！包括user id，据此确定要删除的user
#                     "icon": 'circle',  # point
#                     "iconstyle": {
#                         'fillColor': color,
#                         'fillOpacity': opacity,  # 透明度
#                         'stroke': 'true',
#                         'radius': radius,
#                         'weight': 1,
#                     },
#
#                     "style": {  # line
#                         "color": color,  # color在外部随机生成~
#                     },
#                     "code": 11,
#
#                 },
#             }
#         )
#
#         last = corrent_point_coordinates
#     return features

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

def visualize_matrix(A, file_name):
    if os.path.exists('logs/matrix') is False:
        os.makedirs('logs/matrix')
    save_dir = 'logs/matrix/%s.pdf' % file_name
    print(save_dir)
    pdf = PdfPages(save_dir)

    # create testing data which is 4x5 data
    mat = A
    print(mat)

    # Save Image Function
    plt.figure(figsize=(10,10))
    ax = plt.gca()
    cax=plt.imshow(mat, cmap='viridis', origin = 'lower')
    # set up colorbar
    cbar = plt.colorbar(cax)
    cbar.set_label('Intensity',size=36, weight =  'bold')
    # cbar.ax.tick_params( labelsize=18 )
    # cbar.minorticks_on()
    # set up axis labels
    # ticks=np.arange(0,mat.shape[0],1)
    # ## For x ticks
    # plt.xticks(ticks, fontsize=12, fontweight = 'bold')
    # ax.set_xticklabels(ticks)
    # ## For y ticks
    # plt.yticks(ticks, fontsize=12, fontweight = 'bold')
    # ax.set_yticklabels(ticks)

    pdf.savefig()
    plt.close()
    pdf.close()


if __name__=="__main__":
    plot_covergence(file_name="try")


