from shapely.geometry import Point
import folium
import os
import os.path as osp
import pandas as pd
import numpy as np

project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))


class Roadmap():

    def __init__(self, dataset, env_config=None, poi_num=None):
        assert env_config is not None or poi_num is not None
        self.dataset = dataset
        self.env_config = env_config
        self.poi_num = poi_num

        self.map_props = get_map_props()
        self.lower_left = get_map_props()[dataset]['lower_left']
        self.upper_right = get_map_props()[dataset]['upper_right']

        try:  # movingpandas
            from movingpandas.geometry_utils import measure_distance_geodesic
            self.max_dis_y = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                       Point(self.upper_right[0], self.lower_left[1]))
            self.max_dis_x = measure_distance_geodesic(Point(self.lower_left[0], self.lower_left[1]),
                                                       Point(self.lower_left[0], self.upper_right[1]))
        except:
            # hardcode
            if dataset == 'purdue':
                self.max_dis_y = 1671.8995666382975
                self.max_dis_x = 1221.4710883988212
            elif dataset == 'NCSU':
                self.max_dis_y = 3255.4913305859623
                self.max_dis_x = 2718.3945272795013
            elif dataset == 'KAIST':
                self.max_dis_y = 2100.207579392558
                self.max_dis_x = 2174.930950809533
            else:
                raise NotImplementedError

    def init_pois(self, max_episode_step=120):
        '''读df并处理表头'''
        poi_num = self.env_config['poi_num'] if self.env_config is not None else self.poi_num
        if self.dataset == 'NCSU' and poi_num == 33 or self.dataset == 'KAIST' and poi_num == 92:
            postfix = ''
        else:
            postfix = f'-user{poi_num}'
        csv_name = f'envs/{self.dataset}/human{max_episode_step}{postfix}.csv'
        poi_df = pd.read_csv(osp.join(project_dir, csv_name))

        assert poi_df.columns.to_list()[-2:] == ['px', 'py']
        '''将df转换为np.array'''
        poi_mat = np.expand_dims(poi_df[poi_df['id'] == 0].values[:, -2:], axis=0)  # idt

        for id in range(1, poi_num):
            subDf = poi_df[poi_df['id'] == id]
            poi_mat = np.concatenate((poi_mat, np.expand_dims(subDf.values[:, -2:], axis=0)), axis=0)
        return poi_mat  # shape = (33, 121, 2) 意为33个poi在121个时间步的坐标信息

    def lonlat2pygamexy(self, lon, lat):
        # 计算机坐标系，以左上角为远点，向下为x轴正方向，向右为y轴正方向
        x = - self.max_dis_x * (lat - self.upper_right[1]) / (self.upper_right[1] - self.lower_left[1])
        y = self.max_dis_y * (lon - self.lower_left[0]) / (self.upper_right[0] - self.lower_left[0])
        return x, y

    def pygamexy2lonlat(self, x, y):
        lon = y * (self.upper_right[0] - self.lower_left[0]) / self.max_dis_y + self.lower_left[0]
        lat = - x * (self.upper_right[1] - self.lower_left[1]) / self.max_dis_x + self.upper_right[1]
        return lon, lat


def get_map_props():
    map_props = {
        'NCSU':
            {
                'lower_left': [-78.6988, 35.7651],  # lon, lat
                'upper_right': [-78.6628, 35.7896]
            },
        'purdue':
            {
                'lower_left': [-86.93, 40.4203],
                'upper_right': [-86.9103, 40.4313]
            },
        'KAIST':
            {
                'lower_left': [127.3475, 36.3597],
                'upper_right': [127.3709, 36.3793]
            }
    }
    return map_props


def traj_to_timestamped_geojson(index, trajectory, rm, uav_num, color,
                                input_args, env_config, aois=None, serves=None):
    point_gdf = trajectory.df.copy()
    features = []
    # for Point in GeoJSON type
    for i, (current_time, row) in enumerate(point_gdf.iterrows()):  # 遍历一个人的不同时间步

        if index < uav_num:  # UAV
            radius, opacity = 5, 1
        else:  # human
            radius, opacity = 2, 1

        if aois is not None:
            radius = aois[i][index-uav_num] / 12 + 3  # aoi越大radius越大 1~120映射到3~13
        if serves is not None:
            if serves[i][index-uav_num]:
                color = "red"
            else:
                color = "black"

        # for Point in GeoJSON type
        cur_coord = [row["geometry"].xy[0][0], row["geometry"].xy[1][0]]

        feature1 = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": cur_coord,
            },
            "properties": {
                "times": [current_time.isoformat()],
                "icon": 'circle',  # point
                "iconstyle": {
                    'fillColor': color,
                    'fillOpacity': opacity,
                    'stroke': 'true',
                    'radius': radius,
                    'weight': 1,
                },
                "style": {  # 外边框的属性
                    "color": color,
                    "opacity": opacity
                },
                "code": 11,
                'popup': f"User{index-uav_num}\n"
                         f"AoI={aois[i][index-uav_num]}",
            },
        }
        features.append(feature1)


    return features


def uav_traj_to_timestamped_geojson(index, trajectory, rm, uav_num, color,
                                    input_args, env_config, poi_QoS=None):
    point_gdf = trajectory.df.copy()
    features1, features2 = [], []
    # for Point in GeoJSON type
    last_x, last_y, last_time = None, None, None
    for i, (current_time, row) in enumerate(point_gdf.iterrows()):  # 遍历一个人的不同时间步

        if last_x is None:
            last_x = row["geometry"].xy[0][0]
            last_y = row["geometry"].xy[1][0]
            last_time = current_time.isoformat()
        new_x = row["geometry"].xy[0][0]
        new_y = row["geometry"].xy[1][0]

        if index < uav_num:  # UAV
            radius, opacity = 5, 1
        else:  # human
            radius, opacity = 2, 1

        # for Point in GeoJSON type
        cur_coord = [[last_x, last_y], [new_x, new_y]]

        feature1 = {  # 轨迹的线
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": cur_coord,
            },
            "properties": {
                "times": [last_time, current_time.isoformat()],
                "icon": 'circle',  # point
                "iconstyle": {
                    'fillColor': color,
                    'fillOpacity': opacity,
                    'stroke': 'true',
                    'radius': radius,
                    'weight': 1,
                },
                "style": {  # 外边框的属性
                    "color": color,
                    "opacity": opacity
                },
                "code": 11,
            },
        }
        feature2 = {  # 机头的大点
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": cur_coord[1],
            },
            "properties": {
                "times": [last_time, current_time.isoformat()],
                "icon": 'circle',  # point
                "iconstyle": {
                    'fillColor': color,
                    'fillOpacity': opacity,
                    'stroke': 'true',
                    'radius': 8,
                    'weight': 1,
                },
                "style": {  # 外边框的属性
                    "color": color,
                    "opacity": opacity
                },
                "code": 11,
            },
        }
        features1.append(feature1)
        features2.append(feature2)
        last_x = new_x
        last_y = new_y
        last_time = current_time.isoformat()


    return features1, features2


# 注意，这种画法圆的大小是和**地图**适配的
def folium_draw_circle(map, pos, color, radius, weight):  #
    folium.vector_layers.Circle(
        location=pos,  #
        radius=radius,  # m
        color=color,  #
        # fill=True,  #
        # fill_color='#%02X%02X%02X' % (0, 0, 0),  #
        # fillOpacity=1,  # Fill opacity
        weight=weight  #
    ).add_to(map)


def create_circle(feature):
    return {
        'type': 'Circle',
        'location': feature['geometry']['coordinates'],
        'radius': 200,
        'color': 'red'
    }


# 注意，这种画法圆的大小是和**屏幕**适配的
def folium_draw_CircleMarker(map, pos, color, radius):  #
    folium.CircleMarker(
        location=pos,
        radius=radius,
        color=color,
        stroke=False,
        fill=True,
        fill_opacity=1,
        opacity=1,
        popup="{} ".format(radius),
        tooltip=str(pos),
    ).add_to(map)


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
