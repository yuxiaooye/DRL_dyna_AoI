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
            if dataset == 'NCSU':
                self.max_dis_y = 3255.4913305859623
                self.max_dis_x = 2718.3945272795013
            elif dataset == 'purdue':
                self.max_dis_y = 1671.8995666382975
                self.max_dis_x = 1221.4710883988212
            else: raise NotImplementedError

    def init_pois(self, max_episode_step=120):
        '''读df并处理表头'''
        poi_df = pd.read_csv(osp.join(project_dir, f'envs/{self.dataset}/human{max_episode_step}.csv'))
        try:  # 如果df中有'pz'列, 删除它
            poi_df.drop('pz', axis=1, inplace=True)
        except:
            pass
        assert poi_df.columns.to_list()[-2:] == ['px', 'py']
        '''将df转换为np.array'''
        poi_mat = np.expand_dims(poi_df[poi_df['id'] == 0].values[:, -2:], axis=0)  # idt

        poi_num = self.env_config['poi_num'] if self.env_config is not None else self.poi_num
        for id in range(1, poi_num):
            subDf = poi_df[poi_df['id'] == id]
            poi_mat = np.concatenate((poi_mat, np.expand_dims(subDf.values[:, -2:], axis=0)), axis=0)
        return poi_mat  # shape = (33, 121, 2) 意为33个poi在121个时间步的坐标信息

    def lonlat2pygamexy(self, lon, lat):
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

def traj_to_timestamped_geojson(index, trajectory, rm, poi_QoS, uav_num, color,
                                input_args, env_config, draw_snrth=False):

    point_gdf = trajectory.df.copy()
    features = []
    # for Point in GeoJSON type
    last_vis_coord = rm.lower_left
    for i, (current_time, row) in enumerate(point_gdf.iterrows()):  # 遍历一个人的不同时间步

        if index < uav_num:  # UAV
            radius, opacity = 5, 1
        else:  # human
            if input_args['use_fixed_range']:  # case1 fixed QoS
                raise NotImplementedError
            else:  # case2 dyna QoS
                # QoS = poi_QoS[index-uav_num][min(i, len(poi_QoS[1])-1)]  # 防止数组越界
                # radius, opacity = (400 - QoS)/100 + 1, op['human']  # 将QoS的100~400映射到4~1
                radius, opacity = 2, 1

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
                },
            }
        features.append(feature1)

        # 每个时间步都画SNRth的话图上太乱了，隔五步画一次
        # 人没怎么动还一直画太黑了，仅当人移动超过一定距离再画新的
        def is_draw(cur_coord, last_vis_coord):
            pos1 = rm.lonlat2pygamexy(*cur_coord)
            pos2 = rm.lonlat2pygamexy(*last_vis_coord)
            dis = np.linalg.norm(np.array(pos1)-np.array(pos2), ord=2)
            return dis > 500
        if draw_snrth and index > uav_num and \
                i % 5 == 0 and is_draw(cur_coord, last_vis_coord):
            feature2 = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": cur_coord,
                        # "coordinates": [-78.6988, 35.7651],  # debug 左下角坐标
                    },
                    "properties": {
                        "times": [current_time.isoformat()],
                        "icon": 'circle',  # point
                        "iconstyle": {   #
                            'fillColor': color,
                            'fillOpacity': 0.1,  # 这个改成1就不透明~
                            'stroke': 'true',
                            'radius': poi_QoS[index-uav_num][min(i, poi_QoS.shape[1]-1)] * 840 / 3255,  # 在zoom等级为15时，840等价于3255m
                            'weight': 1,
                        },
                        "style": {  # 外边框的属性
                            "color": color,
                            "opacity": 0.0,
                        },
                        "code": 11,
                    },
                }
            features.append(feature2)
            last_vis_coord = cur_coord
    return features


# 注意，这种画法圆的大小是和**地图**适配的
def folium_draw_circle(map, pos, color, radius, weight):  #
    folium.vector_layers.Circle(
        location=pos,  #
        radius=radius,  #  m
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


