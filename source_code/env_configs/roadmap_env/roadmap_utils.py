from shapely.geometry import Point
import folium
import os
import os.path as osp
import pandas as pd
import numpy as np


project_dir = osp.dirname(osp.dirname(osp.dirname(__file__)))

class Roadmap():

    def __init__(self, dataset, env_config):
        self.dataset = dataset
        self.env_config = env_config

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

    def init_pois(self, max_episode_step):
        '''读df并处理表头'''
        poi_df = pd.read_csv(osp.join(project_dir, f'envs/{self.dataset}/human{max_episode_step}.csv'))
        try:  # 如果df中有'pz'列, 删除它
            poi_df.drop('pz', axis=1, inplace=True)
        except:
            pass
        assert poi_df.columns.to_list()[-2:] == ['px', 'py']
        '''将df转换为np.array'''
        poi_mat = np.expand_dims(poi_df[poi_df['id'] == 0].values[:, -2:], axis=0)  # idt
        for id in range(1, self.env_config['poi_num']):
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

def traj_to_timestamped_geojson(index, trajectory, poi_QoS, uav_num, color,
                                input_args, env_config):

    point_gdf = trajectory.df.copy()
    features = []
    # for Point in GeoJSON type
    for i, (current_time, row) in enumerate(point_gdf.iterrows()):  # 遍历一个人的不同时间步
        # current_time每隔两个item数值才真的会变化，不影响目前的程序逻辑，不过可以看下数据集还有没有挖掘的潜力
        current_point_coordinates = [row["geometry"].xy[0][0], row["geometry"].xy[1][0]]
        ra = {'uav': 5, 'human': 1}
        op = {'uav': 1, 'human': 1}

        if index < uav_num:  # UAV
            radius, opacity = ra['uav'], op['uav']
        else:  # human
            if input_args['use_fixed_range']:  # case1 fixed QoS
                radius, opacity = (400 - input_args['snr']) / 100 + 1, op['human']
            else:  # case2 dyna QoS
                QoS = poi_QoS[index-uav_num][min(i, len(poi_QoS[1])-1)]  # 防止数组越界
                # 将QoS的100~400映射到4~1
                radius, opacity = (400 - QoS)/100 + 1, op['human']

        # for Point in GeoJSON type  (Temporally Deprecated)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": current_point_coordinates,
                },
                "properties": {
                    "times": [current_time.isoformat()],
                    "icon": 'circle',  # point
                    "iconstyle": {
                        'fillColor': color,
                        'fillOpacity': opacity,  #
                        'stroke': 'true',
                        'radius': radius,
                        'weight': 1,
                    },
                    "style": {  # line
                        "color": color,
                        "opacity": opacity
                    },
                    "code": 11,

                },
            }
        )
    return features


# 注意，这种画法圆的大小是和地图适配的
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

# 注意，这种画法圆的大小是和屏幕适配的
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


