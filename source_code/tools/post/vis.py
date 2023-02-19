# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import sys
import numpy as np
import os
import os.path as osp
import json
import folium
import pandas as pd
import geopandas as gpd
import movingpandas as mpd
from folium.plugins import TimestampedGeoJson
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # 不显示pandas的FutureWarning


assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code，否则无法正确导入包'
sys.path.append(os.getcwd())
from env_configs.roadmap_env.roadmap_utils import *


def render_HTML(output_dir, tag='train', draw_snrth=False, traj_filename='eps_best.npz'):

    '''从params.json中拿到训练时参数'''
    json_file = osp.join(output_dir, 'params.json')
    with open(json_file, 'r') as f:
        params = json.load(f)
        input_args = params['input_args']
        env_config = params['env_config']
    uav_num = env_config['uav_num']
    poi_num = env_config['poi_num']
    max_episode_step = env_config['max_episode_step']
    dataset = input_args['dataset']
    # poi_QoS = np.load(os.path.join(f'envs/{dataset}', f"QoS{max_episode_step}/poi_QoS{input_args['dyna_level']}.npy"))
    # assert poi_QoS.shape == (poi_num, max_episode_step)
    rm = Roadmap(dataset, env_config)

    '''从output_dir中拿到轨迹'''
    traj_file = osp.join(output_dir, f'{tag}_saved_trajs/{traj_filename}')
    trajs = np.load(traj_file)
    # poi_trajs, uav_trajs = list(trajs['arr_0']), list(trajs['arr_1'])
    poi_trajs = rm.init_pois(max_episode_step)
    uav_trajs = list(trajs['arr_0'])

    map = folium.Map(location=[(rm.lower_left[1] + rm.upper_right[1]) / 2, (rm.lower_left[0] + rm.upper_right[0]) / 2],
                     tiles="cartodbpositron", zoom_start=15, max_zoom=24)
    folium.TileLayer('Stamen Terrain').add_to(map)
    folium.TileLayer('Stamen Toner').add_to(map)
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('OpenStreetMap').add_to(map)

    grid_geo_json = get_border(rm.upper_right, rm.lower_left)
    color = "red"
    weight = 4 if 'NCSU' in output_dir else 2  # 2 by default
    dashArray = '10,10' if 'NCSU' in output_dir else '5,5'  # '5,5' by default
    border = folium.GeoJson(grid_geo_json,
                            style_function=lambda feature, color=color: {
                                'fillColor': color,
                                'color': "black",
                                'weight': weight,
                                'dashArray': dashArray,
                                'fillOpacity': 0,
                            })
    map.add_child(border)

    # 先将saved_trajs中的xy转换为lonlat，因为后续输入到folium时形式需要时lonlat
    for id in range(uav_num):
        for ts in range(len(uav_trajs[0])):
            uav_trajs[id][ts][:2] = rm.pygamexy2lonlat(*uav_trajs[id][ts][:2])
    for id in range(poi_num):
        for ts in range(len(poi_trajs[0])):
            poi_trajs[id][ts][:2] = rm.pygamexy2lonlat(*poi_trajs[id][ts][:2])

    uv_color_dict = {
        'uav1': '#%02X%02X%02X' % (255, 0, 0),  # red
        'uav2': '#%02X%02X%02X' % (3, 204, 51),  # green
        'uav3': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav4': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav5': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav6': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav7': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav8': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav9': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav10': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav11': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav12': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav13': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav14': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav15': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav16': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav17': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav18': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav19': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav20': '#%02X%02X%02X' % (0, 0, 255),  # blue
    }

    # fillin positions for uav, human
    mixed_df = pd.DataFrame()
    for id in range(uav_num + poi_num):
        df = pd.DataFrame(
            {'id': id,
             't': pd.date_range(start='20230315090000', end=None, periods=max_episode_step + 1, freq='15s'),
             }
        )
        if id < uav_num:  # uav
            df['longitude'], df['latitude'] = uav_trajs[id][:, 0], uav_trajs[id][:, 1]  # x=lat, y=lon
        else:  # human
            df['longitude'], df['latitude'] = poi_trajs[id - uav_num][:, 0], poi_trajs[id - uav_num][:, 1]
        mixed_df = mixed_df.append(df)

    # positions to traj
    mixed_gdf = gpd.GeoDataFrame(mixed_df, geometry=gpd.points_from_xy(mixed_df.longitude, mixed_df.latitude), crs=4326)
    mixed_gdf = mixed_gdf.set_index('t').tz_localize(None)
    trajs = mpd.TrajectoryCollection(mixed_gdf, 'id')

    def get_name_color_by_index(index):
        if index < uav_num:
            name = f"UAV {index}"
            color = uv_color_dict[list(uv_color_dict.keys())[index]]

        elif uav_num <= index:
            name = f"Human {index - uav_num}"
            color = '#%02X%02X%02X' % (0, 0, 0)
        else:
            raise ValueError
        return name, color

    for index, traj in enumerate(trajs.trajectories):
        name, color = get_name_color_by_index(index)
        if index<uav_num:
            uav_features = uav_traj_to_timestamped_geojson(index, traj, rm, uav_num, color,
                                                input_args, env_config, draw_snrth=False) 
            TimestampedGeoJson(  # 这里解注释了一个try except
                {
                    "type": "FeatureCollection",
                    "features": uav_features,
                },
                period="PT15S",
                add_last_point=True,
                transition_time=200,  # The duration in ms of a transition from between timestamps.
                max_speed=0.2,
                loop=True,
            ).add_to(map)
        else:
            features = traj_to_timestamped_geojson(index, traj, rm, uav_num, color,
                                                input_args, env_config, draw_snrth=False)  # True
            TimestampedGeoJson(  # 这里解注释了一个try except
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                period="PT15S",
                duration="PT15S",
                add_last_point=True,
                transition_time=200,  # The duration in ms of a transition from between timestamps.
                max_speed=0.2,
                loop=True,
            ).add_to(map)

        # line for uav

        # if index < uav_num:
        #     geo_col = traj.to_point_gdf().geometry
        #     for s in range(geo_col.shape[0] - 2):
        #         xy = [[y, x] for x, y in zip(geo_col.x[s:s + 2], geo_col.y[s:s + 2])]
        #         f1 = folium.FeatureGroup(name)
        #         folium.PolyLine(locations=xy, color=color, weight=4, opacity=0.7).add_to(f1)  # opacity=1 might more beautiful
        #         f1.add_to(map)

    folium.LayerControl().add_to(map)

    # save
    save_file = os.path.join(output_dir, f'vis_{tag}.html')
    map.get_root().save(save_file)

    print('OK!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--tag", type=str, default='train', choices=['train', 'test'], help='load trajs from train or test')
    parser.add_argument("--draw-snrth", action='store_true')
    args = parser.parse_args()

    render_HTML(
        args.output_dir,
        args.tag,
        args.draw_snrth,
    )

