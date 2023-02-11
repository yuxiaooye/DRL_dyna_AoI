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
assert os.getcwd().endswith('source_code'), '请将工作路径设为source_code'
sys.path.append(os.getcwd())
from env_configs.roadmap_env.roadmap_utils import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir")
parser.add_argument("--tag", type=str, default='train', choices=['train', 'test'], help='load trajs from train or test')
parser.add_argument("--traj_filename", type=str, default='eps_best.npz', help='load which .npz file')
parser.add_argument("--draw_uav_lines", default=True, action='store_false')
parser.add_argument("--diff_color", default=True, action='store_false')

args = parser.parse_args()
args.group_save_dir = args.output_dir

def get_arg_postfix(args):
    arg_postfix = ''
    if args.draw_uav_lines:
        arg_postfix += '_drawUavLines'
    return arg_postfix


def main(args):
    '''从args.output_dir中拿到轨迹'''
    traj_file = osp.join(args.output_dir, f'{args.tag}_saved_trajs/{args.traj_filename}')
    trajs = np.load(traj_file)
    poi_trajs, uav_trajs = list(trajs['arr_0']), list(trajs['arr_1'])  # 当人也在动时，这里也需要读poi_trajs

    '''从params.json中拿到训练时参数'''
    json_file = osp.join(args.output_dir, 'params.json')
    with open(json_file, 'r') as f:
        params = json.load(f)
        input_args = params['input_args']
        env_config = params['env_config']
    uav_num = env_config['uav_num']
    poi_num = env_config['poi_num']
    num_timestep = env_config['max_episode_step']
    dataset = input_args['dataset']
    poi_QoS = np.load(os.path.join(f'envs/{dataset}', f"poi_QoS{input_args['dyna_level']}.npy"))
    assert poi_QoS.shape == (poi_num, num_timestep)
    rm = Roadmap(dataset)

    map = folium.Map(location=[(rm.lower_left[1] + rm.upper_right[1]) / 2, (rm.lower_left[0] + rm.upper_right[0]) / 2],
                     tiles="cartodbpositron", zoom_start=14, max_zoom=24)
    folium.TileLayer('Stamen Terrain').add_to(map)
    folium.TileLayer('Stamen Toner').add_to(map)
    folium.TileLayer('cartodbpositron').add_to(map)
    folium.TileLayer('OpenStreetMap').add_to(map)

    grid_geo_json = get_border(rm.upper_right, rm.lower_left)
    color = "red"
    weight = 4 if 'NCSU' in args.output_dir else 2  # 2 by default
    dashArray = '10,10' if 'NCSU' in args.output_dir else '5,5'  # '5,5' by default
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
    }

    # fillin positions for uav, human
    mixed_df = pd.DataFrame()
    for id in range(uav_num + poi_num):
        df = pd.DataFrame(
            {'id': id,
             't': pd.date_range(start='20230315090000', end=None, periods=num_timestep+1, freq='15s'),
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
            if args.diff_color:
                color = uv_color_dict[list(uv_color_dict.keys())[index]]
            else:
                color = '#%02X%02X%02X' % (255, 0, 0)
        elif uav_num <= index:
            name = f"Human {index - uav_num}"
            color = '#%02X%02X%02X' % (0, 0, 0)
        else:
            raise ValueError
        return name, color

    for index, traj in enumerate(trajs.trajectories):
        name, color = get_name_color_by_index(index)
        # TODO 这里是否读poi_QoS，要看input_args中的--use_fixed_range开关
        features = traj_to_timestamped_geojson(index, traj, poi_QoS, uav_num, color,
                                               input_args, env_config)
        try:
            TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                period="PT15S",
                add_last_point=True,
                transition_time=200,  # The duration in ms of a transition from between timestamps.
                max_speed=0.2,
                loop=True,
            ).add_to(map)
        except:
            pass

        # line for uav
        if args.draw_uav_lines and index < uav_num:
            geo_col = traj.to_point_gdf().geometry
            for s in range(geo_col.shape[0] - 2):
                xy = [[y, x] for x, y in zip(geo_col.x[s:s + 2], geo_col.y[s:s + 2])]
                f1 = folium.FeatureGroup(name)
                folium.PolyLine(locations=xy, color=color, weight=4, opacity=0.7).add_to(f1)  # opacity=1 might more beautiful
                f1.add_to(map)

    folium.LayerControl().add_to(map)

    # save
    arg_postfix = get_arg_postfix(args)
    if args.group_save_dir is None:
        vsave_dir = args.output_dir + f'/gif/{args.tag}_{args.traj_filename}'
        if not osp.exists(vsave_dir): os.makedirs(vsave_dir)
        map.get_root().save(vsave_dir + f'/{args.traj_filename[:-4]}{arg_postfix}.html')
    else:
        if not osp.exists(args.group_save_dir): os.makedirs(args.group_save_dir)
        save_file = os.path.join(args.group_save_dir, f'{arg_postfix}.html')
        print('------args.group_save_dir = ', args.group_save_dir)
        print("------f'{arg_postfix}.html' = ", f'{arg_postfix}.html')
        print('------save_file = ', save_file)
        map.get_root().save(save_file)

    print('OK!')



if __name__ == '__main__':
    main(args)
