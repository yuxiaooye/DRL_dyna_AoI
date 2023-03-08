# -*-coding:utf-8-*-
# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
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


def create_heatmap_from_saved(poi_trajs, aois, rm, input_args, output_dir, iter, best, paper):
    cell_num = 6  # hard-code
    if paper and input_args['dataset'] == 'KAIST':
        rm.max_dis_x = rm.max_dis_x * (5/6)  # 最南边的那一行热力图没意义。。

    cell_span_x = rm.max_dis_x / cell_num
    cell_span_y = rm.max_dis_y / cell_num

    str_iter = f'_{iter}' if iter is not None else ''
    str_best = '_best' if best else ''

    from envs.env_mobile import get_heatmap
    snrmap_dir = os.path.join(output_dir, f'./heatmap{str_iter}{str_best}/snr')
    txsatismap_dir = os.path.join(output_dir, f'./heatmap{str_iter}{str_best}/tx-satis')
    txsatismap2_dir = os.path.join(output_dir, f'./heatmap{str_iter}{str_best}/tx-satis2')
    aoimap_dir = os.path.join(output_dir, f'./heatmap{str_iter}{str_best}/aoi')
    aoisatismap_dir = os.path.join(output_dir, f'./heatmap{str_iter}{str_best}/aoi-satis')
    if not os.path.exists(snrmap_dir): os.makedirs(snrmap_dir)
    if not os.path.exists(txsatismap_dir): os.makedirs(txsatismap_dir)
    if not os.path.exists(txsatismap2_dir): os.makedirs(txsatismap2_dir)
    if not os.path.exists(aoimap_dir): os.makedirs(aoimap_dir)
    if not os.path.exists(aoisatismap_dir): os.makedirs(aoisatismap_dir)

    max_episode_step = input_args['max_episode_step']
    users_aoisatis = np.zeros(input_args['poi_num'])  # 维护每个user截至目前的aoi satis时间步数量
    for t in range(max_episode_step):  # 遍历时间一次性创建所有ts的heatmap
        count = np.zeros((cell_num, cell_num))
        snrmap = np.zeros((cell_num, cell_num))
        txsatismap = np.zeros((cell_num, cell_num))
        txsatismap2 = np.zeros((cell_num, cell_num))
        aoimap = np.zeros((cell_num, cell_num))
        aoisatismap = np.zeros((cell_num, cell_num))

        poi_positions = poi_trajs[:, t, :]
        for poi_index, poi_position in enumerate(poi_positions):
            if paper and input_args['dataset'] == 'KAIST' and poi_index in (100, 105, 114): continue
            if aois[t][poi_index] <= input_args['aoith']:
                users_aoisatis[poi_index] += 1

            x, y = poi_position
            i = np.clip(int(x / cell_span_x), 0, cell_num - 1)
            j = np.clip(int(y / cell_span_y), 0, cell_num - 1)
            count[i][j] += 1
            snrmap[i][j] += 1 if aois[t][poi_index] > input_args['aoith'] else aois[t][poi_index] / max_episode_step
            txsatismap[i][j] += 1 - aois[t][poi_index] / (t+1)  # 截至目前 user的data satis ratio
            txsatismap2[i][j] += (t - aois[t][poi_index]) / max_episode_step  # data collection ratio，当前已收集的数据量 / 总数据量
            aoisatismap[i][j] += users_aoisatis[poi_index] / (t+1)  # 截至目前 user的aoi satis ratio
            aoimap[i][j] += aois[t][poi_index]  # user的aoi

        # 对除snrmap外的map取mean
        txsatismap = np.where(count == 0, 1, txsatismap / count)  # 没有人的格子,置为1,有人的格子，取平均
        txsatismap2 = np.where(count == 0, 1, txsatismap2 / count)
        # aoisatismap = np.where(count == 0, 1, aoisatismap / count)
        #aoimap = np.where(count == 0, 0, aoimap / count)  # 没有人的格子置为0

        get_heatmap(snrmap, snrmap_dir + '/step_%03d' % (t) + '.png', min=0, max=5)
        get_heatmap(txsatismap, txsatismap_dir + '/step_%03d' % (t) + '.png', min=0, max=1)  # 用max是绝对错误的！不同时刻的最大刻度不统一怎么行
        get_heatmap(txsatismap2, txsatismap2_dir + '/step_%03d' % (t) + '.png', min=0, max=1)  # 用max是绝对错误的！不同时刻的最大刻度不统一怎么行
        get_heatmap(aoisatismap, aoisatismap_dir + '/step_%03d' % (t) + '.png', min=0, max=1)
        aoi_max = 100 if input_args['dataset'] == 'NCSU' else 200
        get_heatmap(aoimap, aoimap_dir + '/step_%03d' % (t) + '.png', min=0, max=aoi_max)  # # 用max是绝对错误的！不同时刻的最大刻度不统一怎么行
        print('creating heatmap... {}/{}'.format(t, max_episode_step))

def add_tiles(map):
    # folium.TileLayer('Stamen Terrain').add_to(map)
    # folium.TileLayer('Stamen Toner').add_to(map)
    # folium.TileLayer('cartodbpositron').add_to(map)
    # folium.TileLayer('OpenStreetMap').add_to(map)
    # new
    folium.TileLayer(  # OK!
        tiles='https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Tiles style by <a href="https://www.hotosm.org/" target="_blank">Humanitarian OpenStreetMap Team</a> hosted by <a href="https://openstreetmap.fr/" target="_blank">OpenStreetMap France</a>',
        name='yyx_try'
    ).add_to(map)
    # folium.TileLayer(
    #     tiles='https://{s}.tile.thunderforest.com/landscape/{z}/{x}/{y}.png?apikey=29acf6dd1ae244e2b78f7e01b3f5d1a9',
    #     attr='&copy; <a href="http://www.thunderforest.com/">Thunderforest</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    #     name='yyx_try_apikey'
    # ).add_to(map)




def plot_adj_histogram(uav_trajs, adjs, rm):

    T = 120
    assert len(uav_trajs[0]) - 1 == len(adjs) == T

    #  -- 没必要画出热力图。。。直接打印看下就好了 --
    # def get_heatmap(data, path, min, max, test=False):
    #     import seaborn as sns
    #     f, ax = plt.subplots(figsize=(6, 6))
    #     cmap = sns.diverging_palette(230, 20, as_cmap=True)
    #     cmap.set_under('lightgray')
    #     sns.heatmap(data, vmin=min, vmax=max, cmap=cmap, xticklabels=[], yticklabels=[])
    #     plt.savefig(path)
    # get_heatmap(np.stack(adjs), './try.png', min=0, max=120)

    # -- 查看距离与连接性之间的关系 --
    def dis(pos1, pos2):
        p1, p2 = rm.lonlat2pygamexy(*pos1), rm.lonlat2pygamexy(*pos2)
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**(1/2)

    data = []  # correlation between distance and connection
    for t, adj in enumerate(adjs):
        pass
        if adj[0][1] + adj[1][0] > 0:
            data.append(dis(uav_trajs[0][t], uav_trajs[1][t]))
    plt.hist(data, bins=20, rwidth=0.8)
    plt.show()

    # -- 查看每10个ts的连接性 --
    # for i in range(12):
    #     print(np.sum(np.stack(adjs)[i*10:i*10+10], axis=0))




def read_params(output_dir):
    json_file = osp.join(output_dir, 'params.json')
    with open(json_file, 'r') as f:
        params = json.load(f)
        input_args = params['input_args']
        env_config = params['env_config']
    return env_config, input_args


def render_HTML(output_dir, tag='train', iter=None, best=False, heatmap=False, history=False,
                paper=False, subgraph=False, best_count=None):
    assert (iter is not None and not best) or (iter is None and best)

    # 从params.json中拿到训练时参数
    env_config, input_args = read_params(output_dir)
    if paper and input_args['dataset'] == 'KAIST':
        input_args['aoith'] = 40

    uav_num = env_config['uav_num']
    poi_num = env_config['poi_num']
    max_episode_step = env_config['max_episode_step']
    dataset = input_args['dataset']
    # poi_QoS = np.load(os.path.join(f'envs/{dataset}', f"QoS{max_episode_step}/poi_QoS{input_args['dyna_level']}.npy"))
    # assert poi_QoS.shape == (poi_num, max_episode_step)
    rm = Roadmap(dataset, env_config)

    '''从output_dir中拿到轨迹'''
    if best:
        if best_count is not None:
            postfix = f'best{best_count}'
        else:
            postfix = 'best'
    else:
        postfix = str(iter)
    traj_file = osp.join(output_dir, f'{tag}_saved_trajs/eps_{postfix}.npz')
    aoi_file = osp.join(output_dir, f'{tag}_saved_trajs/eps_{postfix}_aoi.npz')
    serve_file = osp.join(output_dir, f'{tag}_saved_trajs/eps_{postfix}_serve.npz')
    adj_file = osp.join(output_dir, f'{tag}_saved_trajs/eps_{postfix}_adj.npz')

    uav_trajs, aois, serves = list(np.load(traj_file)['arr_0']), list(np.load(aoi_file)['arr_0']), list(np.load(serve_file)['arr_0'])
    adjs = list(np.load(adj_file)['arr_0']) if os.path.exists(adj_file) else None

    print('adj:', np.sum(adjs, axis=0))

    poi_trajs = rm.init_pois(max_episode_step)

    if heatmap:
        create_heatmap_from_saved(poi_trajs, aois, rm, input_args, output_dir, iter, best, paper)

    map = folium.Map(location=[(rm.lower_left[1] + rm.upper_right[1]) / 2, (rm.lower_left[0] + rm.upper_right[0]) / 2],
                     tiles="cartodbpositron", zoom_start=15, max_zoom=24)
    add_tiles(map)


    # # 定义一个变量，表示当前温度
    # current_temperature = 28
    #
    # # 在地图上添加一个标记，并将变量的值添加到弹出框中
    # folium.Marker(
    #     location=[rm.lower_left[1], rm.upper_right[0]],  # 右下角
    #     popup=f"当前温度为{current_temperature}℃",
    #     icon=folium.Icon()
    # ).add_to(map)

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
    if not paper:
        map.add_child(border)

    # 先将saved_trajs中的xy转换为lonlat，因为后续输入到folium时形式需要时lonlat
    for id in range(uav_num):
        for ts in range(len(uav_trajs[0])):
            uav_trajs[id][ts][:2] = rm.pygamexy2lonlat(*uav_trajs[id][ts][:2])  # TODO 把purple uav的轨迹往后错若干步，前面用0填充
    for id in range(poi_num):
        for ts in range(len(poi_trajs[0])):
            poi_trajs[id][ts][:2] = rm.pygamexy2lonlat(*poi_trajs[id][ts][:2])

    uv_color_dict = {
        # 'uav1': '#%02X%02X%02X' % (95, 158, 160),  # green
        # 'uav2': '#%02X%02X%02X' % (130, 43, 226),  # purple
        # 'uav3': '#%02X%02X%02X' % (0, 0, 255),  # blue

        'uav1': '#%02X%02X%02X' % (0, 128, 0),  # green
        'uav2': '#%02X%02X%02X' % (0, 0, 255),  # blue
        'uav3': '#%02X%02X%02X' % (130, 43, 226),  # purple
    }
    for i in range(4, 21):
        uv_color_dict[f'uav{i}'] = '#%02X%02X%02X' % (0, 0, 255),  # blue


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
        if index < uav_num:
            if subgraph:
                uav_features1 = subgraph_uav_traj_to_timestamped_geojson(index, traj, rm, uav_num, color,
                                                                input_args, env_config, subgraph)
                TimestampedGeoJson(
                    {
                        "type": "FeatureCollection",
                        "features": uav_features1,
                    },
                    period="PT15S",
                    # duration="PT15S",  # 使用时，拖尾；注释掉，保留
                    add_last_point=True,
                    transition_time=200,
                    max_speed=0.2,
                    loop=True,
                ).add_to(map)
            else:
                uav_features1, uav_features2 = uav_traj_to_timestamped_geojson(index, traj, rm, uav_num, color,
                                                               input_args, env_config)
                TimestampedGeoJson(
                    {
                        "type": "FeatureCollection",
                        "features": uav_features1,
                    },
                    period="PT15S",
                    # duration="PT15S",  # 使用时，拖尾；注释掉，保留
                    add_last_point=True,
                    transition_time=200,
                    max_speed=0.2,
                    loop=True,
                ).add_to(map)
                TimestampedGeoJson(
                    {
                        "type": "FeatureCollection",
                        "features": uav_features2,
                    },
                    period="PT15S",
                    duration="PT5S",  # PT15S是绘制最新三个点，PT0S大点的绘制会滞后于线段
                    add_last_point=True,
                    transition_time=200,
                    max_speed=0.2,
                    loop=True,
                ).add_to(map)

        else:
            if paper and input_args['dataset'] == 'KAIST' and index-uav_num in (100, 105, 114): continue  # 不展示边缘处的user
            features = traj_to_timestamped_geojson(index, traj, rm, uav_num, color,
                                                   input_args, env_config, aois, serves)

            TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                period="PT15S",
                duration=None if history else "PT15S",
                add_last_point=True,
                transition_time=200,  # The duration in ms of a transition from between timestamps.
                max_speed=0.2,
                loop=True,
            ).add_to(map)

    if adjs is not None:
        plot_adj_histogram(uav_trajs, adjs, rm)
        coor0 = [[rm.upper_right[0]+0.005, rm.upper_right[1]],
                [rm.upper_right[0]+0.01, rm.upper_right[1]]
                ]
        coor1 = [[rm.upper_right[0] + 0.01, rm.upper_right[1]],
                [rm.upper_right[0] + 0.0075, rm.upper_right[1]-0.0035]
                ]
        coor2 = [[rm.upper_right[0] + 0.0075, rm.upper_right[1]-0.0035],
                 [rm.upper_right[0] + 0.005, rm.upper_right[1]]
                ]
        adj_coords = [coor0, coor1, coor2]

        current_times = trajs.trajectories[0].df.copy().index  # 拿来用用
        featuress = adj_to_timestamped_geojson(adjs, current_times, adj_coords, uav_trajs, input_args, env_config)

        for features in featuress:
            TimestampedGeoJson(
                {
                    "type": "FeatureCollection",
                    "features": features,
                },
                period="PT15S",
                duration="PT15S",  # 动态变化
                add_last_point=True,
                transition_time=200,
                max_speed=0.2,
                loop=True,
            ).add_to(map)

        folium_draw_CircleMarker(map, [coor0[0][1], coor0[0][0]], color=uv_color_dict['uav1'], radius=20)
        folium_draw_CircleMarker(map, [coor1[0][1], coor1[0][0]], color=uv_color_dict['uav2'], radius=20)
        folium_draw_CircleMarker(map, [coor2[0][1], coor2[0][0]], color=uv_color_dict['uav3'], radius=20)

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
    str_iter = f'_{iter}' if iter is not None else ''
    str_best = f'_best' if best else ''
    str_paper = '_paper' if paper else ''
    str_history = '_history' if history else ''
    str_subgraph = '_subgraph' if subgraph else ''
    save_file = os.path.join(output_dir, f'vis_{tag}{str_iter}{str_best}{str_history}{str_paper}{str_subgraph}.html')
    map.get_root().save(save_file)

    if best_count is not None:  # 保存所有比之前更好的轨迹
        str_best = f'_best{best_count}'
        save_file = os.path.join(output_dir, f'vis_{tag}{str_iter}{str_best}{str_history}{str_paper}{str_subgraph}.html')

    map.get_root().save(save_file)

    print('vis OK!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--tag", type=str, default='train', choices=['train', 'test'], help='load trajs from train or test')
    parser.add_argument("--paper", action='store_true')
    parser.add_argument("--subgraph", action='store_true')
    parser.add_argument("--iter", type=int, default=-1)
    parser.add_argument("--best_count", type=int)
    parser.add_argument("--history", action='store_true')
    parser.add_argument("--heatmap", action='store_true')
    args = parser.parse_args()

    if args.iter == -1:
        best = True
        iter = None
    else:
        best = False
        iter = args.iter


    render_HTML(
        args.output_dir,
        tag=args.tag,
        best=best,
        iter=iter,
        heatmap=args.heatmap,
        history=args.history,
        paper=args.paper,
        subgraph=args.subgraph,
    )
