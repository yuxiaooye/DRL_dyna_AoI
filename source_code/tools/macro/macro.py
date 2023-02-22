# metrics
METRICS = ['QoI', 'episodic_aoi', 'aoi_satis_ratio', 'data_satis_ratio', 'energy_consuming']

# hyper tuning
HT_INDEX1 = ['EoiCoef=0.001', 'EoiCoef=0.003', 'EoiCoef=0.01', 'EoiCoef=0.03']
HT_INDEX2 = ['w/o SL, w/o CC', 'w/ SL, w/o CC', 'w/o SL, w/ CC', 'w/ SL, w/ CC']

# FIVE
FIVE_UN_INDEX = [2, 3, 4, 5, 7, 10]
FIVE_AT_INDEX = [10, 20, 30, 40, 50]
FIVE_TT_INDEX = [1, 3, 5, 7, 9]
FIVE_UPN_INDEX = [4, 7, 10, 13, 16]


ALGOS = ['G2ANet', 'DPPO', 'CPPO', 'IC3Net', 'ConvLSTM', 'GCRL', 'Random']


yranges = {
        "QoI": [1, 3.5],
        "episodic_aoi": [0, 60],
        "aoi_satis_ratio": [0.0, 1.1],
        "data_satis_ratio": [0.0, 1.1],
        "energy_consuming": [1.0, 2.0],
    }

xlabels = {
    'uav_num': "No. of UAVs",
    'aoith': "AoI threshold",
    'txth': "Tx threshold",
    'update_num': "No. of Antennas",
}
