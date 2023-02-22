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


yrange_for_metrics = {
        "Data Collection Ratio": [0.0, 1.3],
        "Data Loss Ratio": [0, 0.35],
        "Energy Consumption Ratio": [0.0, 0.5],
        "Geographical Fairness": [0.0, 1.3],
        "Efficiency": [0.0, 13.0],
    }

xlabel_for_xs = {
    'NU': "No. of UAVs/UGVs",
    'SD': "SINR threshold (dB)",
    'NS': "No. of Subchannels",
    'UH': "UAV height (m)",
}
xtick_for_xs = {
    'NU': [1, 2, 3, 4, 5, 7, 10],
    'SD': [-7.0, -2.2, 0.0, 3.0, 7.0],
    'NS': [1, 2, 3, 4, 5, 7, 10],
    'UH': [60, 70, 90, 120, 150],
}
