# metrics
METRICS = ['QoI', 'episodic_aoi', 'aoi_satis_ratio', 'data_satis_ratio', 'energy_consuming']

# hyper tuning
HT_INDEX1 = ['EoiCoef=0.001', 'EoiCoef=0.003', 'EoiCoef=0.01', 'EoiCoef=0.03']
HT_INDEX2 = ['w/o SL, w/o CC', 'w/ SL, w/o CC', 'w/o SL, w/ CC', 'w/ SL, w/ CC']

# FIVE
FIVE_UN_INDEX = [2, 3, 4, 5, 7, 10]
FIVE_AT_INDEX = [10, 20, 30, 40, 50]
FIVE_TT_INDEX = [1, 3, 5, 7, 9]
FIVE_AM_INDEX = [0.25, 0.75, 1.25, 1.75, 2.25]
# FIVE_UPN_INDEX = [4, 7, 10, 13, 16]
FIVE_UPN_INDEX = [1, 2, 3, 4, 5]



ALGOS = ['G2ANet', 'ConvLSTM', 'DPPO', 'CPPO', 'IC3Net', 'GCRL', 'Random']


yranges = {
    "QoI": [0.5, 3.3],
    "aoi_satis_ratio": [0.2, 1.02],
    "data_satis_ratio": [0.2, 1.02],
    "episodic_aoi": [0, 60],
    "energy_consuming": [1.0, 2.0],
    }

ylabels = {
    "QoI": r"Overall QoI Index ($I$)",
    "aoi_satis_ratio": r"AoI Satisfactory Ratio ($I_{\tau}$)",
    "data_satis_ratio": r"Throughput Satisfactory Ratio ($I_{\mu}$)",
    "episodic_aoi": r"Episodic AoI (${\tau}$)",
    "energy_consuming": "Energy Consumption",
}

ynames = {
    "QoI": "QoI",
    "aoi_satis_ratio": "AoI Satisfactory Ratio",
    "data_satis_ratio": "Throughput Satisfactory Ratio",
    "episodic_aoi": "Episodic AoI",
    "energy_consuming": "Energy Consumption",
}

xlabels = {
    'uav_num': r"No. of UAVs ($U$)",
    'aoith': r"AoI threshold ($\tau_{\rm th}$)",
    'txth': r"Generated data amount ($D$)",
    'update_num': "No. of Antennas",
}

xnames = {
    'uav_num': "No. of UAVs",
    'aoith': "AoI threshold",
    'txth': "Generated data amount",
    'update_num': "No. of Antennas",
}
