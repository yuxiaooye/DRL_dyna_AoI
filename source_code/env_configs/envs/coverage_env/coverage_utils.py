
class Coverage:
    def __init__(self):
        self.throughput_demand_change_freq = 10  # 单位为MDP中的时间步
        self.throughput_demand = [1e8, 5e8, 10e8]
        self.reward_mapping = {
            1e8: 1,
            5e8: 2,
            10e8: 3
        }
        self.folium_poi_radius_mapping = {
            1e8: 2,
            5e8: 3.5,
            10e8: 5
        }

