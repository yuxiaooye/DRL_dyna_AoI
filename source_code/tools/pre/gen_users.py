import pandas as pd

df = pd.read_csv(r'F:/PycharmProjects/jsac/DRL_dyna_AoI/source_code/envs/NCSU/human120.csv')


new_df = pd.DataFrame()

POI_NUM = 33
T = 240
if T == 240:
    for i in range(df.shape[0]):
        new_df = new_df.append(df.iloc[i], ignore_index=True)
        if i%121 != 0:
            new_df = new_df.append(df.iloc[i], ignore_index=True)
else:
    raise NotImplementedError

assert new_df.shape[0] == POI_NUM * (T+1)

new_df.to_csv(r'F:/PycharmProjects/jsac/DRL_dyna_AoI/source_code/envs/NCSU/human240.csv')