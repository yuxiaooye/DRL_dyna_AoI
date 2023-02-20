import pandas as pd

df = pd.read_csv(r'F:\PycharmProjects\jsac\DRL_dyna_AoI\source_code\envs\NCSU\human120-user50.csv')

ans_df = pd.DataFrame()

for id in range(50):
    sub_df = df[df['id'] == id]
    ans_df = ans_df.append(sub_df.head(10))


ans_df.to_csv(r'F:\PycharmProjects\jsac\DRL_dyna_AoI\source_code\envs\NCSU\human10-user50.csv')