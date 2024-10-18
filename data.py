import pandas as pd
from sklearn.preprocessing import StandardScaler

def data_preprocess(df: pd.DataFrame):
    corr_list = [['spkts', 'sbytes', 'sloss'],
                 ['dpkts', 'dbytes', 'dloss'],
                 ['sinpkt', 'is_sm_ips_ports'],
                 ['swin', 'dwin'],
                 ['tcprtt', 'synack'],
                 ['ct_srv_src'],
                 ['ct_dst_ltm', 'ct_src_dport_ltm'],
                 ['ct_src_dport_ltm', 'ct_dst_sport_ltm'],
                 ['is_ftp_login', 'ct_ftp_cmd']]

    for i in range(len(corr_list)):
        df.drop(columns=corr_list[i][1:], inplace=True, errors="ignore")

    if 'proto' in df.columns:
        df['proto'] = df['proto'].astype('category').cat.codes

    if 'state' in df.columns:
        df['state'] = df['state'].astype('category').cat.codes

    if 'service' in df.columns:
        df.drop(columns=['service'], inplace=True, errors="ignore")

    if 'attack_cat' in df.columns:
        df.drop(columns=['attack_cat'], inplace=True, errors="ignore")

    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
