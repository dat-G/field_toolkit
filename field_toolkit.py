import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib import pyplot as plt
import mvfunc
import psutil
from pandarallel import pandarallel


def plt_dataframe(df, header, cluster=False, url=False):
    """
    :param url:
    :param cluster:
    :param df: Pandas DataFrame
    :param header: the Longitude, Latitude header name in dataframe, format like ['Longitude', 'Latitude']
    :return: None
    """

    positions_lst = df[header].to_numpy()
    if type(cluster) == bool and cluster==False:
        plt.scatter(positions_lst[:, 0], positions_lst[:, 1], marker='o')
    else:
        plt.scatter(positions_lst[:, 0], positions_lst[:, 1], marker='o', c=cluster)
    if type(url) == bool and url==False:
        plt.show()
    else:
        plt.savefig(url)
    plt.close()


def dbscan_field_cluster(df, header):
    """
    :param df: Pandas DataFrame
    :param header: the Longitude, Latitude header name in dataframe, format like ['Longitude', 'Latitude']
    :return: Clustered list
    """
    positions_lst = df[header].to_numpy()
    dbscan = DBSCAN()
    scaler = MinMaxScaler()
    positions_lst_scaled = scaler.fit_transform(positions_lst)
    clusters = dbscan.fit_predict(positions_lst_scaled)
    return clusters


def divided_by_cluster(df, cluster):
    """
    :param df: Pandas DataFrame
    :param cluster: cluster list, format like [0, 0, 1, 1, .. ]
    :return: df_lst, numbers of clusters
    """
    cnt = 0
    cluster_lst = {}
    for it in cluster:
        if cluster_lst.get(it) == None:
            cluster_lst[it] = [cnt]
        else:
            cluster_lst[it].append(cnt)
        cnt += 1
    df_lst = {}
    key_cnt = 0
    for key, value in cluster_lst.items():
        key_cnt += 1
        df_lst[key] = df.loc[value]
    return df_lst, key_cnt


def header_match(df, target):
    """
    :param df: Pandas DataFrame
    :param target: a list consists with keywords in ['time', 'longitude', 'latitude', 'speed', 'direction', 'height', 'tag']
    :return: a list with matched headers in given order
    """
    format_headers_dict = {
        'time': ['time', '时间'],
        'longitude': ['longitude', '经度'],
        'latitude': ['latitude', '纬度'],
        'speed': ['speed', '速度'],
        'direction': ['dir', '方向'],
        'height': ['height', '高度'],
        'tag': ['tag', '标记', '标签', 'tags', 'Tag', 'Tags']
    }
    datasheet_header = df.columns.tolist()
    result = []
    for fh in target:
        for fh_it in format_headers_dict[fh]:
            if fh_it in datasheet_header:
                result.append(fh_it)
                break
        if not any(fh_it in datasheet_header for fh_it in format_headers_dict[fh]):
            raise Exception(fh + " matched failed.")
    return result

def __row_wgs84_GK(row, axis, lon_header, lat_header):
    x, y = mvfunc.WGS84ToGK_Single(row[lon_header], row[lat_header])
    if axis == 'x':
        return x
    elif axis == 'y':
        return y

def std_method_cleaning(source_df):
    """
    :param source_df: Pandas DataFrame
    :return: cleaned dataframe
    """
    # speed_head = header_match(source_df, ['speed'])
    # SpeedRank = source_df[speed_head].sort_values(by=speed_head, ascending=False)
    # SpeedRank

    lat_head = header_match(source_df, ['latitude'])[0]
    lon_head = header_match(source_df, ['longitude'])[0]
    # time_head = ft.header_match(source_df, ['time'])[0]
    speed_head = header_match(source_df, ['speed'])[0]

    # source_df['prev_latitude'] = source_df[lat_head].shift(1)
    # source_df['prev_longitude'] = source_df[lon_head].shift(1)
    # source_df['euclidean_distance'] = source_df.apply(
    #     lambda row: distance.distance((row['prev_latitude'], row['prev_longitude']),
    #                                   (row[lat_head], row[lon_head])).km if pd.notna(row['prev_latitude']) else 0, axis=1)
    #
    # source_df['dtime'] = pd.to_datetime(source_df[time_head])
    # source_df['time_diff'] = source_df['dtime'].diff().dt.total_seconds() / 60 / 60  # 时间差以分钟为单位
    # source_df['average_speed'] = source_df['euclidean_distance'] / source_df['time_diff']
    df_std = source_df[speed_head].std()
    df_mean = source_df[speed_head].mean()
    source_df['condition'] = abs(source_df[speed_head] - df_mean) / df_std > 3
    # source_df['mark'] = source_df[speed_head] / source_df['average_speed']

    # it_minn = source_df.index.min() + 1
    # it_maxx = source_df.index.max() - 1

    it_list = source_df.index.tolist()

    # print(it_minn, it_maxx)

    above_list = source_df.index[source_df['condition']].tolist()
    # print(len(above_list) / len(it_list) * 100, "%")
    #
    # print(it_list)
    # print(above_list)

    source_df = source_df.drop(above_list)
    source_df = source_df.drop(['condition'], axis=1)


    pandarallel.initialize(progress_bar=False, nb_workers=psutil.cpu_count(), verbose=1)
    cal_df = source_df.loc[:, [lon_head, lat_head]]
    x_Series = cal_df.parallel_apply(lambda row: np.float64(__row_wgs84_GK(row, 'x', lon_head, lat_head)), axis=1) # 计算高斯X坐标
    y_Series = cal_df.parallel_apply(lambda row: np.float64(__row_wgs84_GK(row, 'y', lon_head, lat_head)), axis=1) # 计算高斯Y坐标
    # TimeStamp = pd.to_datetime(source_df['Time']).astype(int) / 10 ** 9 # 植入时间戳，步骤已废弃
    source_df.insert(loc=0, column='Gauss_X', value=x_Series)
    source_df.insert(loc=1, column='Gauss_Y', value=y_Series)

    ## for it in range(1, len(source_df) - 1):
    # for it in range(1, len(it_list) - 1):
        # if source_df.loc[it, 'condition']:
    #         source_df.loc[it, speed_head] = int(
    #             (source_df.loc[it_list[it - 1], speed_head] + source_df.loc[it_list[it + 1], speed_head]) / 2)
    # source_df.drop(['euclidean_distance', 'condition', 'dtime', 'time_diff', 'mark', 'average_speed', 'prev_latitude', 'prev_longitude'], axis=1, inplace=True)
    return source_df
