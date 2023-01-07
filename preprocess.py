# TODO: 保存节点数量
# TODO：保存边的关系
import pandas as pd
import pickle
df = pd.read_csv(
    '/Users/tedlau/PostGraduatePeriod/Graduate/Tasks/Task9-神经网络比较-不会--但又给续上了/data_by_day_simple/datatest.txt',
    sep='	', header=None, names=['src', 'dst', 'weight', 'date'])
# for _, group in df.groupby('date'):
#
#     # Create a list of graphs, one for each day
#     graphs = []
#     labels = []
previous_node_numbers = []
for _, group in df.groupby('date'):
    ip_to_id = {}
    id_to_ip = {}
    idx = 0
    for ips in group[['src', 'dst']].values:
        for ip in ips:
            if ip not in ip_to_id:
                ip_to_id[ip] = idx
                id_to_ip[idx] = ip
                idx += 1

    # Convert the src and dst IP addresses to integers using the mapping
    group['src'] = group['src'].apply(lambda x: ip_to_id[x])
    group['dst'] = group['dst'].apply(lambda x: ip_to_id[x])
    # Create a graph for the current day
    previous_node_numbers.append(idx)
print(previous_node_numbers)
pickle.dump(previous_node_numbers, open('./dataset/day_label.pkl','wb'))
