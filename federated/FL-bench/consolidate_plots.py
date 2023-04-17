import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
def collect_csv_files(path, prefix):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(prefix) and file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def extract_from_filename(fname):
    fname = os.path.basename(fname)
    fname = ".".join(fname.split(".")[:-1])
    fname = fname.split("_")
    hyperparams = {}
    for f in fname:
        if f.startswith("clients"):
            num_clients = int(f.split("-")[1])
            hyperparams['num_clients'] = num_clients
        elif f.startswith("jr"):
            jr = float(f.split("-")[1])
            hyperparams['jr'] = jr
    return hyperparams

def get_df(csv_files):
    dfs = []
    for csv_file in csv_files:
        hyperparams = extract_from_filename(csv_file)
        df = pd.read_csv(csv_file)
        df_list = df['test_before'].tolist()
        dfs.append((hyperparams, df_list))
    return dfs

def plot(dfs, title):
    plt.cla()
    matplotlib.rcParams.update({'font.size': 14})
    dfs = sorted(dfs, key=lambda x: x[0]['num_clients'])
    for hyperparams, df_list in dfs:
        # print(hyperparams['num_clients'])
        plt.plot(df_list, label=f"num_clients={hyperparams['num_clients']}")
    plt.title(title)
    plt.xlabel("round")
    plt.ylabel("test accuracy")
    plt.legend()
    plt.savefig(f'./out/FedAvg/fed_avg_jr-{hyperparams["jr"]}.png')

if __name__ == "__main__":
    path = "./out/FedAvg"
    csv_files = collect_csv_files(path, 'tiny_imagenet')
    dfs = get_df(csv_files)
    for jr in [0.1, 0.4, 0.7, 1.0]:
        plot([df for df in dfs if df[0]['jr'] == jr], f"FedAvg - Join Ratio = {jr}")