import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns
import numpy as np
import argparse
sns.set_style("darkgrid")

parser = argparse.ArgumentParser()
parser.add_argument("-p","--path",help="path of logs to draw a figure",required=True)
args = parser.parse_args()

def logs_to_fig(path_to_logs):
    """
    this function return an image that summarize the logs of one day

    Input:
        path_to_logs: path to the logs
    Output:
        png image in the logs folder
    """
    with open(path_to_logs) as f:
        people = list()
        time = list()
        for line in f:
            people.append(int(line.split(":")[-1].split("personnes")[0].strip()))
            time.append(line.split(" ")[1])
    
        date = line.split(" ")[0].replace("/","-")
        plt.figure(figsize=(70,10))
        plt.title(date)
        plt.xticks(np.arange(len(people)),time,rotation=45)
        plt.ylabel("number of people")
        plt.xlabel("time")
        sns.barplot(x=time,y=people,color="blue")
        root = os.path.dirname(path_to_logs)
        plt.savefig(os.path.join(root,date + "_fig"))

if __name__ == "__main__":
    logs_to_fig(args.path)