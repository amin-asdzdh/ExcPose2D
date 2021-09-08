import os
import json
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt


def _get_NEs(results_df):
    # corrects for occluded keypoints when calculating NE

    NEavg = results_df['NEavg'].to_numpy()
    
    num_joints = 6
    NEs = []
    for i in range(num_joints):
        NE = results_df['NE' + str(i + 1)].to_numpy()
        valid_NE = [ne for ne in NE if ne != -1]
        NEs.append(np.average(valid_NE))

    NEavg = np.average(NEavg)
    return NEavg, NEs

def get_results_summary(exp_dir):

    # import results
    results = pd.read_csv(os.path.join(exp_dir, 'results.csv'))

    with open(os.path.join(exp_dir, 'parameters.json')) as json_file:
        params = json.load(json_file)

    # preprocess
    NE, NEs = _get_NEs(results)
    weights = params['weights'][params['weights'].rfind('/')+1:]        
    evalset_dir = params['dataset']
    evalset = evalset_dir[evalset_dir.rfind('/')+1:]
    
    res_summary = [weights, exp_dir, evalset, evalset_dir, NE]
    res_summary.extend(NE_ for NE_ in NEs)

    num_joints = 6
    df_columns = ['weights', 'exp_dir', 'evalset', 'evalset_dir', 'NE']
    df_columns.extend(['NE' + str(i+1) for i in range(num_joints)])

    res_df = pd.DataFrame([res_summary], columns=df_columns)
    res_df.to_csv(exp_dir + '/res_summary.csv', index=False)



if __name__ == "__main__":

    experiment = './runs/val/experiment_1/RealSet_test/*'
    files = glob.glob(experiment)
    for file in files:
        get_results_summary(file)

