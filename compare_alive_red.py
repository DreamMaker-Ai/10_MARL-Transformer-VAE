import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    filetype = '/history/run-.-tag-mean_num_alive_red_ratio.csv'
    filelist = [
        '01_Baseline/trial-1',
        # '02_Add_LN/trial-20',
        # '03_Big_batch/trial-30',
        # '04_Deeper_Transformer/trial-40',
        '05_Frame_stack/trial-50'
        # '06_Obstacles/trial-60'
    ]
    colorlist = ['r', 'b', 'g', 'm', 'y']

    for f, c in zip(filelist, colorlist):
        ff = f + filetype
        csv_path = Path(__file__).parent / ff

        csv_df = pd.read_csv(csv_path)

        wall_time = csv_df[csv_df.columns[0]]
        step = csv_df[csv_df.columns[1]]
        alive_red_ratio = csv_df[csv_df.columns[2]]

        plt.xlabel('learning steps [k]')
        plt.ylabel('mean alive red ratio')

        plt.plot(step/1000, alive_red_ratio, linestyle='solid', color=c, alpha=0.7, linewidth=1,
                 label=f)

    # plt.yscale('log')
    plt.title('Mean alive red ratio vs Learning steps (Over 50 random tests)')
    plt.grid(which="both")
    plt.minorticks_on()
    plt.legend()

    savedir = Path(__file__).parent / 'history_plots'
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    savename = filetype.replace('/history/run-.-tag-', '')
    savename = savename.replace('.csv', '')
    plt.savefig(str(savedir) + '/' + savename + '.png', dpi=300)

    plt.show()


if __name__ == '__main__':
    main()
