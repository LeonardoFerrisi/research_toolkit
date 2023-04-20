import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import os

def generate_graph_lines(ignore_subject=None, datafilename=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, datafilename)

    # Read in the CSV data
    df = pd.read_csv(data_path)

    # Set up the plot
    fig, ax = plt.subplots()

    # Loop over each subject ID and plot the data
    for subject_id in df['Subject ID'].unique():
        if not ignore_subject or subject_id not in ignore_subject:
            sub_df = df[df['Subject ID'] == subject_id]
            ax.plot(sub_df['Order'], sub_df['Prediction Accuracy'], label=subject_id)

    # Add a trend line for the overall data
    x = df.groupby(['Order'])['Prediction Accuracy'].mean().index
    y = df.groupby(['Order'])['Prediction Accuracy'].mean().values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color='black', alpha=0.5, linewidth=3,label='Overall')

    # # Set ticks to be only the values contained in order
    ax.set_xticks([1, 2, 3])

    # Calculate the p-value for the overall trend
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    print("Overall trend p-value:", p_value)

    statistical_test(data_path)

    # tukey_test(data_path)

    # Add a legend and axis labels
    ax.legend(loc='upper right', title='Subject ID')
    ax.set_title("Prediction Accuracy vs # of Conditions Tested")
    ax.set_xlabel('Conditions Tested')
    ax.set_ylabel('Prediction Accuracy')

    # Show the plot
    plt.show()

def statistical_test(datapath):
    import scipy.stats as stats

    print("\n\nPerforming one-way ANOVA test...")

    # Read in the CSV data
    df = pd.read_csv(datapath)

    # Perform one-way ANOVA test
    fvalue, pvalue = stats.f_oneway(
        df[df['Order'] == 1]['Prediction Accuracy'],
        df[df['Order'] == 2]['Prediction Accuracy'],
        df[df['Order'] == 3]['Prediction Accuracy']
    )

    if pvalue < 0.05:
        print("The ANOVA test is statistically significant (p < 0.05).")
    else:
        print("The ANOVA test is not statistically significant (p >= 0.05).")

def generate_graph_boxandwhisker(ignore=None, datafilename=None, axis_fontsize=12, axis_fontweight='bold'):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, datafilename)

    # Read in the CSV data
    df = pd.read_csv(data_path)

    # Set up the plot
    fig, ax = plt.subplots()

    # Generate a subject data frame containing all the subjects if they are not in the parameter ignore
    if type(ignore) == str: ignore = [ignore]
    for subject_id_to_remove in ignore:
        subject_df = df = df[df['Subject ID'] != subject_id_to_remove]

    order_stats = []
    for order in subject_df['Order'].unique():
        order_df = subject_df[subject_df['Order'] == order]
        # create a list of boxplot statistics for the run
        stats = [order_df['Prediction Accuracy'].min(),
                 order_df['Prediction Accuracy'].quantile(0.25),
                 order_df['Prediction Accuracy'].median(),
                 order_df['Prediction Accuracy'].quantile(0.75),
                 order_df['Prediction Accuracy'].max()]
        order_stats.append(stats)
    
    print(order_stats)

    positions = [1, 2, 3]
    ax.set_xticks(positions)
    ax.set_xticklabels(['1', '2', '3'])

    # Set the y lim to accomadate all values
    ax.set_ylim(0.3, max([max(stats) for stats in order_stats]) + 0.05)


    # plot the boxplots and add error bars
    boxprops = dict(facecolor='orange', linewidth=2)
    whiskerprops = dict(linestyle='--', color='red', linewidth=2)
    capprops = dict(color='black', linewidth=2)
    flierprops = dict(marker='o', markersize=8, markerfacecolor='red')
    medianprops = dict(linestyle='-', color='black', linewidth=2)

    ax.boxplot(order_stats, positions=positions, widths=0.5, patch_artist=True,
            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
            flierprops=flierprops, medianprops=medianprops)
    
    # show without outliers
    # ax.boxplot(order_stats, positions=positions, widths=0.5, patch_artist=True,
    #        boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
    #        flierprops=flierprops, medianprops=medianprops, showfliers=False)

    #### 

    # Add a legend and axis labels
    ax.set_title("Prediction Accuracy vs # of Sets Tested", fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_xlabel('# of Sets Tested', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Prediction Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)

    plt.show()

def statistical_test(datapath):
    import scipy.stats as stats

    print("\n\nPerforming one-way ANOVA test...")

    # Read in the CSV data
    df = pd.read_csv(datapath)

    # Perform one-way ANOVA test
    fvalue, pvalue = stats.f_oneway(
        df[df['Order'] == 1]['Prediction Accuracy'],
        df[df['Order'] == 2]['Prediction Accuracy'],
        df[df['Order'] == 3]['Prediction Accuracy']
    )

    if pvalue < 0.05:
        print("The ANOVA test is statistically significant (p < 0.05).")
    else:
        print("The ANOVA test is not statistically significant (p >= 0.05).")

def tukey_test(datapath):
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import MultiComparison

    # Read in the CSV data
    df = pd.read_csv(datapath)

    # Perform one-way ANOVA
    model = ols('`Prediction Accuracy` ~ C(Order)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Calculate the mean prediction accuracy for each order
    mean_acc = df.groupby('Order').mean()['Prediction Accuracy']

    # Perform all-pairs Tukey test
    tukey_result = sm.stats.multicomp.pairwise_tukeyhsd(mean_acc.values, mean_acc.index)

    # Print the ANOVA table and Tukey test results
    print('ANOVA table:\n', anova_table)
    print('\nTukey test results:\n', tukey_result)


if __name__ == '__main__':
    generate_graph_boxandwhisker(ignore="T", datafilename='overtime.csv')
