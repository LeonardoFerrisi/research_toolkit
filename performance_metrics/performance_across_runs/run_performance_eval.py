import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def graph_results(filename, color='blue', axis_fontsize=12, axis_fontweight='bold'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, filename)
    # read the CSV file
    df = pd.read_csv(data_path)

    # Calculate average accuracy for each run
    avg_acc = df.groupby('Run Number')['Estimated Accuracy'].mean()
    std_acc = df.groupby('Run Number')['Estimated Accuracy'].std()

    # Create a bar plot
    fig, ax = plt.subplots()
    
    avg_acc.plot(kind='bar', ax=ax)

    std_acc = std_acc[avg_acc.index]

    ax = avg_acc.plot.bar(yerr=std_acc, color=color, capsize=5)
    # Add x and y labels
    ax.set_xlabel('Run Number', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Average Estimated Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)

    # Add title
    ax.set_title('Average Estimated Accuracy for Each Run', fontsize=axis_fontsize, fontweight=axis_fontweight)

    plt.xticks(rotation=0)

    # Add labels above each bar
    plt.ylim([0, 1])  # Set y-axis limits
    labelheight = 0.25
    for i, v in enumerate(avg_acc):
        barlabel = str(round((v/1.0)*100, 2))+"%"
        ax.annotate(barlabel, xy=(i, v + labelheight), ha='center', fontweight='bold')  # Add labels above each bar
    plt.show()

    # Show the plot
    plt.show()

def plot_for_all_line(filename, ignore=None, axis_fontsize=12, axis_fontweight='bold'):
    """
    Plot a line graph visualizing trends in run performance for all subjects
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, filename)
    # read the CSV file
    df = pd.read_csv(data_path)

    # drop rows with NA values
    df.dropna(inplace=True)

    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3, 4])

    # # loop through each subject and plot a line for each run
    # for subject in df['Subject ID'].unique():
    #     subject_df = df[df['Subject ID'] == subject]
    #     ax.plot(subject_df['Run Number'], subject_df['Estimated Accuracy'], label=subject)

    # loop through each subject and plot a line for each run
    for subject in df['Subject ID'].unique():
        if not ignore or subject not in ignore:
            subject_df = df[df['Subject ID'] == subject]
            # Merge all conditions together and get the average for all of a specific run across all conditions
            subject_df = subject_df.groupby('Run Number')['Estimated Accuracy'].mean().reset_index()
            ax.plot(subject_df['Run Number'], subject_df['Estimated Accuracy'], label=subject)
    
    # Add a trend line for the overall data
    x = df.groupby(['Run Number'])['Estimated Accuracy'].mean().index
    y = df.groupby(['Run Number'])['Estimated Accuracy'].mean().values
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), '--', color='black', alpha=0.5, linewidth=3,label='Overall')


    # set the title and labels for the plot
    ax.legend(loc='lower left', title='Subject ID')
    ax.set_title('Performance Accuracy for Each Run', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_xlabel('Run Number', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Estimated Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)


    # show the plot
    plt.show()

def plot_for_all_subjects_box_and_whisker(filename, ignore=None, axis_fontsize=12, axis_fontweight='bold'):
    """
    Plot a Box and Whisker plot visualizing run performance for all subjects
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, filename)
    # read the CSV file
    df = pd.read_csv(data_path)

    # drop rows with NA values
    df.dropna(inplace=True)

    fig, ax = plt.subplots()
    # ax.set_xticks([1, 2, 3, 4])

    # For lines ##################################################################################
    # # loop through each subject and plot a line for each run
    # for subject in df['Subject ID'].unique():
    #     if not ignore or subject not in ignore:
    #         subject_df = df[df['Subject ID'] == subject]
    #         # Merge all conditions together and get the average for all of a specific run across all conditions
    #         subject_df = subject_df.groupby('Run Number')['Estimated Accuracy'].mean().reset_index()
    #         ax.plot(subject_df['Run Number'], subject_df['Estimated Accuracy'], label=subject)
    # Add a trend line for the overall data
    # x = df.groupby(['Run Number'])['Estimated Accuracy'].mean().index
    # y = df.groupby(['Run Number'])['Estimated Accuracy'].mean().values
    # z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    # ax.plot(x, p(x), '--', color='black', alpha=0.5, linewidth=3,label='Overall')
    # loop through each run and create a list of statistics for each run
    # ################################################################################################

    
    run_stats = []
    runs_used =[2,3,4] # Ignore run 1
    
    # Generate a subject data frame containing all the subjects if they are not in the parameter ignore
    for subject_id_to_remove in ignore:
        subject_df = df = df[df['Subject ID'] != subject_id_to_remove]

    for run in runs_used:
        
        run_df = subject_df[subject_df['Run Number'] == run]
        # create a list of boxplot statistics for the run
        stats = [run_df['Estimated Accuracy'].min(),
                 run_df['Estimated Accuracy'].quantile(0.25),
                 run_df['Estimated Accuracy'].median(),
                 run_df['Estimated Accuracy'].quantile(0.75),
                 run_df['Estimated Accuracy'].max()]
        run_stats.append(stats)

    # create a list of positions for the boxplots
    positions = [2,3,4]

    print(run_stats)
    print(positions)

    # plot the boxplots and add error bars
    boxprops = dict(facecolor='orange', linewidth=2)
    whiskerprops = dict(linestyle='--', color='red', linewidth=2)
    capprops = dict(color='black', linewidth=2)
    flierprops = dict(marker='o', markersize=8, markerfacecolor='red')
    medianprops = dict(linestyle='-', color='black', linewidth=2)

    ax.boxplot(run_stats, positions=positions, widths=0.5, patch_artist=True,
            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
            flierprops=flierprops, medianprops=medianprops)


    # set the title and labels for the plot
    # ax.legend(loc='lower left', title='Subject ID')
    ax.set_title('Performance Accuracy for Each Run', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_xlabel('Run Number', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Estimated Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)


    # show the plot
    plt.show()

if __name__ == "__main__":
    # graph_results(filename='allruns.csv', color='orange')

    plot_for_all_subjects_box_and_whisker(filename='allruns.csv', ignore="T")