import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_graph(datafilename=None, ascending=False, color='blue', axis_fontsize=12, axis_fontweight='bold'):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, datafilename)

    # Read in the CSV data
    df = pd.read_csv(data_path)

    # Group the data by condition and calculate the mean prediction accuracy for each condition
    mean_acc = df.groupby('Condition')['Prediction Accuracy'].mean()
    std_acc = df.groupby('Condition')['Prediction Accuracy'].std()

    # Sort the mean accuracy values in descending order
    mean_acc = mean_acc.sort_values(ascending=ascending)
    std_acc = std_acc[mean_acc.index]

    # Plot a bar graph of the mean prediction accuracy for each condition
    ax = mean_acc.plot.bar(yerr=std_acc, color=color, capsize=5, edgecolor= 'black')
    ax.set_xlabel('Condition', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Mean Prediction Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_title('Mean Prediction Accuracy by Condition', fontsize=axis_fontsize, fontweight=axis_fontweight)
    plt.xticks(rotation=0)

    # Add labels above each bar
    plt.ylim([0, 1])  # Set y-axis limits
    labelheight = 0.175
    for i, v in enumerate(mean_acc):
        barlabel = str(round((v/1.0)*100, 2))+"%"
        ax.annotate(barlabel, xy=(i, v + labelheight), ha='center', fontweight='bold')  # Add labels above each bar
    plt.show()

if __name__ == "__main__":
    generate_graph(datafilename='conditionvsaccuracy.csv', color='orange')