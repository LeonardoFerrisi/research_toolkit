import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_graph(datafilename=None, ascending=False, color='blue', axis_fontsize=12, axis_fontweight='bold'):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(script_dir, datafilename)

    # Read the data from the CSV file
    df = pd.read_csv(data_path)

    # Remove rows with missing values in 'Prediction Accuracy' column
    df = df.dropna(subset=['Prediction Accuracy'])

    # Plot the graph
    fig, ax = plt.subplots()
    ax.scatter(df['StimPerSecond'], df['Prediction Accuracy'], color=color)  # Scatter plot of all data points

    # Calculate the line of best fit using numpy
    x = df['StimPerSecond']
    y = df['Prediction Accuracy']
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    line = p(x)

    # Plot the line of best fit
    ax.plot(x, line, color='red', linestyle='--')

    ax.set_xlabel('Stimulus per Second', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_ylabel('Prediction Accuracy', fontsize=axis_fontsize, fontweight=axis_fontweight)
    ax.set_title('Prediction Accuracy by Stimulus per Second (Stimulus Speed)', fontsize=axis_fontsize, fontweight=axis_fontweight)

    plt.show()

if __name__ == "__main__":
    generate_graph(datafilename='conditionvsaccuracy2.csv', ascending=True, color='orange')
