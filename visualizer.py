import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def visualize(topic):
    file_name = topic.replace(" ","_")

    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_name+'_Sentiment.csv', encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    # Convert the year to an integer
    df['Yıl'] = df['Yıl'].astype(int)

    # Aggregate the data by year
    yearly_data = df.groupby('Yıl')['Positive'].mean().reset_index()
    yearly_data['Count'] = df.groupby('Yıl')['Positive'].size().values

    # Initialize the plot with a Seaborn theme
    sns.set_theme(style='whitegrid')

    # Create a figure and axis
    plt.figure(figsize=(12, 8),dpi=300)

    # Create a scatter plot
    sns.scatterplot(data=yearly_data, x='Yıl',
                    y='Positive', 
                    size='Count', 
                    sizes=(40, 400), 
                    hue='Positive', 
                    palette='magma_r', 
                    legend=None, 
                    alpha=1)

    # Add a smooth line using Seaborn's regplot for a lowess smoothing
    sns.regplot(data=yearly_data, 
                x='Yıl', 
                y='Positive', 
                scatter=False, 
                color='purple' ,
                order=25,
                ci=None,  #ci - confidence interval
                line_kws={'linewidth': 7} )

    plt.xlim(yearly_data['Yıl'].min()-0.5, yearly_data['Yıl'].max()+0.5)
    plt.ylim(yearly_data['Positive'].min() , yearly_data['Positive'].max()+0.01)

    # Enhancing the plot
    plt.title(topic + ' Ekşi Sözlük Yorumlarının Duygu Analizi', fontsize=24)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(yearly_data['Yıl'].unique())
    plt.yticks([])
    plt.tight_layout()

    # Save the plot as a high-resolution image
    plt.savefig(file_name+'.png', dpi=300)

    # Display the plot
    plt.show()

    print(f"Visualization completed. Output saved to {file_name}.png")
