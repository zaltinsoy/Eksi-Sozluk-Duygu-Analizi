import subprocess
import sentiment_reader
import visualizer
import subprocess

# Define the topic you want to search
topic = input("hangi başlığı aramak istiyorsunuz?\n")

# Run the eksi.py script and pass the topic as input
process = subprocess.Popen(
    ["python", "eksi-sozluk-veri-cekme/eksi.py"],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
)

# Send the topic input to the script and get the output
stdout, stderr = process.communicate(input=f"{topic}\n")

# Optionally print the output of the script to check for errors
if stderr:
    print(f"Error: {stderr}")
else:
    print(f"Output: {stdout}")

# Wait for the process to finish
process.wait()

# Use savasy/bert-base-turkish-sentiment-cased for sentiment analysis
sentiment_reader.sentiment_reader(topic.replace(" ","_"))

# Visualize the output
visualizer.visualize(topic)
