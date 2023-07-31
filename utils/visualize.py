import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv('amos22/outputs/progress.csv')

# Extract the 'loss' column
loss = data['grad_norm']

# Create a plot
plt.plot(loss)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
plt.savefig("loss.png")