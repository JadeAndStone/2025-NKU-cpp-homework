import pandas as pd
import matplotlib.pyplot as plt
def plot_loss_curve(csv_file):
    df = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['loss'], linestyle='-', color='b')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
plot_loss_curve('loss_log.csv')