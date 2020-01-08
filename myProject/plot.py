import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)

sns.set_style("whitegrid")

training_loss = []
validation_loss = []
epoch = 60

with open("./myProject/tempModels/ModAgressive/trainLoss.txt") as f:
    for line in f.readlines():
        training_loss.append(float(line))
with open("./myProject/tempModels/ModAgressive/valLoss.txt") as f:        
    for line in f.readlines():
        validation_loss.append(float(line))


training_loss = [float(loss) for loss in training_loss]
validation_loss = [float(loss) for loss in validation_loss]

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)

# ax = sns.lineplot(x=np.linspace(0, len(training_loss), len(training_loss)), y=training_loss)
# ax = sns.lineplot(x=np.linspace(0, len(training_loss), len(validation_loss)), y=validation_loss)

plt.title("Training Loss vs Validation Loss")
plt.xlabel("Iterations")
plt.plot(np.linspace(0, epoch, len(training_loss)), training_loss, 'b', label='Training Loss')
plt.plot(np.linspace(0, epoch, len(validation_loss)), validation_loss, 'g-.', label='Validation Loss')
plt.legend()
plt.grid(False)
plt.show()
