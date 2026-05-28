import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [50, 0, 0, 0],
    [0, 50, 0, 0],
    [0, 0, 50, 0],
    [0, 0, 0, 52]
])

labels = ["HELLO", "YES", "NO", "NO_SIGN"]

plt.imshow(cm)
plt.colorbar()
plt.xticks(range(len(labels)), labels)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png")
plt.show()