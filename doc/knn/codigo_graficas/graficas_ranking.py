import matplotlib.pyplot as plt
import numpy as np

scores = np.array([0.7189, 0.7628, 0.7710, 0.7669, 0.7604, 0.7692, 0.7579, 0.7492, 0.7686, 0.7657, 0.7719, 0.7929, 0.7925, 0.7898, 0.7914, 0.7883, 0.7867, 0.7872, 0.7916, 0.7880, 0.7874, 0.7918, 0.7911])

plt.plot(np.arange(len(scores)), scores, "-o", label="Puntuaciones obtenidas")
plt.legend()
plt.show()
