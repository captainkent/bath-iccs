# plot the binary cross entropy function

import matplotlib.pyplot as plt
import numpy as np
import math

eps = 0.00001
x = np.linspace(0 + eps, 1 - eps, 50)

y1 = [-math.log(i) for i in x]
y2 = [-math.log(1 - i) for i in x]

plt.plot(x, y1, "b-")
plt.plot(x, y2, "g--")
plt.legend(["The actual lable is 1 (y=1)", "The actual lable is 0 (y=0)"])
plt.xlabel("The predict probability")
plt.ylabel("Loss")
plt.title("Binary cross entropy loss")
plt.show()
