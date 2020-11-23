import numpy as np
import random

indices = list(range(50000))
np.random.shuffle(indices)

bag1 = indices[:30000]
bag2 = indices[15000:45000]
bag3 = np.append(indices[30000:45000], indices[:15000])
val = indices[45000:50000]

np.save("./bag1.npy", bag1)
np.save("./bag2.npy", bag2)
np.save("./bag3.npy", bag3)
np.save("./val.npy", val)