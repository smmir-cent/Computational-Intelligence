import numpy as np

ls = np.random.normal(size=(5,4))
ls2 = np.random.normal(size=(4,5))
print(ls)
print("************************************************************")
print(ls2)
print("************************************************************")
print(ls @ ls2)