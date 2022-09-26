
import matplotlib.pyplot as plt
import redis
import numpy as np

generation = []
generation_avg = []
generation_max = []
generation_min = []
r = redis.Redis()
keys = r.keys()    
for counter in range(1,int(len(keys)/3)+1):
    generation.append(counter)
    generation_max.append(r.get(f'{counter}:max').decode("utf-8"))
    generation_min.append(r.get(f'{counter}:min').decode("utf-8"))
    generation_avg.append(r.get(f'{counter}:avg').decode("utf-8"))
plt.plot(generation, generation_min, label = "generation_min")
plt.plot(generation, generation_avg, label = "generation_avg")
plt.plot(generation, generation_max, label = "generation_max")
plt.legend()
plt.savefig('learning_curve.png')

# lst = [1,2,3,4,5]
# print(lst)
# print(list(np.random.choice(lst,2,p=[0.3,0.2,0.2,0.1,0.2])))
