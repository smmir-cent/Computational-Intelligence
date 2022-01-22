
import matplotlib.pyplot as plt
import redis
# x1 = [1,2,3]
# y1 = [2,4,1]
# plt.plot(x1, y1, label = "line 1")
 
# x2 = [1,2,3]
# y2 = [4,1,3]
# plt.plot(x2, y2, label = "line 2")
 
# plt.xlabel('x - axis')
# plt.ylabel('y - axis')
# plt.title('Two lines on same graph!')
# plt.legend()
# # plt.show()
# plt.savefig('foo.png')

generation = []
generation_avg = []
generation_max = []
generation_min = []
r = redis.Redis()
keys = r.keys()
# keys = map(lambda x:x.decode("utf-8"),keys_)
# keys = sorted(keys) 
# print(keys) 
# print(keys[0]) 
# print()
    
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

# r.set("1:min",10)
# print(r.get("1:min").decode("utf-8"))
# r.flushdb()