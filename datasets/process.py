import os 
import numpy as np
f =open('/Users/liuxiaoying/workplace/CV_Code/FANet/datasets/stats.txt','r')
res = open('/Users/liuxiaoying/workplace/CV_Code/FANet/datasets/stats_weights.txt','w')

l = []
for line in f.readlines():
    # print(str(float(line.split(" ")[2])+float(line.split(" ")[1])))
    tmp = float(line.split(" ")[1]) / 202599
    l.append(tmp)
    # tmp = 1 / (tmp / min(l))
    # print(tmp)
    # res.write(str(tmp)+'/n')
# print(l)
npl = np.array(l)
# print(npl)
minVal = min(l)
# print(minVal)
h = []
for val in l:
    # print(val)
    # print(minVal)
    weights = str(("%.8f"%(1+ 1.0 / ( val / minVal))))
    x = float(val - np.min(npl))/(np.max(npl)- np.min(npl))
    print(x)
    h.append(x)

    # print(weights)
    res.write(weights+'\n')
print(sum(h))