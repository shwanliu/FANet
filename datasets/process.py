import os 
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
minVal = min(l)
for val in l:
    # print(val)
    # print(minVal)
    weights = str(("%.8f"%(1.0 / ( val / minVal))))
    print(weights)
    res.write(weights+'\n')