# encoding=utf-8
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def ReadLog(logfile):
    lines = []
    with open(logfile, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            if "Acc" in logfile:
                line = float(line.strip('\n'))/3200
            lines.append(str(line))
    return lines

lines = ReadLog("train0925.log")
print(lines)