from sllurp.reader import Reader, R420, ARU2400, FX9600 # for controlling readers

import logging
import time as ti
import csv

logging.basicConfig(filename='llrp.log', level=logging.DEBUG)
epcs = ["1a2fb6463", "1a2fb6445", "1a336f74f", "1a33707e2"]
tag_dict = {"e280689000000001a2fb6463" : [], "e280689000000001a2fb6445": [], "e280689000000001a336f74f": [], "e280689000000001a33707e2": []}    
i = 0

def process_tags(tags):
    epcs = [reader.getEPC(t) for t in tags]
    for val, rssis in tag_dict.items():
        if val in epcs:
            tag = tags[epcs.index(val)]
            epc = reader.getEPC(tag)
            rssi = tag.get('RSSI', tag.get('PeakRSSI'))
            rssis.append(rssi)
        else:
            rssis.append(0)
    # print(tag_dict)

def collect_tags(tag_report):
    tags = tag_report
    process_tags(tags)

reader = R420('169.254.162.28')

print("starting read")
reader.startLiveReports(collect_tags, 
    freqMHz=reader.freq_table[-1], 
    powerDBm=reader.power_table[-1],
    mode=1003,
    tagInterval=0,
    timeInterval=0.1,
    population=4,
    antennas=(1,))

ti.sleep(60)
reader._liveStop.set()

# while(i < time):
#     print(i)
#     i += 1
#     tags = reader.detectTags(freqMHz=reader.freq_table[-1], powerDBm=reader.power_table[-1])
#     process_tags(tags)
#     ti.sleep(0.5)

print(tag_dict)
with open("rssi_time.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter=",")
    for epc, rssis in tag_dict.items():
        writer.writerow([epc] + rssis)