import numpy as np
import time

def getprocessed_data():
    processed_data = []
    for i in range(1,9):
        f = open("./haikou_data/dwv_order_make_haikou_%d.txt" %i, 'r')
        line = f.readline()
    
        while True:
            line = f.readline()
            if not line: break
            line = line.split()
            #print(line)
            if line[11]=='0000-00-00': continue
            if line[13]=='0000-00-00': continue
            start_time_form = line[13]+' '+line[14]
            arrive_time_form = line[11]+' '+line[12]
            #print(start_time_form, arrive_time_form,line[11])
            start_time = time.mktime(time.strptime(start_time_form,"%Y-%m-%d %H:%M:%S"))
            arrive_time = time.mktime(time.strptime(arrive_time_form,"%Y-%m-%d %H:%M:%S"))
            #print(start_time, arrive_time)
            processed_data.append(np.array([float(line[0]), start_time, arrive_time, float(line[22]), float(line[21]), float(line[20]), float(line[19])]))
        f.close()
    processed_data = np.array(processed_data)
    print(processed_data.shape)
    np.save('./haikou_data/processed_data.npy', processed_data)

if __name__=='__main__':
    getprocessed_data()
