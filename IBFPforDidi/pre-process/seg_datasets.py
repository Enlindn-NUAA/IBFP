import numpy as np

def segdatasets():
    v = np.load("flow_matrix.npy")
    listtime = np.load("listtime_didi.npy")
    print(np.where(listtime==1506268800)[0][0], np.where(listtime==1508860800)[0][0], np.where(listtime==1509206400)[0][0], np.where(listtime==1501516800)[0][0])
    start_sjc = np.where(listtime==1501516800)[0][0]
    rainday_sjc = np.where(listtime==1506268800)[0][0]-start_sjc
    regular_sjc = np.where(listtime==1508860800)[0][0]-start_sjc
    holiday_sjc = np.where(listtime==1509206400)[0][0]-start_sjc
    
    v = v[start_sjc:]
    listtime = listtime[start_sjc:]
    
    v1 = v[rainday_sjc:rainday_sjc+24,...]
    listtime1 = listtime[rainday_sjc:rainday_sjc+24]
    np.save('./testdata_taxi/rainydata.npy', v1)
    np.save('./testdata_taxi/rainytime.npy', listtime1)
    
    v1 = v[holiday_sjc:holiday_sjc+24,...]
    listtime1 = listtime[holiday_sjc:holiday_sjc+24]
    np.save('./testdata_taxi/holidaydata.npy', v1)
    np.save('./testdata_taxi/holidaytime.npy', listtime1)
    
    v1 = v[regular_sjc:regular_sjc+24,...]
    listtime1 = listtime[regular_sjc:regular_sjc+24]
    print(listtime1.shape, v1.shape)
    np.save('./testdata_taxi/regulardata.npy', v1)
    np.save('./testdata_taxi/regulartime.npy', listtime1)
    
    vend = v[0:rainday_sjc,...]
    listtimeend = listtime[0:rainday_sjc]
    v2 = v[rainday_sjc+24:regular_sjc,...]
    vend = np.concatenate((vend, v2))
    listtime2 = listtime[rainday_sjc+24:regular_sjc]
    listtimeend = np.concatenate((listtimeend, listtime2))
    v2 = v[regular_sjc+24:holiday_sjc,...]
    vend = np.concatenate((vend, v2))
    listtime2 = listtime[regular_sjc+24:holiday_sjc]
    listtimeend = np.concatenate((listtimeend, listtime2))
    v2 = v[holiday_sjc+24:,...]
    vend = np.concatenate((vend, v2))
    listtime2 = listtime[holiday_sjc+24:]
    listtimeend = np.concatenate((listtimeend, listtime2))
    
    np.save('./traindata_taxi/traindata.npy', vend)
    np.save('./traindata_taxi/traintime.npy', listtimeend)

if __name__ == '__main__':
    segdatasets()
