import numpy as np

from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000

def if_in_clusters(lon, lat, clusters):
	lon_lat_form = (lon,lat)
	flag = False
	location = -1
	for i in range(clusters.shape[0]):
		if (lon_lat_form in clusters[i]):
			flag = True
			location = i
			break
	return flag, location

def formflowmatrix():
    locmin = [110.13, 19.88]
    clusters = np.load('./1.25and10.npy')
    starttime = 1493568000
    endtime = 1509465600
    flow_matrix = np.zeros((int((endtime - starttime)/3600), len(clusters), len(clusters)))
    v = np.load('./haikou_data/processed_data.npy')
    print(flow_matrix.shape)
    
    for i in range(v.shape[0]):
    	if (v[i][1]>starttime) & (v[i][1]<endtime):
    		data_start = v[i][1]
    		data_end = v[i][2]
    		lon_start = int(haversine(float(v[i][4]), float(v[i][3]), locmin[0], float(v[i][3])) / 50)
    		lat_start = int(haversine(float(v[i][4]), float(v[i][3]), float(v[i][4]), locmin[1]) / 50)
    		lon_end = int(haversine(float(v[i][6]), float(v[i][5]), locmin[0], float(v[i][5])) / 50)
    		lat_end = int(haversine(float(v[i][6]), float(v[i][5]), float(v[i][6]), locmin[1]) / 50)
    		#print(lon_start, lat_start, lon_end, lat_end)
    		if haversine(float(v[i][4]), float(v[i][3]), float(v[i][6]), float(v[i][5]))<50:
    			print('too near')
    			continue
    		if ((v[i][4]>locmin[0])&(v[i][3]>locmin[1])&(lon_start<=564)&(lon_start<=446)):
    			if ((v[i][6]>locmin[0])&(v[i][5]>locmin[1])&(lon_end<=564)&(lon_end<=446)):
    				flag_start, location_start = if_in_clusters(lon_start, lat_start, clusters)
    				flag_end, location_end = if_in_clusters(lon_end, lat_end, clusters)
    				#print(flag_end, flag_start)
    				if (flag_end)&(flag_start):
    					#print(i , location_start, location_end, int((data_start - starttime)/3600))
    					flow_matrix[int((data_start - starttime)/3600)][location_start][location_end] += 1
    np.save('./flow_matrix.npy', flow_matrix)


if __name__=='__main__':
    formflowmatrix()















