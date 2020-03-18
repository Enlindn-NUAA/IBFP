import numpy as np
testdata = np.load('./testdata/testdata.npy')
np.save('./testdata/testdata_part1.npy', testdata[:8,...])
np.save('./testdata/testdata_part2.npy', testdata[8:16,...])
np.save('./testdata/testdata_part3.npy', testdata[16:,...])