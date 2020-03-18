import numpy as np
import get_process_data
import form_flow_matrix
import seg_datasets
import get_company_matrix
import get_basis

#transfor the original data into 'npy' format
#get_process_data.getprocessed_data()

#make flow matrix, whoes shape is NxMxM
#form_flow_matrix.formflowmatrix()

#segment the flow matrix into different datasets
#seg_datasets.segdatasets()

#calculate the company matrix C
#get_company_matrix.getcompmatrix()

#using Kmeans of cosine distance for calculating the basis B
get_basis.getbasis()
