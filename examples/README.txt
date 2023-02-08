Each dataset is given an experiment number, and an associated HDF key.
In the HDF5 file, the dataframe in the key has columns for the evidence when we adjust algo params
These include compute params (i.e. sis_iters) and statistical params (i.e. fitted stated, number of bins)

1. Easy Examples
A. N(-1,1) , N(1,1)
B. N(-2,1) , N(0,1) , N(2,1)

2. Medium Examples
A. N(-0.5,1), N(0.5,1)
B. N(-1,1) , N(0,1) , N(1,1)
C. N(0,1) , N(0,4)
D. N(0,1) , N(0,4) , N(0,9)

3. Hard Examples
A. N(0,4) , N(1,9)
B. N(0,4) , N(1,1) , N(2,9)