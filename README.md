# ukraine_test_case
The folder data_files contains the publicly available data we used to build the case, including generator and city locations, and the locations of transmission lines from the ENTSO-E map. 
To build a test case, run original_test_case.py to save a ukraine_full.mat file. Run convert_to_m.m in Matlab to convert the ukraine_full.mat to ukraine_full.m. Then, run tranformer_parameters.jl in Julia to 
add thermal limits for the transformers. This will save the final test case ukraine_full.mat.