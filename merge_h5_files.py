import h5py
import os
from sys import argv
import numpy as np 


def merge_h5_files(input_dir, output_file):
    # Get a list of all H5 files in the input directory
    h5_files = [file for file in sorted(os.listdir(input_dir)) if file.endswith('.h5')]


    # Load the first input file to get the dataset shape
    first_file_path = os.path.join(input_dir, h5_files[0])
    with h5py.File(first_file_path, 'r') as first_file:
        dataset_shape = first_file['dat'].shape

        # Create the output HDF5 file
        with h5py.File(output_file, 'w') as outfile:
            # Create the header dataset
            # outfile.create_dataset('header/dims', data=['h', 'k', 'l', 'E'])
            # outfile.create_dataset('header/start', data=[-1, -1, -1, -0.1])
            # outfile.create_dataset('header/end', data=[1, 1, 1, 0.1])
            # outfile.create_dataset('header/type', data='hypervolume_native')
            # Create the data dataset
            data_shape = (len(h5_files),) + dataset_shape  # Include the first dimension for number of files

            # Shape = (nbr of cuts with respect to E, nbr of cuts with respect to eta ,... ,... )
            data_dataset = outfile.create_dataset('data', shape=(int(argv[3]),int(argv[4]),int(argv[9]),int(argv[10])), dtype=first_file['dat'].dtype)

            data_dataset.attrs.create("start", data=[-1, -1, -1, -0.1], dtype='f')
            data_dataset.attrs.create("end", data=[1, 1, 1, 0.1], dtype='f')
            data_dataset.attrs.create("dims",data=['h','k','l','E'])
            data_dataset.attrs.create("type",data='hypervolume_native')


            # Copy the datasets from the input files to the merged array
            for i, file_name in enumerate(h5_files):
                file_path = os.path.join(input_dir, file_name)
                with h5py.File(file_path, 'r') as infile:
                    # Extract E and eta from the file name
                    E = float(file_name.split('_E=')[1].split('meV')[0])
                    eta = float(file_name.split('_eta=')[1].split('.h5')[0])

                    # Get index from E and eta
                    # i = (energy + energy_min - demi_energy_step)/energy_step
                    i = int(round((E - float(argv[7]) - float(argv[5])*0.5)/float(argv[5])))
                    j = int(round((eta - float(argv[8]) - float(argv[6])*0.5)/float(argv[6])))

                    # Get the data from the input file
                    data = infile['dat']
                    
                    if j==-1 : print("E = ",E,"Eta = ",eta)
                    print("index :",i," ",j)
                    print("max data 1 :" ,np.max(data))

                    data_dataset[i,j] = np.array(data,dtype=float)
                    print("max data 2 :" ,np.max(data_dataset[i][j]),"\n\n\n")
                    
    print(f"Merge complete. Output file saved as: {output_file}")



if __name__=="__main__":
    # Example usage
    input_directory = argv[1]
    output_h5_file = argv[2]
    # nbr_of_cuts_E = argv[3]  # =25 pour tbtio   ( = 21 pour bnfs)
    # nbr_of_cuts_eta = argv[4] # =22 pour tbtio  ( = 67 pour bnfs )
    # dE = argv[5] # =0.2   (0.24 pour bnfs)
    # dw = argv[6] # =0.2   (0.05 pour bnfs)
    # E_min = argv[7] # =-0.9351 pour tbtio  (-0.9825 pour bnfs)
    # w_min = argv[8] # =-1.247           (-3.0342 pour bnfs)
    # dimensions de images : argv[9] et argv[10]   (175,366  pour bnfs)
    merge_h5_files(input_directory, output_h5_file)