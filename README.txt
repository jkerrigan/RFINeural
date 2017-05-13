CNNRFI.py - Main script used to train the CNN, requires access to a larger dataset for incorporating more baseline types.

FindHypers.py - Extension of CNNRFI.py that uses the same CNN structure to iterate over the parameter space and find the optimal parameters.

NeuralAssignPredict.py - Takes the output model (DEEPcnn.txt) and applies it to the rest of the dataset. Usage is 'python NeuralAssignPredict.py zen.2457691.54970.xx.HH.uvc', output file will be 'zen.2457691.54970.xx.HH.uvcr'. To view output information however requires the installation of AIPY at https://github.com/AaronParsons/aipy. To view output RFI flagged file then run 'plot_uv.py zen.2457691.54970.xx.HH.uvcr -a 9_31', which will show the log amplitude of baseline with antennas 9 and 31.
