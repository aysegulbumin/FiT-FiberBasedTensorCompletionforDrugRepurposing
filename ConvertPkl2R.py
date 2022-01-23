

file_name = str(sys.argv[1])

print("Reading the pickle file and loading to a dictionary")

file_to_read = open(file_name, "rb")

loaded_dictionary = pickle.load(file_to_read)



