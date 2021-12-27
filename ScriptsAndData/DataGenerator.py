import IPFunctions
import time


address = "GeneratedData/" # Folder where the data will be saved
filename = "penData_"     # filename format, the suffix will be data number 

def GenerateData(numberOfData = 50):
	"Generating Simulation Data, total simulated data is numberOfData"

	print ("..start generating")
	start_time = time.time()

	for i in range (numberOfData):

		fname = "{}{:0>6d}.{}".format(filename, i+1, 'csv')

		IPFunctions.SimulatePendulum( "{}{}".format(address, fname) )

		print ("\r", "processing: {}/{} data".format(i+1, numberOfData))

	print("..done")

	end_time = time.time()
	run_time = end_time - start_time

	print("running time: ", "{:.0f} minutes, {:.0f} seconds".format(run_time / 60, run_time * 10 / 6))


if __name__ == "__main__":
	GenerateData(300)