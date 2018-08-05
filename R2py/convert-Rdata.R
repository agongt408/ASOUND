args = commandArgs()

read_directory = args[6]
n_file = args[7]
write_directory = args[8]
print(read_directory)
print(n_file)
print(write_directory)

for (i in strsplit(n_file, split=",")) {
	path = paste(read_directory, 'data', i, '.Rdata', sep="")
	print(path)
	a = load(path)
	print(dataItems)
	print(typeof(dataItems))
	x = as.data.frame(as.matrix(dataItems))
	#write.csv(as.data.frame(dataItems), paste(write_directory, i, '.csv', sep=""))
	#write.csv(as.data.frame(as.matrix(as.matrix(phi@componentProbs))), paste(write_directory, i, #'.phi.csv', sep=""))
	print(i)
}
