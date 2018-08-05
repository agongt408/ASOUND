for (f in list.files('rsi/datasets/newsgroups/achtung/adj_mx/')) {
  #print(f)
	path = paste('rsi/datasets/newsgroups/achtung/adj_mx/', f, sep='')
	# print(path)
	a = load(path)
	root = substr(f, 1, nchar(f) - 6)
	print(root)
	write.csv(as.data.frame(as.matrix(adjMatrix)), paste('rsi/datasets/newsgroups/csv/', root, '.csv', sep=''))
}