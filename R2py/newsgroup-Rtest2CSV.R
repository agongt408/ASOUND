library(Matrix)

for (dir in list.files('rsi/datasets/newsgroups/R/')){
  if (!(file.exists(paste('rsi/datasets/newsgroups/csv/test', dir, sep='')))){
    print(dir)
    dir.create(file.path('rsi/datasets/newsgroups/csv/test', dir))
  }
  
  for (f in list.files(paste('rsi/datasets/newsgroups/R/', dir, sep=''), pattern = '*.Rdata')) {
    #print(f)
    path = paste('rsi/datasets/newsgroups/R/', dir, '/', f, sep='')
    # print(path)
    a = load(path)
    root = substr(f, 1, nchar(f) - 6)
    print(root)
    #write.csv(as.data.frame(as.matrix(dataItems)), paste('rsi/datasets/newsgroups/csv/test/', dir, '/', root, '.csv', sep=''))
    writeMM(dataItems, paste('rsi/datasets/newsgroups/csv/test/', dir, '/', root, '.mtx', sep=''))
  }
}
