library(Matrix)

for (dir in list.files('rsi/datasets/congress/R/')){
  print(dir)
  if (!(file.exists(file.path('rsi/datasets/congress/csv/test', dir)))){
    dir.create(file.path('rsi/datasets/congress/csv/test', dir))
    print(paste(dir, ' successfully created!'))
  }
  
  for (f in list.files(file.path('rsi/datasets/congress/R', dir), pattern = '*.Rdata')) {
    #print(f)
    path = paste('rsi/datasets/congress/R/', dir, '/', f, sep='')
    # print(path)
    a = load(path)
    root = substr(f, 1, nchar(f) - 6)
    print(root)
    n = substr(root, 5, nchar(f))
    print(n)
    #write.csv(as.data.frame(as.matrix(dataItems)), paste('rsi/datasets/newsgroups/csv/test/', dir, '/', root, '.csv', sep=''))
    write(numPositivePairs, paste('rsi/datasets/congress/csv/test/', dir, '/numPositivePairs', n, '.txt', sep=''))
    writeMM(dataItems, paste('rsi/datasets/congress/csv/test/', dir, '/', root, '.mtx', sep=''))
  }
}
