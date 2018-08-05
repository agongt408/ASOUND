To create an archive with selected files:
# create it with all data dirs but no subdirs, and uncompressed
tar -cv --exclude='*inference*' -f newsgroups-expts.tar *.*
