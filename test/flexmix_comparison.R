#!/usr/bin/R --slave
# run1: R CMD BATCH test.R; cat test.Rout
# run2: Rscript test.R

library(rhdf5)
library(flexmix)

# load the data
X = h5read("../data/regression_data.hdf5", "X")
Y = h5read("../data/regression_data.hdf5", "Y")

# generate dataframe for X & Y
data <- data.frame(Y=Y, 
                   x1=X[1,], x2=X[2,], x3=X[3,], x4=X[4,], 
                   x5=X[5,], x6=X[6,], x7=X[7,], x8=X[8,], 
                   x9=X[9,], x10=X[10,], x11=X[11,])

m1 = flexmix(Y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11, 
             data=data, k=3)
BIC(m1)
logLik(m1)

m2 = flexmix(Y ~ x1+x2+x3+x4+x5+x6+x7+x8+x9+x10+x11, 
             data=data, k=4)
BIC(m2)
logLik(m2)
