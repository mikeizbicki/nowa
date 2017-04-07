#!/usr/bin/gnuplot

#plot 'results/cnn-ave-same' using 6:9,\
     #'results/cnn-owa-same' using 6:9,\
     #'results/cnn-dnowa-same' using 6:9

plot 'results/cnn-ave-same' using 8:10,\
     'results/cnn-owa-same' using 8:11,\
     'results/cnn-dowa-same' using 8:11

pause -1
