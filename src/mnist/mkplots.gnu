#!/usr/bin/gnuplot

plot 'results/cnn-ave-same' using 6:9,\
     'results/cnn-owa-same' using 6:9,\
     'results/cnn-dnowa-same' using 6:9

pause -1
