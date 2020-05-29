
import plotter as pl

labels,data = pl.read_file("strech.txt")
pl.plot_data_scatter_errorbars(labels,data)

#comment out these two lines if you only want to plot, not to fit a line
fit=pl.fit_data(data,"linear")
pl.plot_fit(labels,data,fit)

#give this function a file name to save your plot. Can be pdf, png, jpg, etc. 
pl.show_plots("stretch.pdf")