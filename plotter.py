# Royal Holloway Physics department first year fitting toolset
# Fit types: linear, quadratic, exponential, gaussian

# Written by Daniel R. M. Woods

# Code adapted from Fitter by Prof. Stewart T. Boogert

# First Version 06/03/2019

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

# Equations that define fits
# fit after type to avoid accidentally calling the function


def linear_fit(x,p0,p1) : 

    return p0*x+p1


def quadratic_fit(x,p0,p1,p2) : 

    return p0*x**2+p1*x+p2 


def exponential_fit(x,p0,p1) :

    return p0*np.exp(p1*x)


def gaussian_fit(x,p0,p1,p2) : 

    return p0/(p2*np.sqrt(2*np.pi))*np.exp(-(x-p1)**2/(2*p2**2))   

def dampedosc_fit(x,p0,p1,p2,p3,p4) :
    return p0*np.exp(-p1/2.0*x)*np.cos(p2*x+p3)+p4
 
def resonance_fit(x,p0,p1,p2) :
    return p0/np.sqrt(p1*p1*x*x+np.power(x*x-p2*p2,2))    


def read_file(file_name):
    """ 
    
    Reads in a data file and assigns the labels and data to given variables.
    
    The file name must be passed in as a string, i.e. surrounded by ' or ".
    
    Usage: labels, data = read_file('file_name')
    
     File  :

    <Title>

    <xDataLabel>    <yDataLabel>  <--- Tab Separated

    <x1>         <y1>         <x1Err>    <y1Err>

    <x2>         <y2>         <x2Err>    <y2Err> 

    <x3>         <y3>         <x3Err>    <y3Err> 
    
    """

    file = open("stretch.txt")
    
    data = []
    
    labels = []
    
    i_read_labels = 0

    # Loop checks the first lines of the data file, then reads the axis labels
    # and the data
    
    for l in file: 

        if l[0] == '#' : 

        # Ignores commented lines
        
            continue

        elif i_read_labels == 0  : 

        # Reads the first line (i.e. title)

            title = l.strip()  

            labels.append(title)

            i_read_labels += 1

            continue

        elif i_read_labels == 1 : 
        
        # Reads the second line (i.e. x and y-axis)

            axis_names = l.strip()
            
            if not axis_names:    # Checks if axis_names is an empty string, 
                                # meaning that there is no figure title in the
                                # data file
        
                print('\n'+'----------')
                print('''\
    ERROR: Figure title not correctly formatted. First uncommented line of file 
    must be the figure title, next uncommented line must be the axis titles 
    seperated by a tab.'''+'\n')
                print('''\
    You will need to redo read_file before continuing.''')
                print('----------'+'\n')
                labels = None
                data = None
                return
            
            axis_names = axis_names.split('\t')
            
            if len(axis_names) != 2:    # Checks that axis labels were formatted
                                        # correctly
            
                print('\n'+'----------')
                print('''\
    ERROR: Axis titles not correctly formatted. Axes titles must be seperated 
    by a single tab. Some editors may not enter a tab properly (i.e. Spyder), 
    instead you can open your file in notepad and press tab there.'''+'\n')
                print('''\
    You will need to redo read_file before continuing.''')
                print('----------'+'\n')
                labels = None
                data = None
                return
                
            labels.append(axis_names[0])
            labels.append(axis_names[1])

            i_read_labels += 1

            continue

        # Reads every line in file, adding lines with data of length 4, 
        # removing blank lines and throwing an error if line is not length 4

        line = l.strip().split()
        
        if line != []:

            if len(line) == 4:
            
                data.append(list(map(float,line)))
            
            else:
                
                print('\n'+'----------')
                print('''\
    ERROR: line '''+str(line)+'''
    is not of length 4. All lines in the file must contain an x-value, a 
    y-value, an x-error, and a y-error.'''+'\n')
                print('''\
    If this looks like an axis or figure title, then your file was not 
    correctly formatted. First uncommented line of file must be the figure 
    title, next uncommented line must be the axis titles seperated by a tab, 
    and all other uncommented lines must be data.'''+'\n')
                print('''\
    You will need to redo read_file before continuing.''')
                print('----------'+'\n')
                labels = None
                data = None
                return
        
        else:
        
            continue

    data = np.array(data)

    xErr  = data[:,2]

    yErr  = data[:,3]
            
# Checks for 0s in errors
# May not be required with the proper set-up of curve_fit
    for i in xErr:
        if i == 0:
            print('\n'+'----------')
            print('''\
    ERROR: Plotter will not be able to fit a curve to data that has 
    uncertainties of zero.'''+'\n')
            print('''\
    You will need to redo read_file before continuing.''')
            print('----------'+'\n')
            labels = None
            data = None
            return
                
    for i in yErr:
        if i == 0:
            print('\n'+'----------')
            print('''\
    ERROR: Plotter will not be able to fit a curve to data that has 
    uncertainties of zero.'''+'\n')
            print('''\
    You will need to redo read_file before continuing.''')
            print('----------'+'\n')
            labels = None
            data = None
            return

    print('File name             = ',file_name)

    print('Data shape            = ',np.shape(data))
    
    return(labels, data)

    
def plot_data_scatter(labels, data, file_name = None,
                      format_data = ['b','o', 'None']):
    """
    
    Plots a scatter plot of a given set of data with a given set of axis labels.
    
    Data should by of the form of an array of 4 element lists, containing 
    x-data, y-data, x-error, and y-error.
    
    If a string is also passed to the function, it will save the figure in the 
    working directory with that filename.
    
    A list of three strings can be passed to the function to provide line 
    formating in the following format: ['colour', 'marker', 'linestyle'].
    
    Go to the following site for information on allowed formating 
    parameters: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    Usage: plot_data_scatter(labels, data)
       or: plot_data_scatter(labels, data, 'figure1')
           plot_data_scatter(labels, data, 'figure1', ['m','x','-'])
           plot_data_scatter(labels, data, format_data=['m','x','-'])
    
    """
    
    plt.scatter(data[:,0], data[:,1], color=format_data[0],
                marker=format_data[1], linestyle=format_data[2])
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if file_name != None:
        plt.savefig(str(file_name))
    plt.show()


def plot_data_scatter_errorbars(labels, data, file_name = None,
                                format_data = ['b','o','None']):
    """
    
    Plots a scatter plot, with errors, of a given set of data with a given set 
    of axis labels. 
    
    Data should by of the form of an array of 4 element lists, containing 
    x-data, y-data, x-error, and y-error.
    
    If a string is also passed to the function, it will save the figure in the 
    working directory with that filename.
    
    A list of three strings can be passed to the function to provide line 
    formating in the following format: ['colour', 'marker', 'linestyle'].
    
    Go to the following site for information on allowed formating 
    parameters: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    Usage: plot_data_scatter_errorbars(labels, data)
       or: plot_data_scatter_errorbars(labels, data, 'figure1')
           plot_data_scatter_errorbars(labels, data, 'figure1', ['m','x','-'])
           plot_data_scatter_errorbars(labels, data, format_data=['m','x','-'])
       
    """
    
    plt.errorbar(data[:,0], data[:,1], data[:,3], data[:,2], 
                 color=format_data[0], marker=format_data[1], 
                 linestyle=format_data[2])
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if file_name != None:
        plt.savefig(str(file_name))
    #plt.show()
    
def plot_data_HR(labels, data, file_name = None,
                                format_data = ['b','o','None']):
    """
    
    Plots an HR diagram, with errors, of a given set of data with a given set 
    of axis labels. 
    
    Data should by of the form of an array of 4 element lists, containing 
    x-data, y-data, x-error, and y-error.
    
    If a string is also passed to the function, it will save the figure in the 
    working directory with that filename.
    
    A list of three strings can be passed to the function to provide line 
    formating in the following format: ['colour', 'marker', 'linestyle'].
    
    Go to the following site for information on allowed formating 
    parameters: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    Usage: plot_data_HR(labels, data)
       or: plot_data_HR(labels, data, 'figure1')
           plot_data_HR(labels, data, 'figure1', ['m','x','-'])
           plot_data_HR(labels, data, format_data=['m','x','-'])
       
    """
    
    plt.errorbar(data[:,0], data[:,1], data[:,3], data[:,2], 
                 color=format_data[0], marker=format_data[1], 
                 linestyle=format_data[2])
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if file_name != None:
        plt.savefig(str(file_name))
    plt.gca().invert_yaxis()
    plt.show()

    
def plot_data_line(labels, data, file_name = None,
                   format_data = ['b','None','-']):
    """
    
    Plots a line plot of a given set of data with a given set of axis labels.
    
    Data should by of the form of an array of 4 element lists, containing 
    x-data, y-data, x-error, and y-error.
    
    If a string is also passed to the function, it will save the figure in the 
    working directory with that filename.
    
    A list of three strings can be passed to the function to provide line 
    formating in the following format: ['colour', 'marker', 'linestyle'].
    
    Go to the following site for information on allowed formating 
    parameters: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    
    Usage: plot_data_line(labels, data)
       or: plot_data_line(labels, data, 'figure1')
           plot_data_line(labels, data, 'figure1', ['m','x','-'])
           plot_data_line(labels, data, format_data=['m','x','-'])
       
    """
    
    plt.plot(data[:,0], data[:,1], 
             color=format_data[0], marker=format_data[1], 
             linestyle=format_data[2])
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    if file_name != None:
        plt.savefig(str(file_name))
    plt.show()

    
def fit_data(data, fit_type, initial_guess = None):
    """
    
    Performs a fit on a data set of a type given and assigns the fit parameters 
    to some given variable. 
    
    A list of initial guess values of the fit parameters can also be passed 
    into the function, but are not required.
    
    Allowed fit types: linear, quadratic, exponential, gaussian.
    
    The length of the list of initial guess values passed to the function 
    varies based on fit type: 2, 3, 2, 3 respectively
    
    Usage: fit = fit_data(data, 'linear')
       or: fit = fit_data(data, 'linear', [m, c])
              
    """
    if fit_type == 'linear' or fit_type == 'Linear':
        print('Fit Type = linear')
        if initial_guess == None:
            initial_guess =  np.zeros(2)
        elif len(initial_guess) != 2:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a two element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(linear_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12,full_output=True)
        print('Parameters                    =  m, c')
    
    elif fit_type == 'quadratic' or fit_type == 'Quadratic':
        print('Fit Type = quadratic')
        if initial_guess == None:
            initial_guess =  np.zeros(3)
        elif len(initial_guess) != 3:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a three element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(quadratic_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12)
        print('Parameters                    =  a, b, c')
    
    elif fit_type == 'exponential' or fit_type == 'Exponential':
        print('Fit Type = exponential')
        if initial_guess == None:
            initial_guess =  np.zeros(2)
        elif len(initial_guess) != 2:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a two element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(exponential_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12)
        print('Parameters                    =  A, k')

    elif fit_type == 'gaussian' or fit_type == 'Gaussian':
        print('Fit Type = gaussian')
        if initial_guess == None:
            initial_guess =  np.zeros(3)
            initial_guess[0] = data[:,1].max()
            initial_guess[1] = data[:,0].mean()
            initial_guess[2] = 5.0
        elif len(initial_guess) != 3:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a three element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(gaussian_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12)
        print('Parameters                    =  a, sigma, mu')
    elif fit_type == 'resonance':
        print('Fit Type = resonance')
        if initial_guess == None:
            initial_guess =  np.zeros(3)
            initial_guess[0] = data[:,1].max()
            initial_guess[1] = 1.0
            initial_guess[2] = 3.0
        elif len(initial_guess) != 3:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a three element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(resonance_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12)
        print('Parameters                    =  F/m, gamma, w0')
    elif fit_type == 'damped' :
        print('Fit Type = damped')
        if initial_guess == None:
            initial_guess =  np.zeros(5)
            initial_guess[0] = data[:,1].max()
            initial_guess[1] = 1.0
            initial_guess[2] = 1.0
            initial_guess[3] = 1.0
        elif len(initial_guess) != 5:
            print('\n'+'----------')
            print('''\
    ERROR: Initial fit parameters not of the correct format. Parameters must be 
    in a three element list or numpy array.''')
            print('----------'+'\n')
            return
        fit = curve_fit(dampedosc_fit, data[:,0], data[:,1], initial_guess, 
                        data[:,3], xtol=1e-12)
        print('Parameters                    =  a, gamma, w, phi, offset')


    else:
        print('\n'+'----------')
        print('''\
    ERROR: function not known. Known functions are: linear, quadratic, 
    exponential, and gaussian.''')
        print('----------'+'\n')
        return
        
    print('Original fit parameters       = ',initial_guess)
    print('Calculated fit parameters     = ',fit[0])
    print('Errors on fit parameters      = ',np.sqrt(np.diag(fit[1])))
    fit = [fit, fit_type]
    return fit
    
    
def plot_fit(labels, data, fit, file_name = None, 
             format_data = ['b','o','None'], format_fit = ['r','None', '-'],data_label='',fit_label=''):
    """
    
    Plots a fit alongside the starting data.
    
    If a string is also passed to the function, it will save the figure in the 
    working directory with that filename.
    
    Two lists of three strings can be passed to the function to dictate the 
    formating of the data set and of the fit, with the following 
    format: ['colour' , 'marker' , 'linestyle'].
    
    Usage: plot_fit(labels, data, fit)
       or: plot_fit(labels, data, fit, 'figure1')
           plot_fit(labels, data, fit, 'figure1', ['m','x','--'], ['g','^',':'])
           plot_fit(labels, data, fit, format_fit=['m','x','--'])
           etc.
    """
    fit_type = fit[-1]
    
    plt.errorbar(data[:,0], data[:,1], data[:,3], data[:,2], 
                 label=data_label, color=format_data[0], 
                 marker=format_data[1], linestyle=format_data[2])
    x_range = np.linspace(np.amin(data[:,0]), np.amax(data[:,0]), 100)
    if fit_type == 'linear' or fit_type == 'Linear':
        plt.plot(x_range, linear_fit(x_range, *fit[0][0]), 
                 label = 'Linear Fit', color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])
        
    elif fit_type == 'quadratic' or fit_type == 'Quadratic':
        plt.plot(x_range, quadratic_fit(x_range, *fit[0][0]), 
                 label = 'Quadratic Fit', color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])
        
    elif fit_type == 'exponential' or fit_type == 'Exponential':
        plt.plot(x_range, exponential_fit(x_range, *fit[0][0]), 
                 label = 'Exponential Fit', color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])
        
    elif fit_type == 'gaussian' or fit_type == 'Gaussian':
        plt.plot(x_range, gaussian_fit(x_range, *fit[0][0]), 
                 label = 'Gaussian Fit', color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])
    elif fit_type == 'damped':
        plt.plot(x_range, dampedosc_fit(x_range, *fit[0][0]), 
                 label = 'Damped Oscillation Fit', color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])
    elif fit_type == 'resonance':
        plt.plot(x_range, resonance_fit(x_range, *fit[0][0]), 
                 label = fit_label, color=format_fit[0], 
                 marker=format_fit[1], linestyle=format_fit[2])        
        
    else:
        print('\n'+'----------')
        print('''\
    ERROR: function not known. Known functions are: linear, quadratic, 
    exponential, and gaussian.''')
        print('----------'+'\n')
        return 
        
    plt.title(labels[0])
    plt.xlabel(labels[1])
    plt.ylabel(labels[2])
    plt.legend(loc='best')
    if file_name != None:
        plt.savefig(str(file_name))
    #plt.show()

def show_plots(file_name=''):
    if file_name != None:
        plt.savefig(str(file_name))
    plt.show()