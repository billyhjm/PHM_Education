import numpy as np
from scipy import signal, stats
from scipy.interpolate import interp1d
from matplotlib import pyplot
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def z_test(x):
    
    out = x*x
    
    print('Test Performed - Input is ', x)
    print('Processed Output will be ', out)
    
    return out

def z_fft_simple(xn,fs):
  N=len(xn)
  T=N/fs
  f=np.arange(0,fs,1/T)
  A=abs(np.fft.fft(xn.T)).T
  f=f[:int(N/2+1)]
  A=A[:int(N/2+1)]/N
  A[1:]=2*A[1:]
  A=abs(A)

  return f, A
  
def  z_feature_time(v):
  RMS = np.sqrt(np.mean(v**2))
  SKEW = stats.skew(v)
  KURT = stats.kurtosis(v, axis=0, fisher=False)
  CF = max(v)/np.sqrt(np.mean(v**2))

  feature = np.array([RMS, SKEW, KURT, CF])
  feature_name = ['RMS', 'Skew','Kurt','CF']

  return feature, feature_name

def  z_feature_freq(v, fs, band=[]):
  
  feature = []
  feature_name = []
  
  if band.size != 0:
    f, A = z_fft_simple(v-np.mean(v), fs)
    for n in range(np.size(band,0)):
      Wn=np.array(band[n,:].flat)
      idx_1= f > band[n,0] 
      idx_2= f < band[n,1]
      idx_and=idx_1*idx_2
      
      A_band = A[idx_and]
      feature_band_rms = np.sqrt(np.mean(A_band**2))
      feature.append(feature_band_rms)
      feature_name.append('Band'+str(n+1))
      
  feature = np.array(feature)
      
  return feature, feature_name

def z_draw_features(feature_t_n, feature_t_f, feature_t_name, feature_f_n, feature_f_f, feature_f_name):
    feature_t_ratio = feature_t_f/feature_t_n
    feature_f_ratio = feature_f_f/feature_f_n
    
    fig, [ax1, ax2] = plt.subplots(1,2,figsize=(15,5))
    ax1.plot(feature_t_n,'-bo')
    ax1.plot(feature_t_f,'-rx')
    ax1.set_xticks(np.arange(0,np.size(feature_t_name)), feature_t_name)
    ax1.set_ylabel('Time Features')

    ax2.plot(feature_f_n,'-bo')
    ax2.plot(feature_f_f,'-rx')
    ax2.set_xticks(np.arange(0,np.size(feature_f_name)), feature_f_name)
    ax2.set_ylabel('Freq. Features')
    ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.tight_layout(pad=1)
    plt.show()

    fig, [ax1, ax2] = plt.subplots(1,2,figsize=(15,5))
    ax1.plot(feature_t_ratio,'-bo')
    ax1.axhline(y=1, color='r', linestyle='--')
    ax1.set_xticks(np.arange(0,np.size(feature_t_name)), feature_t_name)
    ax1.set_ylabel('Time Features Ratio')
    ax1.set_ylim(bottom=0)

    ax2.plot(feature_f_ratio,'-bo')
    ax2.set_xticks(np.arange(0,np.size(feature_f_name)), feature_f_name)
    ax2.set_ylabel('Freq. Features Ratio')
    ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    ax2.axhline(y=1, color='r', linestyle='--')
    ax1.set_ylim(bottom=0)
    fig.tight_layout(pad=1)
    plt.show()

def z_resampling(time,v_sample,degree_sample,f_resampling, trig_rot):
    idx_zero_deg=degree_sample==0
    time[idx_zero_deg]=[]; v_sample[idx_zero_deg]=[]; degree_sample[idx_zero_deg]=[]
    starting=degree_sample[0]
    ending=degree_sample[-1]
    degree_re_delta = 360/f_resampling
    degree_resampling = np.arange(starting+degree_re_delta, (ending-degree_re_delta), degree_re_delta)
    fx_t_resampling = interp1d(degree_sample, time)
    t_resampling=fx_t_resampling(degree_resampling)

    fx_v_resampling = interp1d(time, v_sample)
    v_resampling=fx_v_resampling(t_resampling)

    if trig_rot==1:
        rem = np.mod(len(v_resampling),f_resampling)
        v_resampling = v_resampling[:-rem]

    return t_resampling, v_resampling, degree_resampling, f_resampling


def plot3(a,b,c,mark="o",col="r",az=-60,el=30,lw=10):
  ion()
  fig=plt.figure(figsize=(8,8))
  ax = Axes3D(fig,azim=az,elev=el)
  ax.plot3D(a.flatten(),b.flatten(),c.flatten(),color=col, linewidth=lw)
  #ax.scatter(a, b, c,marker=mark,color=col)


def surf(Z, colormap, X=None, Y=None, C=None, shade=None):
    if X is None and Y is None:
        X, Y = meshgrid_of(Z)
    elif X is None:
        X, _ = meshgrid_of(Z)
    elif Y is None:
        _, Y = meshgrid_of(Z)

    if C is None:
        C = Z
        
    scalarMap = cm.ScalarMappable(norm=Normalize(vmin=C.min(), vmax=C.max()), cmap=colormap)

    # outputs an array where each C value is replaced with a corresponding color value
    C_colored = scalarMap.to_rgba(C)

    ax = gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=C_colored, shade=shade)

    return ax, surf

def meshgrid_of(A):
    xx, yy = meshgrid(range(shape(A)[1]), range(shape(A)[0]))
    return xx, yy


def filtering(v, fs, Wn, ftype):
    filter_order=10; #filter order
    
    Fn=fs/2;

    [z,p,k] = signal.butter(filter_order,Wn/Fn,btype=ftype,output='zpk')

    sos = signal.zpk2sos(z,p,k); #g is same as k
    sos[0,0:3]=sos[0,0:3]/k 
    
    v_filter = signal.sosfilt(sos,v)*k
    return v_filter