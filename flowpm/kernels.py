""" Implementation of kernels required by FastPM. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy.special import erfc

def fftk(shape, symmetric=True, finite=False, dtype=np.float64):
  """ Return k_vector given a shape (nc, nc, nc) and box_size
  """
  k = []
  for d in range(len(shape)):
    kd = np.fft.fftfreq(shape[d])
    kd *= 2 * np.pi
    kdshape = np.ones(len(shape), dtype='int')
    if symmetric and d == len(shape) -1:
        kd = kd[:shape[d]//2 + 1]
    kdshape[d] = len(kd)
    kd = kd.reshape(kdshape)
    k.append(kd.astype(dtype))
  del kd, kdshape
  return k

def laplace_kernel(kvec):
  """
  Compute the Laplace kernel from a given K vector

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  Returns:
  --------
  wts: array
    Complex kernel
  """
  kk = sum(ki**2 for ki in kvec)
  mask = (kk == 0).nonzero()
  kk[mask] = 1
  wts = 1./kk
  imask = (~(kk==0)).astype(int)
  wts *= imask
  return wts

def gradient_kernel(kvec, direction, order=0):
  """
  Computes the gradient kernel in the requested direction

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  direction: int
    Index of the direction in which to take the gradient

  Returns:
  --------
  wts: array
    Complex kernel
  """
  if order == 0:
    wts = 1j * kvec[direction]
    wts = np.squeeze(wts)
    wts[len(wts) //2] = 0
    wts = wts.reshape(kvec[direction].shape)
    return wts
  else:
    nc = len(kvec[0])
    w = kvec[direction]
    a = 1 / 6.0  * (8 * np.sin(w) - np.sin(2 * w))
    wts = a*1j
    return wts

def longrange_kernel(kvec, r_split):
  """
  Computes a long range kernel

  Parameters:
  -----------
  kvec: array
    Array of k values in Fourier space

  r_split: float
    TODO: @modichirag add documentation

  Returns:
  --------
  wts: array
    kernel
  """
  if r_split != 0:
    kk = sum(ki ** 2 for ki in kvec)
    return np.exp(-kk * r_split**2)
  else:
    return 1.

def shortrange_kernel(x1,x2,eps_s,split=2):
    """
    Computes short range force "kernel"
    Parameters:
    ----------
    position: array of p

    softening length for kernel: float,

    Returns:
    --------
    force for a single particle - can make more efficient for more particles?

    """
    def srkern(p1,p2,eps_s):
        "Input: single-particle Position  (3), particle you want to get current force on, softening length"
        "Output: single-particle Force (3) - last entry of state"
        "can probably come back and pass p2 as a list of positions...drop one of the inner loops"
        # if(np.any(p2)<-900):
        #     return 0

        disp = p2-p1
        rsq=np.sum((disp**2),axis=0) #simple dist, square in place and sum, sqrt
        r = np.sqrt(rsq)
        gadget_erfc = erfc(r/split/2) + r/(np.sqrt(np.pi)*split) * erfc(-r**2 /split**2 /4) #to send force to zero if too large, but in theory shouldn't be a problem since not computing beyond 1 neighbor anyway

        kernel = (rsq  +eps_s**2)**(-3/2)
        plummer = kernel*disp*gadget_erfc #kernel*disp
        return plummer

    #Idea is to replace this with spline...

    return srkern(x1,x2,eps_s)


def chain_mesh(shape,cm_scale,binLengths=False):
    "Return the chaining mesh for a given (rectangular) subset of the full grid"
    "shape - (nc_x,nc_y,nc_z) "
    "nc/cm_scale should be an integer"
    num_bins_x,num_bins_y,num_bins_z = int(shape[0]/cm_scale),int(shape[1]/cm_scale),int(shape[2]/cm_scale)

    num_bins = num_bins_x*num_bins_y*num_bins_z

    x0,y0,z0 = np.arange(num_bins_x),np.arange(num_bins_y),np.arange(num_bins_z)
    xg,yg,zg = np.meshgrid(x0,y0,z0)

    cm3d = np.concatenate([xg.reshape([num_bins,1]),yg.reshape([num_bins,1]),zg.reshape([num_bins,1])],axis=1)\

    if(binLengths):
        return cm3d,num_bins_x,num_bins_y,num_bins_z
    else:
        return cm3d
