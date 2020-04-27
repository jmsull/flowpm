""" Core FastPM elements"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from astropy.cosmology import Planck15

from .utils import white_noise, c2r3d, r2c3d, cic_paint, cic_readout
from .kernels import fftk, laplace_kernel, gradient_kernel, longrange_kernel
from .kernels import shortrange_kernel, chain_mesh
from .background import MatterDominated

__all__ = ['linear_field', 'lpt_init', 'nbody']

PerturbationGrowth = lambda cosmo, *args, **kwargs: MatterDominated(Omega0_lambda = cosmo.Ode0,
                                                                    Omega0_m = cosmo.Om0,
                                                                    Omega0_k = cosmo.Ok0,
                                                                    *args, **kwargs)


def linear_field(nc, boxsize, pk, batch_size=1,
                 kvec=None, seed=None, name=None, dtype=tf.float32):
  """Generates a linear field with a given linear power spectrum

  Parameters:
  -----------
  nc: int
    Number of cells in the field

  boxsize: float
    Physical size of the cube, in Mpc/h TODO: confirm units

  pk: interpolator
    Power spectrum to use for the field

  kvec: array
    k_vector corresponding to the cube, optional

  Returns
  ------
  linfield: tensor (batch_size, nc, nc, nc)
    Realization of the linear field with requested power spectrum
  """

  with tf.name_scope(name, "LinearField"):
    if kvec is None:
      kvec = fftk((nc, nc, nc), symmetric=False)
    kmesh = sum((kk / boxsize * nc)**2 for kk in kvec)**0.5
    pkmesh = pk(kmesh)

    whitec = white_noise(nc, batch_size=batch_size, seed=seed, type='complex')
    lineark = tf.multiply(whitec, (pkmesh/boxsize**3)**0.5)
    linear = c2r3d(lineark, norm=nc**3, name=name, dtype=dtype)
    return linear

def lpt1(dlin_k, pos, kvec=None, name=None):
  """ Run first order LPT on linear density field, returns displacements of particles
      reading out at q. The result has the same dtype as q.

  Parameters:
  -----------
  dlin_k: TODO: @modichirag add documentation

  Returns:
  --------
  displacement: tensor (batch_size, npart, 3)
    Displacement field
  """
  with tf.name_scope(name, "LPT1", [dlin_k, pos]):
    shape = dlin_k.get_shape()
    batch_size, nc = shape[0], shape[1].value
    if kvec is None:
      kvec = fftk((nc, nc, nc), symmetric=False)

    lap = tf.cast(laplace_kernel(kvec), tf.complex64)

    displacement = []
    for d in range(3):
      kweight = gradient_kernel(kvec, d) * lap
      dispc = tf.multiply(dlin_k, kweight)
      disp = c2r3d(dispc, norm=nc**3)
      displacement.append(cic_readout(disp, pos))
    displacement = tf.stack(displacement, axis=2)
    return displacement
#
def lpt2_source(dlin_k, kvec=None, name=None):
  """ Generate the second order LPT source term.

  Parameters:
  -----------
  dlin_k: TODO: @modichirag add documentation

  Returns:
  --------
  source: tensor (batch_size, nc, nc, nc)
    Source term
  """
  with tf.name_scope(name, "LPT2Source", [dlin_k]):
    shape = dlin_k.get_shape()
    batch_size, nc = shape[0], shape[1].value
    if kvec is None:
      kvec = fftk((nc, nc, nc), symmetric=False)
    source = tf.zeros(tf.shape(dlin_k))
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]

    phi_ii = []
    # diagnoal terms
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)

    for d in range(3):
        grad = gradient_kernel(kvec, d)
        kweight = lap * grad * grad
        phic = tf.multiply(dlin_k, kweight)
        phi_ii.append(c2r3d(phic, norm=nc**3))

    for d in range(3):
        source = tf.add(source, tf.multiply(phi_ii[D1[d]], phi_ii[D2[d]]))

    # free memory
    phi_ii = []

    # off-diag terms
    for d in range(3):
        gradi = gradient_kernel(kvec, D1[d])
        gradj = gradient_kernel(kvec, D2[d])
        kweight = lap * gradi * gradj
        phic = tf.multiply(dlin_k, kweight)
        phi = c2r3d(phic, norm=nc**3)
        source = tf.subtract(source, tf.multiply(phi, phi))

    source = tf.multiply(source, 3.0/7.)
    return r2c3d(source, norm=nc**3)

def lpt_init(linear, a0, order=2, cosmology=Planck15, kvec=None, name=None):
  """ Estimate the initial LPT displacement given an input linear (real) field

  Parameters:
  -----------
  TODO: documentation
  """
  with tf.name_scope(name, "LPTInit", [linear]):
    assert order in (1, 2)
    shape = linear.get_shape()
    batch_size, nc = shape[0], shape[1].value

    dtype = np.float32
    Q = np.indices((nc, nc, nc)).reshape(3, -1).T.astype(dtype)
    Q = np.repeat(Q.reshape((1, -1, 3)), batch_size, axis=0)
    pos = Q

    a = a0

    lineark = r2c3d(linear, norm=nc**3)

    pt = PerturbationGrowth(cosmology, a=[a], a_normalize=1.0)
    DX = tf.multiply(dtype(pt.D1(a)) , lpt1(lineark, pos))
    P = tf.multiply(dtype(a ** 2 * pt.f1(a) * pt.E(a)) , DX)
    F = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf(a) / pt.D1(a)) , DX)
    if order == 2:
      DX2 = tf.multiply(dtype(pt.D2(a)) , lpt1(lpt2_source(lineark), pos))
      P2 = tf.multiply(dtype(a ** 2 * pt.f2(a) * pt.E(a)) , DX2)
      F2 = tf.multiply(dtype(a ** 2 * pt.E(a) * pt.gf2(a) / pt.D2(a)) , DX2)
      DX = tf.add(DX, DX2)
      P = tf.add(P, P2)
      F = tf.add(F, F2)

    X = tf.add(DX, Q)
    return tf.stack((X, P, F), axis=0)

def apply_longrange(x, delta_k, split=0, factor=1, kvec=None, name=None):
  """ like long range, but x is a list of positions
  TODO: Better documentation, also better name?
  """

  with tf.name_scope(name, "ApplyLongrange", [x, delta_k]):
    shape = delta_k.get_shape()
    batch_size, nc = shape[1], shape[2].value

    if kvec is None:
      kvec = fftk((nc, nc, nc), symmetric=False)

    ndim = 3
    norm = nc**3
    lap = tf.cast(laplace_kernel(kvec), tf.complex64)
    fknlrange = longrange_kernel(kvec, split)
    kweight = lap * fknlrange
    pot_k = tf.multiply(delta_k, kweight)

    f = []
    for d in range(ndim):
      force_dc = tf.multiply(pot_k, gradient_kernel(kvec, d))
      forced = c2r3d(force_dc, norm=norm)
      force = cic_readout(forced, x)
      f.append(force)

    f = tf.stack(f, axis=2)
    f = tf.multiply(f, factor)
    return f


def apply_shortrange(state,nc,cm_scale=4, eps_s=.05, split=2, name=None,#x,nc,cm_scale=4, eps_s=.05, name=None):
                     nsubcycles=1): #adding nsubcycles for eta comparison

  '''
  Note that everything is in numpy right now for testing - using TF at the moment is too slow to run in my dumb implementation of loops.
  Input: x - positions, tensor (batch_size, npart,3) - Right now taking state for debug using but put it back to x later
         nc - require this to be passed to compute chaining mesh, might want to pass a subset of particles != npart
  Optional: cutoff - where to start the chaining mesh in grid units
            eps_s - softening length in grid units

  Output: force_s - short range force updates, tensor (batch_size,npart,3)

  '''

  '''Also put this back to [x] when done debugging'''
  with tf.name_scope(name, "ApplyShortrange", [state]): #[x]):
      x,p,f = state[0],state[1],state[2]
      shape = x.get_shape()
      batch_size,npart = shape[0].value,shape[1].value



      #Have to do this to be able to do the particle computation, if you are forced to use
      #TF tensors when constructing bins (i.e. without calling x to get bin values), it is
      #excruciatingly slow (10^4 x slower than non-TF numpy for loop) due to graph additions in every loop - need to figure this out...

      '''For debgging/diagnostic'''
      with tf.Session() as sess:
          x_eval = x.eval()
          p_eval = p.eval()
          f_eval = f.eval()

      with tf.Session() as sess: print('min x={0}, max x={1}'.format(np.max(x_eval),
                                                                               np.min(x_eval)
                                                                               ))

      x_eval = x_eval % nc

      with tf.Session() as sess: print('Post mod (what SR sees) min x={0}, max x={1}'.format(np.max(x_eval),
                                                                               np.min(x_eval)
                                                                               ))

      #bin the particles
      num_bins_1d = int(nc/cm_scale)
      num_bins = num_bins_1d**2
      bin_size=nc/num_bins_1d
      nc_x,nc_y,nc_z = nc,nc,nc


      cm_cells,nbx,nby,nbz = chain_mesh((nc_x,nc_y,nc_z),cm_scale,binLengths=True) #assume full box for now, but could e.g. split along z axis an make nc_z shorter than others


      #probably a better way to do this in one line but...
      neighbor_stencil_array = np.array([
                                       [-1, -1, -1],[-1, 0, -1],[-1, 1, -1],
                                       [0, -1, -1],[0, 0, -1],[0, 1, -1],
                                       [1, -1, -1],[1, 0, -1],[1, 1, -1],

                                       [-1, -1, 0],[-1, 0, 0],[-1, 1, 0],
                                       [0, -1, 0],[0, 0, 0],[0, 1, 0],
                                       [1, -1, 0],[1, 0, 0],[1, 1, 0],

                                       [-1, -1, 1],[-1, 0, 1],[-1, 1, 1],
                                       [0, -1, 1],[0, 0, 1],[0, 1, 1],
                                       [1, -1, 1],[1, 0, 1],[1, 1, 1]
                                       ],dtype=np.int32)

      #probably some better ways to do this with broadcasting...
      f_batch = [] #to store the total sr force per batch

      '''Loop over '''
      for batch_idx in range(batch_size): #presumably we will not run multiple batches with particles on?

          bin_array = ((x_eval[batch_idx]% nc)/bin_size).astype(np.int32) #send negative x to other side where they belong

          #Hockney and Eastwood Chaining mesh - for now will store bin locations as vectors but can come back and set C index
          HOC_array = np.zeros((nbx,nby,nbz),dtype=np.int32)
          bin_count_array = np.copy(HOC_array)

          LL = np.zeros((npart),dtype=np.int32) #linked list for particles in the same bin within the same bin, 0 o.w.


          '''Not TF - don't need it for bin bookkeeping'''
          #fill the chaining mesh
          for i in range(npart):
              bin_vector = bin_array[i]

              LL[i] = (HOC_array[bin_vector[0],bin_vector[1],bin_vector[2]])
              HOC_array[bin_vector[0],bin_vector[1],bin_vector[2]] = i+1
              bin_count_array[bin_vector[0],bin_vector[1],bin_vector[2]]+=1

          f_all = [] #to store the total sr force for one particle

          p_rms,p_max,f_max =[],[],[] #debug lists


          for b in cm_cells: #loop over CM cells
              p_tmp,f_tmp =[],[] #debug lists

              for k in range(bin_count_array[b[0],b[1],b[2]]): #loop once over all particles via bins * particles_per_bin ~ O(N)
                  #need extra factor of 1 because HOC starts at 1

                  idx1 = LL[HOC_array[b[0],b[1],b[2]]-k-1] #get the index of the "head" particle of this CM cell, decrement until run out of particles

                  #debug info for checking timestep
                  p_tmp.append(p_eval[batch_idx,idx1])
                  f_tmp.append(f_eval[batch_idx,idx1])

                  f_neighbor = [] #to store force to be computed on this particle due to neighbor particles
                  for j in range(27):
                      nb = (b+neighbor_stencil_array[j])

                      #PBCs
                      nb = nb % num_bins_1d

                      for m in range(bin_count_array[nb[0],nb[1],nb[2]]): #loop once over all neighbor particles via neighbor_cell * particles_per_cell~ O(27*parts_per_cm_cell)
                            idx2 = LL[HOC_array[nb[0],nb[1],nb[2]]-m -1]
                            f_neighbor.append(shortrange_kernel(x_eval[batch_idx,idx1] % nc ,x_eval[batch_idx,idx2] % nc,eps_s,split=split)) #actually evaluate force between 2 particles

                  f_single = np.stack(f_neighbor,axis=0)
                  f_all.append(np.sum(f_single,axis=0)) #sum force on a single particle

         #timestep debug statements ---from here /*...
              if(len(p_tmp)>0):
                  p_rms.append(np.sqrt(np.mean(np.sum(np.stack(p_tmp,axis=0)**2,axis=1),axis=0)))
                  p_max.append(np.max(np.sqrt(np.stack(p_tmp,axis=0)**2)))
                  f_max.append(np.max(np.sqrt(np.stack(f_tmp,axis=0)**2)))
              else:
                  p_rms.append(-1)
                  p_max.append(-1)
                  f_max.append(-1)
          p_rms = np.stack(p_rms,axis=0)
          p_max = np.stack(p_max,axis=0)
          f_max = np.stack(f_max,axis=0)

          def_step = 0.1 #the default a value of 0.1 for stages, used for
          sr_step = def_step/nsubcycles

          #for our fixed step, the equivalent value of eta and zeta
          eta_def = def_step/(p_rms/f_max)
          zeta_def = def_step*(p_max/bin_size)
          eta = sr_step/(p_rms/f_max)
          zeta = sr_step*(p_max/bin_size)

          print('global max eta for default timestep is {0:.2f}, and for sr timestep is {1:.2f}'.format(np.max(eta_def),np.max(eta)))
          print('global max zeta for default timestep is {0:.2f} and for sr timestep is {1:.2f}'.format(np.max(zeta_def),np.max(zeta)))

          #to here...*/

          #the TF version
          #for b in cm_cells:
          # for bidx in range(cm_cells.shape[0]):
          #     b = bin_tensor[batch_idx,bidx]
          #     for k in range(tf.gather_nd(HOC,b)): #loop once over all particles via bins * particles_per_bin ~ O(N)
          #         idx1 = LL[tf.gather_nd(HOC,b)-k]
          #         f_single = []
          #         for j in range(27):
          #             nb = tf.add(b,neighbor_stencil[j])
          #             for m in range(tf.gather_nd(HOC,nb)): #loop once over all neighbor particles via neighbor_bins * particles_per_bin ~ O(1)
          #                   idx2 = LL[tf.gather_nd(HOC,nb)-m]
          #                   f_single.append(shortrange_kernel(x[batch_idx,idx1],x[batch_idx,idx2],eps_s))
          #         f_all.append(tf.sum(f_single,axis=0)) #sum along npart axis, not along coords
          #f = tf.stack(f_all, axis=1) #stack along 2nd index, get (npart,3)


          print('bin distr summary (for rough idea of load balance): min={0}, max={1}, mean={2:.2f}, median={3}, total/expected_total={4}'.format(
                np.min(bin_count_array),np.max(bin_count_array),np.mean(bin_count_array),np.median(bin_count_array),np.sum(bin_count_array)/npart))

          f_batch.append(np.stack(f_all,axis=0))

      force_s = np.stack(f_batch,axis=0) \

      #tf force_s = tf.stack(f_batch,axis=0)

      return force_s

def kick(state, ai, ac, af, cosmology=Planck15, dtype=np.float32, name=None,
         **kwargs):
  """Kick the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  with tf.name_scope(name, "Kick", [state]):
    pt = PerturbationGrowth(cosmology, a=[ai, ac, af], a_normalize=1.0)
    fac = 1 / (ac ** 2 * pt.E(ac)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ac) #fastPM kick
    indices = tf.constant([[1]])
    update = tf.expand_dims(tf.multiply(dtype(fac), state[2]), axis=0)
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state

def drift(state, ai, ac, af, cosmology=Planck15, dtype=np.float32,
          name=None, **kwargs):
  """Drift the particles given the state

  Parameters
  ----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3)

  ai, ac, af: float
  """
  with tf.name_scope(name, "Drift", [state]):
    pt = PerturbationGrowth(cosmology, a=[ai, ac, af], a_normalize=1.0)
    fac = 1. / (ac ** 3 * pt.E(ac)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ac) #fastPM drift
    indices = tf.constant([[0]])
    update = tf.expand_dims(tf.multiply(dtype(fac), state[1]), axis=0)
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    state = tf.add(state, update)
    return state

def force(state, nc, cosmology=Planck15, pm_nc_factor=1, kvec=None,
          dtype=np.float32,
          short_range=False, cm_scale=4,eps_s=0.05,split=2,
          name=None, **kwargs):
  """
  Estimate force on the particles given a state.

  Parameters:
  -----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3) #where does this leading 3 come from??

  boxsize: float
    Size of the simulation volume (Mpc/h) TODO: check units

  cosmology: astropy.cosmology
    Cosmology object

  pm_nc_factor: int
    TODO: @modichirag please add doc
  """
  with tf.name_scope(name, "Force", [state]):
    shape = state.get_shape()
    batch_size = shape[1]
    ncf = nc * pm_nc_factor

    rho = tf.zeros((batch_size, ncf, ncf, ncf))
    wts = tf.ones((batch_size, nc**3))
    nbar = nc**3/ncf**3

    rho = cic_paint(rho, tf.multiply(state[0], pm_nc_factor), wts)
    rho = tf.multiply(rho, 1./nbar)
    delta_k = r2c3d(rho, norm=ncf**3)
    fac = dtype(1.5 * cosmology.Om0)
    update = apply_longrange(tf.multiply(state[0], pm_nc_factor), delta_k, split=split, factor=fac)

    '''Short-range force correction goes here.'''
    #debugging to see how big accel in SR is vs LR
    with tf.Session() as sess:
        print('dbug max LR force is', tf.reduce_max(update).eval())

    if(short_range):
        '''Put this back to only passing state[0]  (x) when done debugging'''
        update_short = apply_shortrange(state,nc,cm_scale=cm_scale,eps_s=eps_s,split=split)

        update = tf.add(update, update_short)
        with tf.Session() as sess:
            print('dbug max SR force is', tf.reduce_max(update_short).eval())


    update = tf.expand_dims(update, axis=0)

    indices = tf.constant([[2]])
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    mask = tf.stack((tf.ones_like(state[0]), tf.ones_like(state[0]), tf.zeros_like(state[0])), axis=0)

    state = tf.multiply(state, mask)
    state = tf.add(state, update)
    return state

def force_sr(state, nc, cosmology=Planck15,
          dtype=np.float32,
          cm_scale=4,eps_s=0.05,split=2,
          nsubcycles=1, #again, add subcycles for debug eta
          name=None, **kwargs):
  """
  Short-range force, only calls short-range function.

  Parameters:
  -----------
  state: tensor
    Input state tensor of shape (3, batch_size, npart, 3) #where does this leading 3 come from??

  boxsize: float
    Size of the simulation volume (Mpc/h) TODO: check units

  cosmology: astropy.cosmology
    Cosmology object

  pm_nc_factor: int
    TODO: @modichirag please add doc
  """
  with tf.name_scope(name, "Force", [state]):
    shape = state.get_shape()
    batch_size = shape[1]

    '''Short-range force correction goes here.'''

    '''Put this back to only passing state[0]  (x) when done debugging'''
    update_short = apply_shortrange(state,nc,cm_scale=cm_scale,eps_s=eps_s,nsubcycles=nsubcycles,split=split)

    with tf.Session() as sess:
        print('dbug max SR force is', tf.reduce_max(update_short).eval())

    update = tf.expand_dims(update_short, axis=0)

    indices = tf.constant([[2]])
    shape = state.shape
    update = tf.scatter_nd(indices, update, shape)
    mask = tf.stack((tf.ones_like(state[0]), tf.ones_like(state[0]), tf.zeros_like(state[0])), axis=0)

    state = tf.multiply(state, mask)
    state = tf.add(state, update)
    return state

def nbody(state, stages, nc, cosmology=Planck15, pm_nc_factor=1,split=2,
          short_range=False, cm_scale=4,eps_s=0.05,nsubcycles = 2,
          name=None):
  """
  Integrate the evolution of the state across the givent stages

  Parameters:
  -----------
  state: tensor (3, batch_size, npart, 3)
    Input state

  stages: array
    Array of scale factors

  nc: int
    Number of cells

  pm_nc_factor: int
    Upsampling factor for computing

  Returns
  -------
  state: tensor (3, batch_size, npart, 3)
    Integrated state to final condition
  """
  with tf.name_scope(name, "NBody", [state]):
    shape = state.get_shape()

    # Unrolling leapfrog integration to make tf Autograph happy
    if len(stages) == 0:
      return state

    ai = stages[0] #stages is just an array of a values to timestep through

    # first force calculation for jump starting
    state = force(state, nc, pm_nc_factor=pm_nc_factor, cosmology=cosmology,short_range=False,cm_scale=cm_scale,eps_s=eps_s,split=split) #initial shape will be (3, batch, nparts, 3)

    x, p, f = ai, ai, ai #state keeps a running list of position, momentum, and force coordinates in x,y,z for each batch
    # Loop through the stages
    for i in range(len(stages) - 1):
        a0 = stages[i] #current timestep
        a1 = stages[i + 1] #upcoming timestep
        ah = (a0 * a1) ** 0.5 #geometric mean - see fastPM

        #Long range kick
        state = kick(state, p, f, ah, cosmology=cosmology) #kick at long range timestep, remains unchanged
        p = ah

        if(short_range):
            #now split into substeps (say 2)
            print("in short range")
            sub_stages = np.linspace(a0,a1,nsubcycles+1)
            print('substages', sub_stages)
            x_sr,p_sr,f_sr = x,p,f

            for ss in range(nsubcycles-1):
                print("subcycling: step={0} of {1}".format(ss+1,nsubcycles))
                print('beginning - x_sr={0:.3f},p_sr={1:.3f},f_sr={2:.3f}'.format(x_sr,p_sr,f_sr))

                a0_s = sub_stages[ss] #current timestep
                a1_s = sub_stages[ss + 1] #upcoming timestep
                ah_s = (a0_s * a1_s) ** 0.5 #half step for sr kick

                #short range sub-kick at half step - ideally only want to kick those that need it but leaving that alone for now-will need to define function that only kicks some
                state = kick(state, p_sr, f_sr, ah_s, cosmology=cosmology) #kick at short-range timestep,
                p_sr = ah_s

                #short-range sub-drift at full sub step
                state = drift(state, x_sr, p_sr, a1_s, cosmology=cosmology) #drift everyone at SR timestep
                x_sr = a1_s

                #need to corral stray particles back inside the box (the absolute distance matters for the p-p kernel, can also go back and change the kernel)

                #short-range force and subsequent kick at half step
                state = force_sr(state, nc,cosmology=cosmology,cm_scale=cm_scale,eps_s=eps_s,nsubcycles=nsubcycles,split=split) #update force at SR
                f_sr = a1_s

                state = kick(state, p_sr, f_sr, a1_s, cosmology=cosmology) #kick again at short range timestep
                p_sr =  a1_s

                print('end - x_sr={0:.3f},p_sr={1:.3f},f_sr={2:.3f}'.format(x_sr,p_sr,f_sr))


        #Long range force on long timestep from stages
        state = force(state, nc, pm_nc_factor=pm_nc_factor,cosmology=cosmology,short_range=False,cm_scale=cm_scale,eps_s=eps_s,split=split) #update force at SR
        f = a1
        #long range kick again
        state = kick(state, p, f, a1, cosmology=cosmology) #kick at long range timestep, remains unchanged
        p = a1




    return state
