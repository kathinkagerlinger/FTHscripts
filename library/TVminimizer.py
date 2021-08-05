"""
Python class for TV minimization of a reconstructed image

2020
@author: Erik Malm, Lund University
"""

from scipy.fft import fft2, ifft2, fftshift
import numpy as np
from numpy import gradient
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation



class TVMinimizer(object):
    """
    Class for Fourier holography reconstructions.

    *For inputting full-sized data see: FTH2 class.
    """

    def __init__(self, rec, mask=None, mask_threshold=0.1):
        """
        Images should be fftshifted into corners.

        rec: Cropped reconstruction (ifft2(Intensity)) (fftshifted into corners)
        mask: full-sized mask (fftshifted into corners) 1: measured data, 0 otherwise

        """
        self.u_fft0 = rec/np.max(np.abs(rec))
        self.u_fft = rec/np.max(np.abs(rec))
        self.b = fft2(self.u_fft, norm='ortho')
        self.shape = rec.shape

        if mask is not None:
            self.mask0 = mask > 0.5
            self.mask = mask > 0.5
            self.downsample_mask(mask_threshold = mask_threshold)
        else:
            print('No mask given! The functions will not work correctly!')

    def downsample_mask(self,iterations=4, mask_threshold=0.1, w_outer=5):
        """
        Downsample mask to the size of the cropped data.

        w_outer: outer width to set mask to zero
        """
        # m = resize(fftshift(self.mask_full).astype('float'), self.shape, mode='constant')
        mask = 1 - fftshift(self.mask0).astype('float')
        w = w_outer
        mask[:w,:] = mask[-w:,:] = mask[:,:w] = mask[:,-w:] = 0
        m = resize(binary_dilation(mask,iterations=iterations).astype('float'),
                            self.shape, mode='wrap')
        self.mask = fftshift(m <= mask_threshold)


    def solve_tv(self, iterations, step_size=1e-3, zero_border=True):
        """
        Projected gradient descent.
        iterations:
        step_size:
        """
        if iterations is None:
            print('No iteration number given! Abort!')
        else:
            self.u_tv = fftshift(self.u_fft)
            for a in range(iterations):
                self.u_tv -= step_size * self._tv_grad(self.u_tv)
                self.u_tv = fft2(fftshift(self.u_tv), norm='ortho')
                self.u_tv[self.mask] = self.b[self.mask]
                self.u_tv = fftshift(ifft2(self.u_tv, norm='ortho'))
                if zero_border:
                    self.u_tv[0,:] = self.u_tv[-1,:] = self.u_tv[:,0] = self.u_tv[:,-1] = 0

            self.u_tv = fftshift(self.u_tv)

    def focus_reconstruction(self, a):
        """
        Propagate reconstruction to be in focus

        a: constant used to propagate field
        """
        x00 = np.linspace(-self.shape[0]//2,self.shape[0]//2-1, self.shape[0])
        x01 = np.linspace(-self.shape[1]//2,self.shape[1]//2-1, self.shape[1])
        x0, x1 = np.meshgrid(x00,x01)
        r = np.sqrt(np.square(x0) + np.square(x1))
        H = fftshift(np.exp(1j * a * r**2))
        self.u_fft = ifft2(fft2(self.u_fft0)*H)


    def solve_l1(self):
        pass


    def _tv_grad(self, u):
        """
        Total variation gradient for solving TV min problem.
        """
        rec = fftshift(u)
        grad_u = np.array(gradient(u))
        grad_mag = np.sqrt(np.sum(np.square(np.abs(grad_u)), axis=(0)))
        grad_mag = np.maximum(1e-16, grad_mag)
        tv_grad0 = np.array(gradient(grad_u[0]/grad_mag))
        tv_grad1 = np.array(gradient(grad_u[1]/grad_mag))
        grad_f = tv_grad0[0] + tv_grad1[1]
        return -grad_f
