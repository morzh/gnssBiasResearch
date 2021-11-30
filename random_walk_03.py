from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# additional imports
from skimage import color as skolor # see the docs at scikit-image.org/
from skimage import measure
from scipy.ndimage import gaussian_filter


n = 8 # Number of possibly sharp edges
r = .7 # magnitude of the perturbation from the unit circle,
# should be between 0 and 1
N = n*3+1 # number of points in the Path
# There is the initial point and 3 points per cubic bezier curve. Thus, the curve will only pass though n points, which will be the sharp edges, the other 2 modify the shape of the bezier curve

angles = np.linspace(0,2*np.pi,N)
codes = np.full(N,Path.CURVE4)
codes[0] = Path.MOVETO


sigma = 7 # smoothing parameter
verts = np.stack((np.cos(angles), np.sin(angles))).T*(2*r*np.random.random(N)+1-r)[:, None]
verts[-1, :] = verts[0, :] # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
path = Path(verts, codes)

patch = patches.PathPatch(path, facecolor='k', lw=2) # Fill the shape in black
fig = plt.figure(figsize=(8, 8), dpi=200)
ax = fig.add_subplot(111)
# patch = patches.PathPatch(path, facecolor='none', lw=2)
# ax.add_patch(patch)
# ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
# ax.axis('off') # removes the axis to leave only the shape
# ax.scatter(verts[:, 0], verts[:, 1])

# fig.canvas.draw()

##### Smoothing ####
# get the image as an array of values between 0 and 1
data = data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
gray_image = skolor.rgb2gray(data)

# filter the image
smoothed_image = gaussian_filter(gray_image,sigma)

# Retrive smoothed shape as 0.5 contour
smooth_contour = measure.find_contours(smoothed_image[::-1,:], 0.5)[0]
# Note, the values of the contour will range from 0 to smoothed_image.shape[0]
# and likewise for the second dimension, if desired,
# they should be rescaled to go between 0,1 afterwards

# compare smoothed ans original shape
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1,2,1)
patch = patches.PathPatch(path, facecolor='none', lw=2)
ax1.add_patch(patch)
ax1.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
ax1.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
ax1.axis('off') # removes the axis to leave only the shape
ax2 = fig.add_subplot(1,2,2)
ax2.plot(smooth_contour[:, 1], smooth_contour[:, 0], linewidth=2, c='k')
ax2.axis('off')
plt.show()