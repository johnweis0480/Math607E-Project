import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, diags, csr_matrix
from scipy import sparse
import scipy.sparse.linalg
from scipy.interpolate import interp2d
from scipy.sparse.linalg import LinearOperator

import matplotlib.colors as colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patches as patches


class Two_d_tensor_mesh():

    def __init__(self, nx, nz, hx, hz):
        # Define number of cells and cell spacings
        self.nx = nx
        self.nz = nz
        self.hx = hx
        self.hz = hz

        #Create cell center Matrix
        cx = np.linspace(hx / 2, (hx * nx) - hx / 2, nx)
        cz = np.linspace(-hz/2, (- hz*nz) +hz/2 , nz)
        self.cc_vec = self.ft(cx,cz)


        # Create cell edge locations in X direction
        cx = np.linspace(0, (hx * nx), nx + 1)
        cz = np.linspace(-hz/2, (- hz*nz) +hz/2 , nz)
        self.cx_vec = self.ft(cx, cz)

        #Define interior Cells to take gradient for plotting
        cx = np.linspace(0, (hx * nx), nx + -1)
        cz = np.linspace(hz / 2, -((hz * nz) - hz / 2), nz)
        self.c_int = self.ft(cx, cz)

        # Create cell edge locations in Z direction
        cx = np.linspace(hx / 2, (hx * nx) - hx / 2, nx)
        cz = np.linspace(0, -(hz * nz), nz + 1)
        self.cz_vec = self.ft(cx, cz)

        self.cc_source = []

        self.conductivities = np.zeros_like(self.cc_vec[:, 0])
        self.sources = np.zeros_like(self.cc_vec[:, 0])

    def ft(self, cx, cz):
        # Returns Flattened mesh vector from two vectors
        meshxx, meshzz = np.meshgrid(cx, cz)
        cx = meshxx.flatten(order='C')
        cz = meshzz.flatten(order='C')
        return np.array([cx, cz]).T

    def set_pp(self, conductivity):
        '''
        Takes in conductivity, sets all values in mesh
        '''
        self.conductivities[:] = conductivity

        air_ind = np.where(self.cc_vec[:, 1] == np.amax(self.cc_vec[:, 1]))
        # self.conductivities[air_ind] = 1*10**-8


    def set_block(self, conductivity, coords):
        '''
        Takes in coordinates and conductivity, sets those values in mesh
        '''
        x1, x2, y1, y2 = coords
        c1 = np.argwhere((self.cc_vec[:, 0] < x2))
        c2 = np.argwhere((self.cc_vec[:, 0] > x1))
        c3 = np.argwhere((self.cc_vec[:, 1] > y2))
        c4 = np.argwhere((self.cc_vec[:, 1] < y1))
        c = np.intersect1d(c1, c2)
        c = np.intersect1d(c, c3)
        c = np.intersect1d(c, c4)
        con = np.ones_like(c) * conductivity
        self.conductivities.flat[c] = con

    def set_sources(self, magnitude_list, location_list):
        '''
        Takes in sources, loops through and sets sources
        '''
        for i in range(len(magnitude_list)):
            x, z = location_list[i]
            xloc = abs(self.cc_vec[:, 0] - x)
            zloc = abs(self.cc_vec[:, 1] - z)
            xloc = np.where(xloc == xloc.min())
            zloc = np.where(zloc == zloc.min())
            loc = np.intersect1d(xloc, zloc)
            self.sources[loc[0]] = magnitude_list[i] / (self.hx * self.hz)
            self.cc_source.append(loc[0])


    def Neumann_Laplacian(self):
        Lx = np.ones((nx, nx))
        Lx[0, 1] = 2
        Lx[-1, -2] = 2
        I = np.ones((nz, nz))
        Lx = np.kron(I, Lx)
        Ly = np.ones((nz, nz))
        Ly[0, 1] = 2
        Ly[-1, -2] = 2
        I = np.ones((nx, nx))
        Ly = np.kron(Ly, I)
        b = (Ly * Lx)
        return b


    def Gradient_operator(self):
        '''
        Operator to take the gradient of Voltage
        Used to plot vector fields
        '''
        Dx = diags([-1, 1], [0, 1], shape=(self.nx - 1, self.nx)).toarray()
        I = np.identity(nz)
        Gx = np.kron(I, Dx) / self.hx
        Dz = diags([-1, 1], [0, 1], shape=(self.nz - 1, self.nz)).toarray()
        I = np.identity(nx)
        Gz = np.kron(Dz, I) / self.hz
        Grad = np.array([Gx, Gz])
        return (Grad)

    def Central_diff_grad(self):
        '''
        Returns gradient operator from centered derivative
        '''
        Dx = diags([-1, 1], [-1, 0], shape=(self.nx +1, self.nx)).toarray()
        I = np.identity(nz)
        Gx = np.kron(I, Dx) / self.hx

        Dz = diags([1, -1], [-1, 0], shape=(self.nz + 1, self.nz)).toarray()
        I = np.identity(nx)
        Gz = np.kron(Dz, I) / self.hz
        Grad = np.concatenate((Gx, Gz), axis=0)

        return Grad

    def Harmonic_Averaging(self):
        '''
        Preforms Harmonic averaging on conductivities, returns values at cell edges
        '''
        har_con = 1 / self.conductivities
        Ax = diags([1, 1], [-1, 0], shape=(self.nx + 1, self.nx)).toarray()
        I = np.identity(nz)
        AvX = np.kron(I, Ax)
        Az = diags([1, 1], [-1, 0], shape=(self.nz + 1, self.nz)).toarray()
        I = np.identity(nx)
        AvZ = np.kron(Az, I)
        H_Avg = np.concatenate((AvX, AvZ), axis=0)
        edge_conductivity = 2 / (H_Avg @ har_con)
        edge_conductivity = sparse.diags(edge_conductivity)
        return (edge_conductivity)

# Define grid spacings
nx = 64
nz = 64

# Spacing
hx = 1
hz = 1

#Create Mesh, Define Operators
Mesh = Two_d_tensor_mesh(nx, nz, hx, hz)
Grad = Mesh.Central_diff_grad()
Div = -Grad.T

#Set conductivity, block if wanted
Mesh.set_pp(1)
#Mesh.set_block([1000],[33,40,-4,-14])

#Preform Harmonic averaging
edge_conductivity = Mesh.Harmonic_Averaging()


Op = -Div @ edge_conductivity @ Grad

Op = csr_matrix(Op)

#Set Sources
Mesh.set_sources([1, -1], [[31.5, -27.5], [32.5, -27.5]])
source = sparse.csr_matrix(Mesh.sources)

#Solve for Voltage
V = sparse.linalg.spsolve(Op, Mesh.sources)

#Plot Solution
fig, ax = plt.subplots()

tcf = ax.tricontourf(Mesh.cc_vec[:, 0], Mesh.cc_vec[:, 1], V, 50,cmap='seismic')
# plt.scatter([25,25,30,30],[-10,-25,-10,-25])
cbar=plt.colorbar(tcf, extend='both')
cbar.set_label('Voltage')
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Background: 1 Siemen, Block: 1000 Siemens')

#plt.title('Background: 1 Siemen')

rect = patches.Rectangle((33, -4), 7, -14, linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()