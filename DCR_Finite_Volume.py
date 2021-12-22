import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.sparse import spdiags, diags, csr_matrix
from scipy import sparse
import scipy.sparse.linalg



class Two_d_tensor_mesh():
    '''
    A Class to create a two-dimensional evenly spaced tensor mesh
    '''

    def __init__(self, nx, nz, hx, hz):
        # Define number of cells and cell spacings
        self.nx = nx
        self.nz = nz
        self.hx = hx
        self.hz = hz

        # Create cell center Matrix
        cx = np.linspace(hx / 2, (hx * nx) - hx / 2, nx)
        cz = np.linspace(-hz / 2, (- hz * nz) + hz / 2, nz)
        self.cc_vec = self.ft(cx, cz)

        # Create cell edge locations in X direction
        cx = np.linspace(0, (hx * nx), nx + 1)
        cz = np.linspace(-hz / 2, (- hz * nz) + hz / 2, nz)
        self.cx_vec = self.ft(cx, cz)

        # Define interior Cells to take gradient for plotting
        cx = np.linspace(0, (hx * nx), nx + -1)
        cz = np.linspace(hz / 2, -((hz * nz) - hz / 2), nz)
        cz = cz - 1
        self.c_int = self.ft(cx, cz)

        # Create cell edge locations in Z direction

        cx = np.linspace(hx / 2, (hx * nx) - hx / 2, nx)
        cz = np.linspace(0, -(hz * nz), nz + 1)
        self.cz_vec = self.ft(cx, cz)
        self.cz_vec = self.ft(cx, cz)
        self.cc_source = []
        self.conductivities = np.zeros_like(self.cc_vec[:, 0])

        #Conductivities for anisotropy
        self.conductivities_xx = np.zeros_like(self.cc_vec[:, 0])
        self.conductivities_yy = np.zeros_like(self.cc_vec[:, 0])
        self.conductivities_xy = np.zeros_like(self.cc_vec[:, 0])
        self.conductivities_yx = np.zeros_like(self.cc_vec[:, 0])
        self.sources = np.zeros_like(self.cc_vec[:, 0])

    def ft(self, cx, cz):
        '''
        Returns Flattened mesh vector from two vectors
        '''
        meshxx, meshzz = np.meshgrid(cx, cz)
        cx = meshxx.flatten(order='C')
        cz = meshzz.flatten(order='C')
        return np.array([cx, cz]).T

    def set_pp(self, conductivity):
        '''
        Takes in conductivity, sets all values in mesh
        '''
        try:
            self.conductivities_xx[:] = conductivity[0]
            self.conductivities_yy[:] = conductivity[1]
            self.conductivities_xy[:] = conductivity[2]
            self.conductivities_yx[:] = conductivity[3]
        except:
            self.conductivities[:] = conductivity


    def set_block(self, conductivity, coords):
        '''
        Set block in mesh to particular value by coordiantes
        '''
        x1, x2, y1, y2 = coords
        c1 = np.argwhere((self.cc_vec[:, 0] < x2))
        c2 = np.argwhere((self.cc_vec[:, 0] > x1))
        c3 = np.argwhere((self.cc_vec[:, 1] > y2))
        c4 = np.argwhere((self.cc_vec[:, 1] < y1))
        c = np.intersect1d(c1, c2)
        c = np.intersect1d(c, c3)
        c = np.intersect1d(c, c4)
        try:
            conxx = np.ones_like(c) * conductivity[0]
            conyy = np.ones_like(c) * conductivity[1]
            conxy = np.ones_like(c) * conductivity[2]
            conyx = np.ones_like(c) * conductivity[3]
            self.conductivities_xx.flat[c] = conxx
            self.conductivities_yy.flat[c] = conyy
            self.conductivities_xy.flat[c] = conxy
            self.conductivities_yx.flat[c] = conyx
        except:
            con = np.ones_like(c) * conductivity
            self.conductivities.flat[c] = con

    def set_sources(self, magnitude_list, location_list):
        '''
        Set Sources of mesh to closest location by magnitude and location list
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

    def Divergence_Operator(self):
        '''
        #Retutns the Divergence Operator based on the mesh as sparse matrix
        '''
        Dx = diags([-1, 1], [0, 1], shape=(self.nx, self.nx + 1)).toarray()
        I = np.identity(nz)
        DivX = np.kron(I, Dx) / self.hx
        Dz = diags([1, -1], [0, 1], shape=(self.nz, self.nz + 1)).toarray()
        I = np.identity(nx)
        DivZ = np.kron(Dz, I) / self.hz
        Div = np.concatenate((DivX, DivZ), axis=1)
        Div = sparse.csr_matrix(Div)

        return (Div)

    def inner_product(self):
        '''
        Returns inner Product of faces
        '''
        Plist = []
        for i in range(2):
            for j in range(2):
                #Create inner product matrices using Kronecker products
                P1x = diags([1], [i], shape=(self.nx, self.nx + 1)).toarray()
                I = np.identity(nz)
                P1x = np.kron(I, P1x) / self.hx
                P1z = diags([0], [i], shape=(self.nz, self.nz + 1)).toarray()
                I = np.identity(nx)
                P1Z = np.kron(P1z, I) / self.hz
                P11 = np.concatenate((P1x, P1Z), axis=1)

                P1x = diags([0], [i], shape=(self.nx, self.nx + 1)).toarray()
                I = np.identity(nz)
                P1x = np.kron(I, P1x) / self.hx
                P1z = diags([1], [j], shape=(self.nz, self.nz + 1)).toarray()
                I = np.identity(nx)
                P1Z = np.kron(P1z, I) / self.hz
                P12 = np.concatenate((P1x, P1Z), axis=1)
                P = np.concatenate((P11, P12), axis=0)
                Plist.append(P)

        #Create inverse conductivity Matrix from tensor
        con_xx = np.diag(1/self.conductivities_xx)
        con_yy = np.diag(1/self.conductivities_yy)
        con_xy = np.diag(1 / self.conductivities_xy)
        con_yx = np.diag(1 / self.conductivities_yx)
        con_xy_l = np.concatenate((con_xx,con_yx))
        con_xy_r = np.concatenate((con_xy, con_yy))
        con_xy = np.concatenate((con_xy_l,con_xy_r),axis=1)

        #Cell "Volumes"
        rt_v = np.sqrt(self.hz*self.hx)

        #Apply operators to conductivity, return inverse of output
        Mf = 0
        for p in Plist:
            Mf = Mf + .25*(p.T @ (rt_v*con_xy*rt_v)) @ p
        Mf = sparse.csc_matrix(Mf)
        Mf_inv=sparse.linalg.inv(Mf)

        return(Mf_inv)


# Define grid cells
nx = 64
nz = 64

# Spacing
hx = 1
hz = 1


Mesh = Two_d_tensor_mesh(nx, nz, hx, hz)


#Initiallizing with zero conductivity for xy,yx sets coordinate wise anisotropy or isotropy
Mesh.set_pp([1,1,1e15,1e15])

#Initialize block with conductivity xmin,xmax, zmax,zmin
#Mesh.set_block([1,1,1,1],[0,64,0,-20])
##
#Return inner product inverse Matrix
Mf_inv=Mesh.inner_product()

#Return Divergence Operator
Div = Mesh.Divergence_Operator()


#Set Sources
Mesh.set_sources([1, -1], [[27.5, -.5], [36.5, -.5]])
source = sparse.csr_matrix(Mesh.sources)

#Solve for Voltage
V = sparse.linalg.spsolve((Div@Mf_inv@Div.T), Mesh.sources)

#Plot Solution

fig, axes = plt.subplots(nrows=1, ncols=1)

im = plt.tricontourf(Mesh.cc_vec[:, 0], Mesh.cc_vec[:, 1], V, 30,cmap='seismic')

plt.colorbar(im)
#plt.title('XX_Conductivity: 1 Siemen, YY_Conductivity: 1000 Siemens')
plt.title('XX,YY Conductivities: 1 Siemen, XY conductivity 1 Siemen')
plt.show()

















