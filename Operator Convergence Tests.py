import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.sparse import spdiags, diags, csr_matrix
from scipy import sparse
import scipy.sparse.linalg
from scipy.interpolate import interp2d
from scipy.sparse.linalg import LinearOperator

import matplotlib.colors as colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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
        # Returns Flattened mesh vector from two vectors
        meshxx, meshzz = np.meshgrid(cx, cz)
        cx = meshxx.flatten(order='C')
        cz = meshzz.flatten(order='C')
        return np.array([cx, cz]).T


    def Divergence_Operator(self):
        '''
        #Retutns the Divergence Operator based on the mesh as sparse matrix
        '''
        Dx = diags([-1, 1], [0, 1], shape=(self.nx, self.nx + 1)).toarray()
        I = np.identity(self.nz)
        DivX = np.kron(I, Dx) / self.hx
        Dz = diags([1, -1], [0, 1], shape=(self.nz, self.nz + 1)).toarray()
        I = np.identity(self.nx)
        DivZ = np.kron(Dz, I) / self.hz
        Div = np.concatenate((DivX, DivZ), axis=1)

        return (Div)

    def Central_diff_grad(self):
        Dx = diags([-1, 1], [-1, 0], shape=(self.nx +1, self.nx)).toarray()
        I = np.identity(nz)
        Gx = np.kron(I, Dx) / self.hx

        Dz = diags([1, -1], [-1, 0], shape=(self.nz + 1, self.nz)).toarray()
        I = np.identity(nx)
        Gz = np.kron(Dz, I) / self.hz
        Grad = np.concatenate((Gx, Gz), axis=0)

        return Grad



#Divergence Convergence Test
step_size = []
error = []
for i in range(2,8,1):
    nx = 2**i
    nz = 2**i
    hx = 1/nx
    hz=1/nz

    #Define mesh of size nx,nz, Div operator
    Mesh = Two_d_tensor_mesh(nx,nz,hx,hz)
    Div = Mesh.Divergence_Operator()

    #Define Functions of x fields, z fields
    x_field_fun = lambda x, z: x*2*z
    z_field_fun = lambda x, z: np.exp(z)

    #Define Analytic solution
    div_analytic = lambda x,z: 2*z+np.exp(z)

    x_field = x_field_fun(Mesh.cx_vec[:,0], Mesh.cx_vec[:,1])
    z_field = z_field_fun(Mesh.cz_vec[:,0], Mesh.cz_vec[:,1])

    vec_field = np.concatenate([x_field, z_field],axis=0)


    #Compare analytic to exact
    num=Div@vec_field
    exact = div_analytic(Mesh.cc_vec[:,0],Mesh.cc_vec[:,1])

    #Append grid size, error
    step_size.append(hx)
    error.append(max(abs(num-exact)))

#Plot Results
plt.loglog(step_size, error, '*')
j = np.polyfit(np.log(step_size), np.log(error), deg=1)
step = np.arange(-5, -1, .1)
y = np.exp(step * j[0] + j[1])
plt.title('Divergence Convergence Test')
plt.xlabel('Spatial Grid Size (h)')
plt.ylabel('Max Error')
plt.plot(np.exp(step), y, label='Max Norm(Numeric - Analytic), slope = ' + str(round(j[0], 3)))
plt.legend()

plt.show()


#Gradient Convergence Test
step_size = []
error = []
for i in range(2,8,1):
    nx = 2**i
    nz = 2**i
    hx = 1/nx
    hz=1/nz

    # Define mesh of size nx,nz, Div operator
    Mesh = Two_d_tensor_mesh(nx,nz,hx,hz)
    Grad = Mesh.Central_diff_grad()

    #Define scalar function for taking solution, analytical solutions in x and z
    scalar_fun = lambda x, z: x*np.cos(z)
    grad_analytic_x = lambda x,z: np.cos(z)
    grad_analytic_z = lambda x,z: -x*np.sin(z)


    ##

    num=Grad@scalar_fun(Mesh.cc_vec[:,0],Mesh.cc_vec[:,1])

    exact_x = grad_analytic_x(Mesh.cx_vec[:, 0], Mesh.cx_vec[:, 1])
    exact_z = grad_analytic_z(Mesh.cz_vec[:, 0], Mesh.cz_vec[:, 1])

    exact = np.hstack([exact_x, exact_z])
    Stacked=np.vstack((Mesh.cx_vec,Mesh.cz_vec))

    #Remove Exterior Edge (Boundary conditions) for test
    remove_edge =np.where(Stacked[:,0]==0)[0]
    remove_edge = np.hstack((remove_edge,np.where(Stacked[:,0]==1.0)[0]))
    remove_edge = np.hstack((remove_edge, np.where(Stacked[:, 1] == 0)[0]))
    remove_edge = np.hstack((remove_edge, np.where(Stacked[:, 1] == -1.0)[0]))

    num[remove_edge] = 0
    exact[remove_edge] = 0

    #Append Errors
    step_size.append(hx)
    error.append(max(abs(num-exact)))

#Plot Result
plt.loglog(step_size, error, '*')
j = np.polyfit(np.log(step_size), np.log(error), deg=1)
step = np.arange(-5, -1, .1)
y = np.exp(step * j[0] + j[1])
plt.title('Gradient Convergence Test')
plt.xlabel('Spatial Grid Size (h)')
plt.ylabel('Max Error')
plt.plot(np.exp(step), y, label='Max Norm(Numeric - Analytic), slope = ' + str(round(j[0], 3)))
plt.legend()

plt.show()