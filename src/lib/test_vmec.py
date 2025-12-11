import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
import os
import f90nml
import argparse

class VmecReader():
    """
    This Class reads a vmec wout file.

    It computes geometric quantities such as surface area
    """

    def __init__(self, fin):

        from netCDF4 import Dataset
        f = Dataset(fin, mode='r')

        def get(f, key):
            return f.variables[key][:]

        # 0D array
        self.nfp = get(f, 'nfp')
        self.ns = get(f, 'ns')
        self.mnmax = get(f, 'mnmax')
        self.aminor = get(f, 'Aminor_p')
        self.Rmajor = get(f, 'Rmajor_p')
        self.volume = get(f, 'volume_p')
        self.volavgB = get(f, 'volavgB')
        self.B0 = get(f, 'b0')

        # 1D array
        self.xm = get(f, 'xm')
        self.xn = get(f, 'xn')

        self.iotaf = get(f, 'iotaf')
        self.presf = get(f, 'presf')

        self.s = np.linspace(1e-10, 1, self.ns)
        self.rho = np.sqrt(self.s)

        # 2D array (order of indices is rmnc[i_s,i_mn] etc)
        self.rmnc = np.array(get(f, 'rmnc'))
        self.zmns = np.array(get(f, 'zmns'))
        self.lmns = np.array(get(f, 'lmns'))
        self.bmnc = np.array(get(f, 'bmnc'))
        self.bsupumnc = np.array(get(f, 'bsupumnc'))
        self.bsupvmnc = np.array(get(f, 'bsupvmnc'))
        self.gmnc = np.array(get(f, 'gmnc'))  # this is sqrt(g), the jacobian

        # Get B field
        self.phi = get(f, 'phi')  # toroidal flux in SI webers (signed)
        phiedge = self.phi[-1]
        Ba = phiedge / (np.pi * self.aminor**2)  # GX normalizes Btor = phi/ pi a2
        self.Ba_vmec = np.abs(Ba)

        # save
        # self.data = # f storing netcdf dataset in VmecReader class causes issues with storing the geo object in the log file.
        self.N_modes = len(self.xm)
        self.filename = fin

        # dV/drho = ds/drho dV/ds = 2 rho dV/ds because s = rho**2
        # dV/ds = int dtheta dphi |sqrt(g)| = 4 pi^2 |sqrt(g_00)|
        self.dVdrho = 2*self.rho*4*np.pi*np.pi*np.abs(self.gmnc[:, 0])  # This is only used locally so unnormalized def is ok
        self.compute_area_grad_rho()

    def compute_area_grad_rho(self, N_phi=32, N_theta=32):
        """
        compute surface area and <|grad(rho)|> on each flux surface
        This code follows the implementation of simsopt's vmec_diagnostics.vmec_compute_geometry
        """

        # set up toroidal and poloidal angles
        theta_vmec = np.linspace(0, 2*np.pi, N_theta)
        phi = np.linspace(0, 2*np.pi/self.nfp, N_phi)
        # make theta and phi 3D arrays
        theta_vmec = np.kron(np.ones((self.ns, 1, N_phi)), theta_vmec.reshape(1, N_theta, 1))
        phi = np.kron(np.ones((self.ns, N_theta, 1)), phi.reshape(1, 1, N_phi))

        # local variables for shorthand
        # mnmax = self.mnmax
        xm = self.xm
        xn = self.xn
        rmnc = self.rmnc
        zmns = self.zmns
        # gmnc = self.gmnc

        # Now that we know theta_vmec, compute all the geometric quantities
        angle = xm[:, None, None, None] * theta_vmec[None, :, :, :] - xn[:, None, None, None] * phi[None, :, :, :]
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        mcosangle = xm[:, None, None, None] * cosangle
        ncosangle = xn[:, None, None, None] * cosangle
        msinangle = xm[:, None, None, None] * sinangle
        nsinangle = xn[:, None, None, None] * sinangle
        # Order of indices in cosangle and sinangle: mn, s, theta, phi
        # Order of indices in rmnc, bmnc, etc: s, mn
        R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
        d_R_d_theta_vmec = -np.einsum('ij,jikl->ikl', rmnc, msinangle)
        d_R_d_phi = np.einsum('ij,jikl->ikl', rmnc, nsinangle)

        # Z = np.einsum('ij,jikl->ikl', zmns, sinangle)
        d_Z_d_theta_vmec = np.einsum('ij,jikl->ikl', zmns, mcosangle)
        d_Z_d_phi = -np.einsum('ij,jikl->ikl', zmns, ncosangle)

        # *********************************************************************
        # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
        # components of the gradient basis vectors using the dual relations:
        # *********************************************************************
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        # X = R * cosphi
        d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
        d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
        # Y = R * sinphi
        d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
        d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi

        grad_s_X_sqrtg = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi)
        grad_s_Y_sqrtg = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi)
        grad_s_Z_sqrtg = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi)

        grad_s_dot_grad_s_g = grad_s_X_sqrtg * grad_s_X_sqrtg + grad_s_Y_sqrtg * grad_s_Y_sqrtg + grad_s_Z_sqrtg * grad_s_Z_sqrtg
        abs_grad_s_sqrtg = np.sqrt(grad_s_dot_grad_s_g)  # this is effectively dA

        dtheta = 2*np.pi / N_theta
        dphi = 2*np.pi / N_phi / self.nfp
        self.area = np.zeros(self.ns)
        self.grad_rho = np.zeros(self.ns)

        # integrate dA = | grad s | sqrt(g) to get surface area
        self.area = np.sum(abs_grad_s_sqrtg * dtheta * dphi, axis=(1, 2)) * self.nfp

        # <|grad rho|> = A / (dV/drho)
        self.grad_rho[1:] = self.area[1:]/self.dVdrho[1:]

    def fourier2space(self, Cmn, tax, pax, s_idx=48, sine=True):
        """
        Taking Fourier modes CMN, selects for flux surface s_idx
        select sine or cosine for array
        input toroidal and poloidal angle axis (tax, pax)
        outputs 2D array Z(p, t)
        """

        arr = []
        for j in np.arange(self.N_modes):

            m = int(self.xm[j])
            n = int(self.xn[j])

            c = Cmn[s_idx, j]

            if (sine):
                A = [[c * np.sin(m*p - n*t) for t in tax] for p in pax]
            else:
                A = [[c * np.cos(m*p - n*t) for t in tax] for p in pax]

            arr.append(A)

        return np.sum(arr, axis=0)

    def get_xsection(self, N, phi=0, s=-1):
        '''
        Gets a poloidal cross section at const toroidal angle phi
        '''

        pax = np.linspace(0, np.pi*2, N)  # poloidal
        tax = np.array([phi])             # toroidal

        # positions
        R2d = self.fourier2space(self.rmnc, tax, pax, sine=False, s_idx=s)
        Z2d = self.fourier2space(self.zmns, tax, pax, sine=True, s_idx=s)

        # cartesian coordinates for flux surface
        R = R2d[:, 0]
        Z = Z2d[:, 0]

        return R, Z

    def getLambda(self, theta, zeta=0, s_idx=100):
        '''
        from (theta_v, zeta_v) compute lambda
        on surface s_idx

        assume both theta,zeta are scalars
        in vmec input coordiantes
        ==
        can I make this flexible for vec theta, vec zeta, or both?
        '''

        # sum over Fourier modes
        x = self.xm*theta - self.xn*zeta
        L = np.sum(self.lmns[s_idx] * np.sin(x))
        return L

    def invertTheta(self, thetaStar, zeta=0, N_interp=50, s_idx=100):
        '''
        This function finds the theta
        that satisfies theta* for a given zeta
        and surface s_idx.

        It does so using root finding
        on an interpolated function
        with N_interp points.
        '''

        # define theta range to straddle 0
        tax = np.linspace(-np.pi, np.pi, N_interp)

        # Given: theta* = theta + lambda
        # compute: f = RHS - theta*
        RHS = [t + self.getLambda(t, zeta=zeta, s_idx=s_idx) for t in tax]
        f = np.array(RHS) - thetaStar

        # check for wrap around
        offset = 0
        while np.min(f) > 0:
            f = f - np.pi
            offset = offset - np.pi
        while np.max(f) < 0:
            f = f + np.pi
            offset = offset + np.pi

        # find root from interpolated function
        func = interp1d(tax, f)
        theta = root_scalar(func, method='toms748', bracket=(-np.pi, np.pi))['root']
        return theta - offset

    def getFieldline(self, zax, alpha=0, s_idx=100):
        '''
        Follow 3D fieldline alpha along input zeta array zax
        on surface s_idx.

        Uses lambda to compute a 1D theta array
        that corresponds to zeta.

        Returns 1D arrays for X, Y, Z.
        '''

        # for each zeta, find theta*
        tax = []
        iota = self.iotaf[s_idx]
        for zeta in zax:
            thetaStar = alpha + iota*zeta
            theta = self.invertTheta(thetaStar, zeta=zeta, s_idx=s_idx)
            tax.append(theta)

        # get R,Z from (zeta,theta*)
        cos = []
        sin = []
        N = len(zax)
        for j in np.arange(N):
            x = self.xm*tax[j] - self.xn*zax[j]
            cos.append(np.cos(x))
            sin.append(np.sin(x))
        R = np.sum(self.rmnc[s_idx] * cos, axis=-1)
        Z = np.sum(self.zmns[s_idx] * sin, axis=-1)

        # get XYZ
        X = R*np.cos(zax)
        Y = R*np.sin(zax)
        return X, Y, Z

    def getSurfaceMesh(self, Nzeta=10, Ntheta=12, s_idx=100, full_torus=False,
                       zeta_zero_mid=False, theta_zero_mid=False):
        '''
        Computes XYZ surface shape
        from Ntheta points
        and Nzeta points, per period
        on surface s_idx.

        If zeta_zero_mid is True, zeta axis straddles zeta=0
        else zeta goes from (0,2pi/Nfp).

        If theta_zero_mid is True, theta (-pi,pi)
        Else (0,2pi).

        Returns a 2D array for X Y and Z
        Saves a corresponding 2D array of Lambda as self.L2
        '''

        # establish input coordinates
        if theta_zero_mid:
            tax = np.linspace(-np.pi, np.pi, Ntheta)
        else:
            tax = np.linspace(0, np.pi*2, Ntheta)
        z0 = np.pi/self.nfp
        if zeta_zero_mid:
            zax = np.linspace(-z0, z0, Nzeta)
        else:
            zax = np.linspace(0, 2*z0, Nzeta)

        # compute m and n modes
        cos = []
        sin = []
        for zeta in zax:
            for theta in tax:
                x = self.xm*theta - self.xn*zeta
                cos.append(np.cos(x))
                sin.append(np.sin(x))

        # sum of Fourier amplitudes
        R = np.sum(self.rmnc[s_idx] * cos, axis=-1)
        Z = np.sum(self.zmns[s_idx] * sin, axis=-1)
        L = np.sum(self.lmns[s_idx] * sin, axis=-1)

        # go to R-Phi-Z
        phi = []
        for zeta in zax:
            phi.append(zeta*np.ones_like(tax))
        R2 = np.reshape(R, (Nzeta, Ntheta))
        Z2 = np.reshape(Z, (Nzeta, Ntheta))
        L2 = np.reshape(L, (Nzeta, Ntheta))
        if full_torus:
            # apply NFP symmetry
            phi_n = []
            R_n = []
            Z_n = []
            L_n = []
            zn = np.linspace(0, np.pi*2, self.nfp, endpoint=False)
            for z in zn:
                phi_n.append(np.array(phi)+z)
                R_n.append(R2)
                Z_n.append(Z2)
                L_n.append(L2)

            phi = np.concatenate(phi_n, axis=0)
            R2 = np.concatenate(R_n, axis=0)
            Z2 = np.concatenate(Z_n, axis=0)
            L2 = np.concatenate(L_n, axis=0)

        # convert to xyz
        X2 = R2*np.cos(phi)
        Y2 = R2*np.sin(phi)
        self.Lambda = L2
        return X2, Y2, Z2

    def get_surface(self, surface, N_zeta=20, N_theta=8, save_cloud=False):
        '''
        Compute area on a single flux surface
        using finite difference area elements
        with resolution (N_zeta, N_theta).
        '''

        # get points
        nfp = self.nfp
        r_arr = []
        for p in np.linspace(0, np.pi*2/nfp, N_zeta):
            r, z = self.get_xsection(N_theta, phi=p, s=surface)

            x = r*np.cos(p)
            y = r*np.sin(p)

            r_arr.append(np.transpose([x, y, z]))

        r_arr = np.transpose(r_arr)
        if save_cloud:
            self.r_cloud.append(r_arr)

        # get displacements
        def uv_space(X_arr, Y_arr, Z_arr):
            # modifying the code such that toroidal (and poloidal) directions need not be closed
            # this enables area computation on a field period for stellarators
            # if this is correct, the previous algorithm had an (n-1) edge error
            # yes, I believe that is the case. The previous implementation double counted the edge [0, 1]

            dXdu = X_arr[1:, :-1] - X_arr[:-1, :-1]
            dYdu = Y_arr[1:, :-1] - Y_arr[:-1, :-1]
            dZdu = Z_arr[1:, :-1] - Z_arr[:-1, :-1]

            dXdv = X_arr[:-1, 1:] - X_arr[:-1, :-1]
            dYdv = Y_arr[:-1, 1:] - Y_arr[:-1, :-1]
            dZdv = Z_arr[:-1, 1:] - Z_arr[:-1, :-1]

            return dXdu, dYdu, dZdu, dXdv, dYdv, dZdv

        X_arr, Y_arr, Z_arr = r_arr
        dXdu, dYdu, dZdu, dXdv, dYdv, dZdv = uv_space(X_arr, Y_arr, Z_arr)

        # get area
        dRdu = np.array([dXdu, dYdu, dZdu])
        dRdv = np.array([dXdv, dYdv, dZdv])

        # compute cross product and take norm
        dArea = np.linalg.norm(np.cross(dRdu, dRdv, axis=0), axis=0)
        if save_cloud:
            self.A_cloud.append(dArea)

        return np.sum(dArea) * nfp

    def overwrite_simple_torus(self, R=6, a=2):
        '''
        Overwrites VMEC file with concentric circle geometry.

        This is used for testing.
        '''

        sax = np.linspace(0, 1, self.ns)
        rax = a * np.sqrt(sax)

        # overwrite with simple torus
        self.N_modes = 2
        self.xn = np.array([0, 0])
        self.xm = np.array([0, 1])

        self.rmnc = np.array([[R, r] for r in rax])
        self.zmns = np.array([[0, r] for r in rax])

        self.Rmajor = R
        self.aminor = a

    def plot_surface_mesh(self, Nzeta=100, Ntheta=100, s_idx=50, full_torus=True, color="b"):
        ''' Function to plot a surface mesh '''

        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

        X, Y, Z = self.getSurfaceMesh(Nzeta=Nzeta,
                                      Ntheta=Ntheta,
                                      s_idx=s_idx,
                                      full_torus=full_torus,
                                      zeta_zero_mid=False,
                                      theta_zero_mid=False)

        _ = ax.plot_surface(X, Y, Z, color=color)
        ax.set_aspect('equal')

        plt.show()


class VmecRunner():

    def __init__(self, input_file, engine):

        self.data = f90nml.read(input_file)
        self.input_file = input_file

        self.engine = engine

    def run(self, f_input, ncpu=2):

        import vmec as vmec_py
        from mpi4py import MPI

        self.data.write(f_input, force=True)
        tag = f_input.split('/')[-1][6:]
        vmec_wout = f"wout_{tag}.nc"

        # overwrite previous input file
        self.input_file = f_input
        self.data = f90nml.read(f_input)

        path = self.engine.path
        if os.path.exists(path + vmec_wout):
            info(f" completed vmec run found: {vmec_wout}, skipping run")
            return

        verbose = True
        fcomm = MPI.COMM_WORLD.py2f()
        reset_file = ''
        ictrl = np.zeros(5, dtype=np.int32)
        ictrl[0] = 1 + 2 + 4 + 8 + 16
        # see VMEC2000/Sources/TimeStep/runvmec.f
        vmec_py.runvmec(ictrl, f_input, verbose, fcomm, reset_file)
        info('slurm vmec completed')

        # right now VMEC writes output to root (input is stored in engine.gx_path)
        #    this cmd manually moves VMEC output to gx_path after vmec_py.run() finishes
        info(f"  moving VMEC files to {path}")
        cmd = f"mv *{tag}* {path}"
        os.system(cmd)


def main():

    # Set-up command line arguments
    description = "Plot VMEC surface mesh"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("vmec_file", help="Name of VMEC file")
    parser.add_argument("-z", "--nzeta", help="Number of toroidal points (zeta)", type=int, default=100)
    parser.add_argument("-t", "--ntheta", help="Number of poloidal points (theta)", type=int, default=100)
    parser.add_argument("-s", "--surface", help="Specify surface index from [0, N-1], -1 sets last surface", type=int, default=50)
    parser.add_argument("--notorus", help="Do not plot as full torus", action="store_false")

    # Parse command line arguments
    args = parser.parse_args()

    if not os.path.exists(args.vmec_file):
        raise FileNotFoundError(f"VMEC file does not exist: {args.vmec_file}")

    if not args.vmec_file.lower().endswith(".nc"):
        raise ValueError("VMEC file should be in NetCDF format with .nc extension")

    if args.nzeta < 10:
        raise ValueError("Minimum 10 points in toroidal direction required")

    if args.ntheta < 10:
        raise ValueError("Minimum 10 points in poloidal direction required")

    # Read the VMEC file
    vmec = VmecReader(fin=args.vmec_file)

    if args.surface < -1 or args.surface > len(vmec.rmnc) - 1:
        raise ValueError(f"Specified surface number, {args.surface}, does not exist within range [0, {len(vmec.rmnc)}] ")

    # Plot the surface mesh
    vmec.plot_surface_mesh(Nzeta=args.nzeta,
                           Ntheta=args.ntheta,
                           s_idx=args.surface,
                           full_torus=args.notorus)

vmec = VmecReader("/Users/mattwang/Downloads/wout_2007_000_000000.nc")
# Plot the surface mesh
vmec.plot_surface_mesh()