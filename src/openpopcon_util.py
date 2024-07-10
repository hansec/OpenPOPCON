import numpy as np
import contourpy as cntr
import numba as nb
import scipy.constants as const


def read_eqdsk(filename):
    ''' Taken from OpenFUSION toolkit. Perhaps do an import instead?
    '''
    def read_1d(fid, n):
        j = 0
        output = np.zeros((n,))
        for i in range(n):
            if j == 0:
                line = fid.readline()
            output[i] = line[j:j+16]
            j += 16
            if j == 16*5:
                j = 0
        return output

    def read_2d(fid, n, m):
        j = 0
        output = np.zeros((n, m))
        for k in range(n):
            for i in range(m):
                if j == 0:
                    line = fid.readline()
                output[k, i] = line[j:j+16]
                j += 16
                if j == 16*5:
                    j = 0
        return output
    # Read-in data
    eqdsk_obj = {}
    with open(filename, 'r') as fid:
        # Get sizes
        line = fid.readline()
        eqdsk_obj['case'] = line[:48]
        split_line = line[48:].split()
        eqdsk_obj['nr'] = int(split_line[-2])
        eqdsk_obj['nz'] = int(split_line[-1])
        # Read header content
        line_keys = [['rdim',  'zdim',  'rcentr',  'rleft',  'zmid'],
                     ['raxis', 'zaxis', 'psimag', 'psibry', 'bcentr'],
                     ['ip',    'skip',  'skip',   'skip',   'skip'],
                     ['skip',  'skip',  'skip',   'skip',   'skip']]
        for i in range(4):
            line = fid.readline()
            for j in range(5):
                if line_keys[i][j] == 'skip':
                    continue
                line_seg = line[j*16:(j+1)*16]
                eqdsk_obj[line_keys[i][j]] = float(line_seg)
        # Read flux profiles
        keys = ['fpol', 'pres', 'ffprim', 'pprime']
        for key in keys:
            eqdsk_obj[key] = read_1d(fid, eqdsk_obj['nr'])
        # Read PSI grid
        eqdsk_obj['psirz'] = read_2d(fid, eqdsk_obj['nz'],
                                        eqdsk_obj['nr'])
        # Read q-profile
        eqdsk_obj['qpsi'] = read_1d(fid, eqdsk_obj['nr'])
        # Read limiter count
        line = fid.readline()
        eqdsk_obj['nbbs'] = int(line.split()[0])
        eqdsk_obj['nlim'] = int(line.split()[1])
        # Read outer flux surface
        eqdsk_obj['rzout'] = read_2d(fid, eqdsk_obj['nbbs'], 2)
        # Read limiting corners
        eqdsk_obj['rzlim'] = read_2d(fid, eqdsk_obj['nlim'], 2)
    return eqdsk_obj

def get_fluxvolumes(gEQDSK: dict, Npsi:int=50, nres:int=300):
    """
    Calculates the flux surface volumes from the gEQDSK object.

    Parameters
    ----------
    gEQDSK : dict
        gEQDSK object.
    Npsi : int, optional
        Number of flux surfaces to calculate. The default is 50.
    nres : int, optional
        Number of resolution points. The default is 300.
    """
    rs = np.linspace(gEQDSK['rleft'], 
                     gEQDSK['rleft'] + gEQDSK['rdim'], 
                     gEQDSK['nr'])
    zs = np.linspace(gEQDSK['zmid'] - gEQDSK['zdim']/2, 
                     gEQDSK['zmid'] + gEQDSK['zdim']/2, 
                     gEQDSK['nz'])

    # Get and normalize fluxes
    fluxes = gEQDSK['psirz'] - gEQDSK['psibry']
    fluxes/= np.max(fluxes)
    if gEQDSK['psibry'] < gEQDSK['psimag']:
        fluxes = 1-fluxes

    cgen = cntr.contour_generator(x=rs, y=zs, z=fluxes)
    allsegs = []
    psin = np.linspace(0,1,Npsi)**2
    for level in psin:
        allsegs.append(cgen.create_contour(level))

    # Filter out stuff below the divertor
    closed_fluxsurfaces = []
    for segset in allsegs:
        true = []
        for i in range(len(segset)):
            if np.abs(np.sqrt( (np.average(segset[i][:,1])-gEQDSK['zaxis'])**2 + (np.average(segset[i][:,0])-gEQDSK['raxis'])**2 ) ) < gEQDSK['zdim']/5:
                true.append(i)
        if len(true) == 0:
            raise ValueError('No true path found')
        for truei in true:
            closed_fluxsurfaces.append(segset[truei])

    h = closed_fluxsurfaces[-1][:,1].max() - closed_fluxsurfaces[-1][:,1].min()
    dh_target = h/nres

    # Get flux surface volumes
    hs = np.linspace(closed_fluxsurfaces[-1][:,1].min()+dh_target, closed_fluxsurfaces[-1][:,1].max()-dh_target, nres)
    dh = hs[1] - hs[0]
    Volgrid = np.zeros(len(closed_fluxsurfaces))

    for icontour, contour in enumerate(closed_fluxsurfaces):
        xinners = np.zeros(nres)
        xouters = np.zeros(nres)
        for i in range(nres):
            xs = np.zeros(2)
            k = 0
            # find the two intervals that cross the height hs[i]
            for j in range(len(contour)-1):
                if (hs[i] - contour[j][1])*(hs[i] - contour[j+1][1]) < 0:
                    # interpolate to find the x value
                    x = contour[j][0] + (hs[i] - contour[j][1])*(contour[j+1][0] - contour[j][0])/(contour[j+1][1] - contour[j][1])
                    xs[k] = x
                    k += 1
            if k != 2:
                print(i)
                raise ValueError('No two x values found')
            xinner = min(xs)
            xouter = max(xs)
            xinners[i] = xinner
            xouters[i] = xouter

        # Trapezoidal sum, assuming small dr
        V = 0
        # Bottom cap
        V += 0.5*np.pi*dh* ( xouters[0]**2 - xinners[0]**2 )
        # Middle
        for i in range(nres-1):
            V += np.pi*dh* ( xouters[i]**2 + 0.5*(xouters[i+1]**2 - xouters[i]**2) ) - np.pi*dh* ( xinners[i]**2 + 0.5*(xinners[i+1]**2 - xinners[i]**2) )
        # Top cap
        V += 0.5*np.pi*dh* ( xouters[-1]**2 - xinners[-1]**2 )
        Volgrid[icontour] = V
    
    return psin, Volgrid

@nb.njit
def get_n_GR(Ip: float, a: float) -> float:
    return Ip/(np.pi*a**2)

@nb.njit
def get_Zeff(impfrac:float, imcharg:float, fHe:float, dil:float) -> float: #TODO: Arbitrary set of impurities
    return ( (1-impfrac-fHe) + impfrac*imcharg**2 + 4*fHe)*dil

@nb.njit
def get_q_a(a:float, B0:float, kappa:float, R:float, Ip:float) -> float: #TODO: Update with Kikuchi eq?
    return 2*np.pi*a**2*B0*(kappa**2+1)/(2*R*const.mu_0*Ip*10**6)

@nb.njit
def get_Cspitz(Zeff:float, Ip:float, q_a:float, a:float, kappa:float, volavgcurr:float) -> float:
    Fz    = (1+1.198*Zeff + 0.222*Zeff**2)/(1+2.966*Zeff + 0.753*Zeff**2)
    eta1  = 1.03e-4*Zeff*Fz
    j0avg = Ip/(np.pi*a**2*kappa)*1.0e6
    if (volavgcurr == True):
        Cspitz = eta1*q_a*j0avg**2
    else:
        Cspitz = eta1
    Cspitz /= 1.6e-16*1.0e20 #unit conversion to keV 10^20 m^-3
    return Cspitz

@nb.njit
def get_plasma_dilution(impfrac:float, imcharg:float, fHe:float) -> float:
    return 1/(1 + impfrac*imcharg + 2*fHe)

@nb.njit
def get_P_LH_threshold(n20:float, bs:float) -> float: #TODO: Update with Oak paper?
    return 0.049*n20**(0.72)*bs

@nb.njit
def get_bs_factor(B0:float, R:float, a:float, kappa:float) -> float:
    return B0**(0.8)*(2.*np.pi*R * 2*np.pi*a * np.sqrt((kappa**2+1)/2))**(0.94)

@nb.njit
def get_plasma_vol(a:float, R:float, kappa:float) -> float: #TODO: Update with Kikuchi eq?
    return 2.*np.pi**2*a**2*R*kappa

@nb.njit
def get_plasma_dvolfac(a:float, R:float, kappa:float) -> float: #TODO: Update with Kikuchi eq?
    return 4*np.pi**2*a**2*R*kappa