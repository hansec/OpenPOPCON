import numpy as np
import contourpy as cntr
import csv
import os
import pathlib
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline as RBS

def yaml_edit(filename, key, value) -> None:
    with open(filename, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip(' ').startswith(key):
            ind = line.find(':')
            lines[i] = lines[i][:ind+1] + ' ' + str(value) + '\n'
            break
    
    with open(filename, 'w') as f:
        f.writelines(lines)
    
    return


def get_POPCON_homedir(path=[]):
    return str(pathlib.Path(__file__).resolve().parent.parent.parent) + os.sep + str(os.sep).join(path)

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

        R = np.linspace(eqdsk_obj['rleft'],
                        eqdsk_obj['rleft'] + eqdsk_obj['rdim'],
                        eqdsk_obj['nr'])
        
        Z = np.linspace(eqdsk_obj['zmid'] - eqdsk_obj['zdim']/2,
                        eqdsk_obj['zmid'] + eqdsk_obj['zdim']/2,
                        eqdsk_obj['nz'])
        
        eqdsk_obj['R'], eqdsk_obj['Z'] = np.meshgrid(R, Z)

        # Toroidal field from B_t = F(psi)/R, which shows the dia/paramagnetism
        # of the plasma. np.interp clamps outside the flux range, so this falls
        # back to the vacuum field F_edge/R = bcentr*rcentr/R outside the LCFS.
        psi1d = np.linspace(eqdsk_obj['psimag'], eqdsk_obj['psibry'], eqdsk_obj['nr'])
        fluxorder = slice(None, None, 1 - 2*(eqdsk_obj['psimag'] > eqdsk_obj['psibry']))
        fpolrz = np.interp(eqdsk_obj['psirz'], psi1d[fluxorder], eqdsk_obj['fpol'][fluxorder])
        eqdsk_obj['Bt_rz'] = fpolrz / eqdsk_obj['R']
        
        dpsidZ, dpsidR = np.gradient(eqdsk_obj['psirz'], Z, R)
        eqdsk_obj['Br_rz'] = -dpsidZ / eqdsk_obj['R']
        eqdsk_obj['Bz_rz'] = dpsidR / eqdsk_obj['R']

    return eqdsk_obj

def get_fluxvolumes(gEQDSK: dict, Npsi: int = 50):
    """
    Calculates the flux surface volumes and areas from the gEQDSK object.

    Parameters
    ----------
    gEQDSK : dict
        gEQDSK object.
    Npsi : int, optional
        Number of flux surfaces to calculate. The default is 50.
    """
    psin, closed_fluxsurfaces = get_contours(gEQDSK, Npsi)

    Volgrid = np.zeros(len(closed_fluxsurfaces))
    Agrid = np.zeros(len(closed_fluxsurfaces))
    for icontour, contour in enumerate(closed_fluxsurfaces):
        r = contour[:, 0]
        z = contour[:, 1]
        # Volume of revolution about R=0.
        # V = (pi/3) sum (r_i + r_{i+1})(r_i z_{i+1} - r_{i+1} z_i).
        cross = r[:-1]*z[1:] - r[1:]*z[:-1]
        Volgrid[icontour] = np.abs(np.pi/3 * np.sum((r[:-1] + r[1:])*cross))
        # Get the surface area of the flux surface
        d = np.diff(contour, axis=0)
        ds = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        Agrid[icontour] = np.trapezoid(2*np.pi*contour[:-1,0] * ds, axis=0)

    return psin, Volgrid, Agrid, closed_fluxsurfaces

def get_trapped_particle_fraction(gEQDSK: dict, Npsi: int=50):
    """
    Trapped particle fraction on each flux surface.

    Lin-Liu and Miller [9] gives analytic lower and upper
    bounds ft_lower <= ft <= ft_upper, and recommend the blend
    ft = 0.75*ft_upper + 0.25*ft_lower as the best cheap approximation to the
    exact value. That blend is returned here.
    """
    psin, closed_fluxsurfaces = get_contours(gEQDSK, Npsi)

    Bt_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], gEQDSK['Bt_rz'].T)
    Br_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], gEQDSK['Br_rz'].T)
    Bz_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], gEQDSK['Bz_rz'].T)

    ft_lower = np.zeros(len(psin))
    ft_upper = np.zeros(len(psin))

    for i, psi in enumerate(psin):
        contour = closed_fluxsurfaces[i]
        ri = contour[:,0]
        zi = contour[:,1]
        rmid = 0.5*(ri[1:]+ri[:-1])
        zmid = 0.5*(zi[1:]+zi[:-1])
        lengths = np.sqrt(np.diff(ri)**2 + np.diff(zi)**2)

        Bt = Bt_interp(rmid, zmid, grid=False)
        Br = Br_interp(rmid, zmid, grid=False)
        Bz = Bz_interp(rmid, zmid, grid=False)
        Bp2 = Br**2 + Bz**2
        B2 = Bt**2 + Bp2
        B = np.sqrt(B2)
        Bmax = np.max(B)
        b = B/Bmax


        Bpol = np.sqrt(Bp2)
        Bpol = np.maximum(Bpol, 1e-6*np.max(Bpol))
        weights = lengths/Bpol
        totalweight = np.sum(weights)
        avg = lambda x: np.sum(weights*x)/totalweight

        h = avg(B)/Bmax
        h2 = avg(B2)/Bmax**2
        # Lin-Liu & Miller eq. 4 (upper bound) and eq. 7 (lower bound)
        ft_upper[i] = 1 - h2/h**2 * (1 - np.sqrt(1-h)*(1 + 0.5*h))
        ft_lower[i] = 1 - h2 * avg((1 - np.sqrt(1-b)*(1 + b/2))/b**2)

    # Lin-Liu & Miller eqs. 18-19
    f = 0.75*ft_upper + 0.25*ft_lower

    return psin, f

def _polygon_area(contour):
    r = contour[:, 0]
    z = contour[:, 1]
    return np.abs(np.sum(r[:-1]*z[1:] - r[1:]*z[:-1])/2)

def get_contours(gEQDSK: dict, Npsi: int=50):
    """
    Traces the closed flux surface enclosing the magnetic axis at each of
    Npsi normalized flux levels. Returns exactly one contour per level, so
    that closed_fluxsurfaces[i] always corresponds to psin[i].
    """
    rs = np.linspace(gEQDSK['rleft'],
                        gEQDSK['rleft'] + gEQDSK['rdim'],
                        gEQDSK['psirz'].shape[1])
    zs = np.linspace(gEQDSK['zmid'] - gEQDSK['zdim']/2,
                        gEQDSK['zmid'] + gEQDSK['zdim']/2,
                        gEQDSK['psirz'].shape[0])
    raxi = np.argmin(np.abs(rs - gEQDSK['raxis']))
    zaxi = np.argmin(np.abs(zs - gEQDSK['zaxis']))
    psi_axis = gEQDSK['psirz'][zaxi, raxi]
    fluxes = (gEQDSK['psirz'] - psi_axis)/(gEQDSK['psibry'] - psi_axis)

    cgen = cntr.contour_generator(x=rs, y=zs, z=fluxes)
    psin = np.linspace(0.001, 1, Npsi)**2
    axis_point = (gEQDSK['raxis'], gEQDSK['zaxis'])

    closed_fluxsurfaces = []
    for level in psin:
        segset = [seg for seg in cgen.create_contour(level) if len(seg) > 2]
        if len(segset) == 0:
            raise ValueError(f'No flux surface contour found at psin={level}')
        enclosing = [seg for seg in segset if Path(seg).contains_point(axis_point)]
        if len(enclosing) != 0:
            chosen = min(enclosing, key=_polygon_area)
        else:
            chosen = min(segset, key=lambda seg: np.hypot(np.mean(seg[:, 0])-axis_point[0],
                                                          np.mean(seg[:, 1])-axis_point[1]))
        closed_fluxsurfaces.append(chosen)

    return psin, closed_fluxsurfaces

def get_current_density(gEQDSK: dict, Npsi: int=50):
    """
    Flux surface averaged current densities from a gEQDSK, in A/m^2.
    """
    mu0 = 4e-7*np.pi
    # psirz has shape (nz, nr): axis 0 is Z, axis 1 is R
    nz, nr = np.shape(gEQDSK['psirz'])
    zs = np.linspace(gEQDSK['zmid']-0.5*gEQDSK['zdim'], gEQDSK['zmid']+0.5*gEQDSK['zdim'], nz)
    rs = np.linspace(gEQDSK['rleft'], gEQDSK['rleft']+gEQDSK['rdim'], nr)
    psirz = gEQDSK['psirz']
    RR, _ = np.meshgrid(rs, zs)

    psin, fs = get_contours(gEQDSK, Npsi)

    ffp = np.asarray(gEQDSK['ffprim'])
    psi = np.linspace(gEQDSK['psimag'],gEQDSK['psibry'],np.shape(ffp)[0])
    reverse = gEQDSK['psimag'] > gEQDSK['psibry']
    fluxorder = slice(None, None, 1-2*reverse)
    fprimrz = np.interp(psirz, psi[fluxorder], np.gradient(gEQDSK['fpol'][fluxorder], psi[fluxorder]))
    ffprz = np.interp(psirz, psi[fluxorder], gEQDSK['ffprim'][fluxorder])
    pprz = np.interp(psirz, psi[fluxorder], gEQDSK['pprime'][fluxorder])

    dpsidZ, dpsidR = np.gradient(psirz, zs, rs)
    B_pol = np.sqrt(dpsidR**2 + dpsidZ**2)/RR

    # mu0*J_pol = F'(psi)*B_pol
    jpolrz = fprimrz*B_pol/mu0
    # Grad-Shafranov, Delta*psi = -mu0*R^2*p' - FF', combined with
    # Delta*psi = -mu0*R*J_tor
    jtorrz = -(RR*pprz + ffprz/(mu0*RR))

    jtot = np.sqrt(jpolrz**2 + jtorrz**2)
    jtotspline = RBS(zs, rs, jtot)
    jtorspline = RBS(zs, rs, jtorrz)
    Bpolspline = RBS(zs, rs, B_pol)

    jtoravg = np.zeros_like(psin)
    javgrms = np.zeros_like(psin)
    cross_sec_areas = np.zeros(len(fs))
    for i, level in enumerate(psin):
        contour = fs[i]
        ri = contour[:,0]
        zi = contour[:,1]
        rmid = 0.5*(ri[1:]+ri[:-1])
        zmid = 0.5*(zi[1:]+zi[:-1])
        lengths = np.sqrt(np.diff(ri)**2 + np.diff(zi)**2)
        Bpol_c = np.abs(Bpolspline.ev(zmid, rmid))
        Bpol_c = np.maximum(Bpol_c, 1e-6*np.max(Bpol_c))
        weights = lengths/Bpol_c
        totalweight = np.sum(weights)
        # Root mean square of the current density since OH Power is ~ J^2
        javgrms[i] = np.sqrt(np.sum(weights*jtotspline.ev(zmid, rmid)**2)/totalweight)
        jtoravg[i] = np.sum(weights*jtorspline.ev(zmid, rmid))/totalweight

        dr = np.diff(ri)
        dz = np.diff(zi)
        cross_sec_areas[i] = np.abs(np.sum(rmid*dz - zmid*dr)/2)
    
    return psin, javgrms, jtoravg, cross_sec_areas

def read_profsfile(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        profstable = {}
        for h in header:
            profstable[h] = []
        for row in reader:
            for h, v in zip(header, row):
                profstable[h].append(v)
    
    for k in profstable.keys():
        profstable[k] = np.asarray(profstable[k], dtype=np.float64)
    if 'rho' not in profstable and 'r' not in profstable:
        if 'psi' not in profstable:
            raise ValueError('No rho, r or psi in profile file')
        else:
            profstable['rho'] = np.sqrt(np.asarray(profstable['psi'], dtype=np.float64))
    elif 'rho' not in profstable:
        profstable['rho'] = profstable['r']

    return profstable

def safe_get(unsafe_dict, key, default=None):
    if key in unsafe_dict:
        return unsafe_dict[key]
    else:
        return default
    

def conditional_njit(enable, **kwargs):
    def dec(func):
        if enable:
            return nb.njit(**kwargs)(func)
        else:
            return func
    return dec