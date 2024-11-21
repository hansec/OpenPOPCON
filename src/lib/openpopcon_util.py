import numpy as np
import contourpy as cntr
import csv
import os
import pathlib
import numba as nb
import matplotlib.pyplot as plt
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

        eqdsk_obj['Bt_rz'] = eqdsk_obj['bcentr'] * eqdsk_obj['rcentr'] / eqdsk_obj['R']
        
        dpsidZ, dpsidR = np.gradient(eqdsk_obj['psirz'], R, Z)
        eqdsk_obj['Br_rz'] = -dpsidZ / eqdsk_obj['R']
        eqdsk_obj['Bz_rz'] = dpsidR / eqdsk_obj['R']

    return eqdsk_obj

def get_fluxvolumes(gEQDSK: dict, Npsi: int = 50, nres: int = 300):
    """
    Calculates the flux surface volumes and areas from the gEQDSK object.

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
    fluxes = gEQDSK['psirz']-gEQDSK['psibry']
    if fluxes[0,0] < 0:
        fluxes = -fluxes
    raxi = np.argmin(np.abs(rs - gEQDSK['raxis']))
    zaxi = np.argmin(np.abs(zs - gEQDSK['zaxis']))

    fluxes_axis = fluxes[zaxi, raxi]
    fluxes += -fluxes_axis
    fluxes /= np.abs(fluxes_axis)

    cgen = cntr.contour_generator(x=rs, y=zs, z=fluxes)
    allsegs = []
    psin = np.linspace(0.001, 1, Npsi)**2
    for level in psin:
        allsegs.append(cgen.create_contour(level))
    
    # Sometimes the flux surfaces have multiple paths including levels outside the plasma, filter these out
    closed_fluxsurfaces = []
    for psii, segset in enumerate(allsegs):
        true = []
        for i in range(len(segset)):
            if np.abs(np.sqrt((np.average(segset[i][:, 1])-gEQDSK['zaxis'])**2 + (np.average(segset[i][:, 0])-gEQDSK['raxis'])**2)) < gEQDSK['zdim']/5:
                true.append(i)
        if len(true) == 0:
            raise ValueError(f'No true path found for flux surface psi={psin[psii]}')
        for truei in true:
            closed_fluxsurfaces.append(segset[truei])

    h = closed_fluxsurfaces[-1][:, 1].max() - \
            closed_fluxsurfaces[-1][:, 1].min()
    dh_target = h/nres

    # Get flux surface volumes
    hs = np.linspace(closed_fluxsurfaces[-1][:, 1].min()+dh_target,
                        closed_fluxsurfaces[-1][:, 1].max()-dh_target, nres)
    dh = hs[1] - hs[0]
    Volgrid = np.zeros(len(closed_fluxsurfaces))
    Agrid = np.zeros(len(closed_fluxsurfaces))
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
                    x = contour[j][0] + (hs[i] - contour[j][1])*(contour[j+1]
                                                                    [0] - contour[j][0])/(contour[j+1][1] - contour[j][1])
                    xs[k] = x
                    k += 1
            if k != 2:
                # raise ValueError('No two x values found')
                continue
            xinner = min(xs)
            xouter = max(xs)
            xinners[i] = xinner
            xouters[i] = xouter

        # Trapezoidal sum, assuming small dr
        V = 0
        # Bottom cap
        V += 0.5*np.pi*dh * (xouters[0]**2 - xinners[0]**2)
        # Middle
        for i in range(nres-1):
            V += np.pi*dh * (xouters[i]**2 + 0.5*(xouters[i+1]**2 - xouters[i]**2)) - \
                np.pi*dh * (xinners[i]**2 + 0.5 *
                            (xinners[i+1]**2 - xinners[i]**2))
        # Top cap
        V += 0.5*np.pi*dh * (xouters[-1]**2 - xinners[-1]**2)
        Volgrid[icontour] = V
        # Get the surface area of the flux surface
        d = np.diff(contour, axis=0)
        ds = np.sqrt(d[:, 0]**2 + d[:, 1]**2)
        Agrid[icontour] = np.trapz(2*np.pi*contour[:-1,0] * ds, axis=0)

    return psin, Volgrid, Agrid, closed_fluxsurfaces

def get_trapped_particle_fraction(gEQDSK: dict, Npsi: int=50):
    psin, closed_fluxsurfaces = get_contours(gEQDSK, Npsi)
    # closed_fluxsurfaces[5]
    Bt = gEQDSK['Bt_rz']
    Br = gEQDSK['Br_rz']
    Bz = gEQDSK['Bz_rz']

    Bt_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], Bt.T)
    Br_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], Br.T)
    Bz_interp = RBS(gEQDSK['R'][0,:], gEQDSK['Z'][:,0], Bz.T)

    B2avg = np.zeros(len(psin))
    Bm2avg = np.zeros(len(psin))
    funcavg = np.zeros(len(psin))

    for i, psi in enumerate(psin):
        contour = closed_fluxsurfaces[i]
        R = contour[:,0]
        Z = contour[:,1]
        Bt = Bt_interp(R, Z, grid=False)
        Br = Br_interp(R, Z, grid=False)
        Bz = Bz_interp(R, Z, grid=False)
        B2 = Bt**2 + Br**2 + Bz**2
        Bm2 = 1/(Bt**2 + Br**2 + Bz**2)
        B = np.sqrt(B2)
        Bc = np.max(B)
        func = np.sqrt(1-B/Bc) - (1/3) * (1-B/Bc)**(3/2)
        func *= Bm2

        B2avg[i] = np.mean(B2)
        Bm2avg[i] = np.mean(Bm2)
        funcavg[i] = np.mean(func)

    f = 1 - B2avg * Bm2avg + (3/2) * B2avg * funcavg

    return psin, f

def get_contours(gEQDSK: dict, Npsi: int=50):
    rs = np.linspace(gEQDSK['rleft'],
                        gEQDSK['rleft'] + gEQDSK['rdim'],
                        gEQDSK['nr'])
    zs = np.linspace(gEQDSK['zmid'] - gEQDSK['zdim']/2,
                        gEQDSK['zmid'] + gEQDSK['zdim']/2,
                        gEQDSK['nz'])
    # Get and normalize fluxes
    fluxes = gEQDSK['psirz']-gEQDSK['psibry']
    if fluxes[0,0] < 0:
        fluxes = -fluxes
    raxi = np.argmin(np.abs(rs - gEQDSK['raxis']))
    zaxi = np.argmin(np.abs(zs - gEQDSK['zaxis']))

    fluxes_axis = fluxes[zaxi, raxi]
    fluxes += -fluxes_axis
    fluxes /= np.abs(fluxes_axis)

    cgen = cntr.contour_generator(x=rs, y=zs, z=fluxes)
    allsegs = []
    psin = np.linspace(0.001, 1, Npsi)**2
    for level in psin:
        allsegs.append(cgen.create_contour(level))
    # Filter out stuff below the divertor
    closed_fluxsurfaces = []
    for psii, segset in enumerate(allsegs):
        true = []
        for i in range(len(segset)):
            if np.abs(np.sqrt((np.average(segset[i][:, 1])-gEQDSK['zaxis'])**2 + (np.average(segset[i][:, 0])-gEQDSK['raxis'])**2)) < gEQDSK['zdim']/5:
                true.append(i)
        if len(true) == 0:
            raise ValueError('No true path found')
        for truei in true:
            closed_fluxsurfaces.append(segset[truei])

    
    return psin, closed_fluxsurfaces

def get_current_density(gEQDSK: dict, Npsi: int=50):
    zs = np.linspace(gEQDSK['zmid']-0.5*gEQDSK['zdim'], gEQDSK['zmid']+0.5*gEQDSK['zdim'], gEQDSK['psirz'].shape[1])
    rs = np.linspace(gEQDSK['rleft'], gEQDSK['rleft']+gEQDSK['rdim'], gEQDSK['psirz'].shape[0])
    psirz = gEQDSK['psirz']
    psinrz = (psirz-gEQDSK['psimag'])/(gEQDSK['psibry']-gEQDSK['psimag'])

    psin, volgrid, agrid, fs = get_fluxvolumes(gEQDSK, Npsi)
    sqrtpsin = np.sqrt(psin)
    volgrid = np.interp(sqrtpsin,np.sqrt(psin),volgrid)

    ffp = np.asarray(gEQDSK['ffprim'])
    psi = np.linspace(gEQDSK['psimag'],gEQDSK['psibry'],np.shape(ffp)[0])

    fpolrz = np.interp(psirz, psi, gEQDSK['fpol'])
    fprimrz = np.interp(psirz, psi, np.gradient(gEQDSK['fpol'], psi))
    ffprz = np.interp(psirz, psi, gEQDSK['ffprim'])
    pprz = np.interp(psirz, psi, gEQDSK['pprime'])
    jpolrz = 1/rs*fprimrz*np.gradient(psirz, rs,zs)
    gradstarpsi = -(4e-7*np.pi)*rs**2*pprz - 0.5*ffprz
    jtorrz = 1/rs * gradstarpsi
    jtot = np.sqrt(jpolrz[0]**2 + jpolrz[1]**2 + jtorrz**2)
    tantheta = np.sqrt(jpolrz[0]**2 + jpolrz[1]**2)/jtorrz
    pitchangle = np.arctan(tantheta)
    jtotspline = RBS(zs, rs, jtot)
    jtorspline = RBS(zs, rs, jtorrz)

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
        totallength = np.sum(lengths)
        # Root mean square of the current density since OH Power is ~ J^2
        javgrms[i] = np.sqrt(np.sum(np.abs(jtotspline.ev(zmid, rmid)**2*lengths))/totallength)
        jtoravg[i] = np.sum(np.abs(jtorspline.ev(zmid, rmid)*lengths))/totallength

        dr = np.diff(ri)
        dz = np.diff(zi)
        cross_sec_areas[i] = np.sum(rmid*dz - zmid*dr)/2
    
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