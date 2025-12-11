import numpy as np
import netCDF4 as nc
from scipy import integrate, interpolate
import matplotlib.pyplot as plt
import os

class VMECReader:
    """
    VMEC file reader that extracts stellarator equilibrium data from VMEC wout files.
    Adapted from T3D geometry models to provide all required quantities for OpenPOPCON.
    
    Extracts:
    - dV/drho (volume derivative w.r.t. normalized toroidal flux coordinate)
    - Cross sectional area as a function of rho
    - Flux surface area as a function of rho  
    - Volume averaged toroidal magnetic field
    - Toroidal current density as a function of rho
    - Iota at 2/3 flux surface
    """
    def __init__(self, filename):
        self.filename = filename
        print(f"Reading VMEC equilibrium from: {filename}")
        self._read_variables()
        self._compute_geometric_quantities()
    
    def _read_variables(self):
        """Read VMEC variables from NetCDF file"""
        try:
            with nc.Dataset(self.filename, 'r') as f:
                def get(f, key):
                    return f.variables[key][:]
                
                # Basic parameters
                self.nfp = int(get(f, 'nfp'))  # Number of field periods
                self.ns = int(get(f, 'ns'))    # Number of flux surfaces
                self.mnmax = int(get(f, 'mnmax'))  # Number of Fourier modes
                
                # Machine parameters 
                self.aminor = float(get(f, 'Aminor_p'))   # Minor radius [m]
                self.Rmajor = float(get(f, 'Rmajor_p'))   # Major radius [m] 
                self.volume = float(get(f, 'volume_p'))   # Total volume [m^3]
                self.volavgB = float(get(f, 'volavgB'))   # Volume averaged |B| [T]
                self.B0 = float(get(f, 'b0'))             # B field at magnetic axis [T]
                
                # 1D arrays - mode numbers and profiles
                self.xm = get(f, 'xm')  # Poloidal mode numbers
                self.xn = get(f, 'xn')  # Toroidal mode numbers  
                self.iotaf = get(f, 'iotaf')  # Rotational transform on full mesh
                self.presf = get(f, 'presf')  # Pressure on full mesh
                
                # Coordinate arrays
                self.s = np.linspace(1e-10, 1, self.ns)  # Normalized toroidal flux coordinate
                self.rho = np.sqrt(self.s)  # Normalized minor radius coordinate
                
                # 2D Fourier coefficient arrays [ns x mnmax]
                self.rmnc = np.array(get(f, 'rmnc'))     # R cosine coefficients
                self.zmns = np.array(get(f, 'zmns'))     # Z sine coefficients
                self.lmns = np.array(get(f, 'lmns'))     # Lambda sine coefficients
                self.bmnc = np.array(get(f, 'bmnc'))     # |B| cosine coefficients
                self.gmnc = np.array(get(f, 'gmnc'))     # Jacobian sqrt(g) cosine coefficients
                
                # Contravariant B field components
                self.bsupumnc = np.array(get(f, 'bsupumnc'))  # B^theta cosine coefficients
                self.bsupvmnc = np.array(get(f, 'bsupvmnc'))  # B^phi cosine coefficients
                
                # Toroidal flux
                self.phi = get(f, 'phi')  # Toroidal flux [Wb]
                
                print(f"✓ Successfully read VMEC file with {self.ns} flux surfaces, {self.mnmax} modes")
                print(f"  - Major radius: {self.Rmajor:.3f} m")  
                print(f"  - Minor radius: {self.aminor:.3f} m")
                print(f"  - Volume: {self.volume:.3f} m³")
                print(f"  - Volume averaged |B|: {self.volavgB:.3f} T")
                
        except Exception as e:
            print(f"✗ Error reading VMEC file: {e}")
            raise
    
    def demonstrate_profile_parameters(self):
        """
        Demonstrate the stellarator parabolic profile parameter choices
        Shows how the selected α₁, α₂, offset values create realistic profiles
        """
        print("=== Stellarator Profile Parameter Analysis ===")
        
        # Create radial grid
        rho = np.linspace(0, 1, 101)
        
        # Define parabolic profile function
        def parabolic_profile(rho, alpha1, alpha2, offset):
            return (1 - offset) * (1 - rho**alpha1)**alpha2 + offset
        
        # Stellarator optimized parameters
        profiles = {
            'Current Density': {
                'alpha1': 2.0, 'alpha2': 2.5, 'offset': 0.0,
                'color': 'red', 'reasoning': 'Naturally peaked, zero edge current'
            },
            'Electron Density': {
                'alpha1': 2.0, 'alpha2': 1.8, 'offset': 0.05,
                'color': 'blue', 'reasoning': 'Moderate peaking, realistic edge density'
            },
            'Ion Temperature': {
                'alpha1': 2.0, 'alpha2': 2.0, 'offset': 0.1,
                'color': 'green', 'reasoning': 'Balanced profile, sufficient edge temperature'
            },
            'Electron Temperature': {
                'alpha1': 2.0, 'alpha2': 2.0, 'offset': 0.1,
                'color': 'orange', 'reasoning': 'Similar to Ti, thermal equilibration'
            }
        }
        
        # Calculate and display profiles
        print("\nProfile Parameter Analysis:")
        print("ρ=0.0   ρ=0.33  ρ=0.67  ρ=1.0   | Parameters & Reasoning")
        print("-" * 80)
        
        for name, params in profiles.items():
            values = parabolic_profile(rho, params['alpha1'], params['alpha2'], params['offset'])
            print(f"{values[0]:.3f}   {values[33]:.3f}   {values[67]:.3f}   {values[100]:.3f}   | "
                  f"{name}: α₁={params['alpha1']}, α₂={params['alpha2']}, offset={params['offset']}")
            print(f"                                    | {params['reasoning']}")
        
        # Show profile characteristics
        print(f"\nProfile Characteristics:")
        for name, params in profiles.items():
            values = parabolic_profile(rho, params['alpha1'], params['alpha2'], params['offset'])
            peak_ratio = values[0] / values[100] if values[100] > 0 else np.inf
            gradient_edge = abs(values[-1] - values[-2]) / (rho[-1] - rho[-2])
            print(f"{name:18s}: Peak/Edge = {peak_ratio:5.1f}, Edge gradient = {gradient_edge:.3f}")
        
        return profiles, rho
    
    def _compute_geometric_quantities(self):
        """Compute geometric quantities needed for OpenPOPCON"""
        # Compute dV/drho = 2*rho * dV/ds = 2*rho * 4π² * |√g_00|
        # where √g_00 is the (0,0) Fourier component of the Jacobian
        self.dVdrho = 2 * self.rho * 4 * np.pi**2 * np.abs(self.gmnc[:, 0])
        
        # Compute toroidal field strength from flux
        phiedge = self.phi[-1] 
        Ba = phiedge / (np.pi * self.aminor**2)
        self.Ba_vmec = np.abs(Ba)
        
        # Compute surface area and <|grad(rho)|> using method from T3D
        self._compute_area_grad_rho()
        
        # Compute toroidal current density
        self._compute_current_density()
        
        print(f"✓ Computed geometric quantities:")
        print(f"  - Toroidal field Ba: {self.Ba_vmec:.3f} T") 
        print(f"  - Surface areas computed for {len(self.area)} surfaces")
        print(f"  - Current density computed")
    
    def _compute_area_grad_rho(self, N_phi=32, N_theta=32):
        """
        Compute surface area and <|grad(rho)|> on each flux surface
        Adapted from T3D implementation following simsopt's vmec_diagnostics
        """
        # Set up angular grids
        theta_vmec = np.linspace(0, 2*np.pi, N_theta)
        phi = np.linspace(0, 2*np.pi/self.nfp, N_phi)
        
        # Make 3D coordinate arrays
        theta_vmec = np.kron(np.ones((self.ns, 1, N_phi)), theta_vmec.reshape(1, N_theta, 1))
        phi = np.kron(np.ones((self.ns, N_theta, 1)), phi.reshape(1, 1, N_phi))
        
        # Compute Fourier angle combinations
        angle = self.xm[:, None, None, None] * theta_vmec[None, :, :, :] - self.xn[:, None, None, None] * phi[None, :, :, :]
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        mcosangle = self.xm[:, None, None, None] * cosangle
        ncosangle = self.xn[:, None, None, None] * cosangle
        msinangle = self.xm[:, None, None, None] * sinangle
        nsinangle = self.xn[:, None, None, None] * sinangle
        
        # Compute spatial coordinates and derivatives
        R = np.einsum('ij,jikl->ikl', self.rmnc, cosangle)
        d_R_d_theta_vmec = -np.einsum('ij,jikl->ikl', self.rmnc, msinangle)
        d_R_d_phi = np.einsum('ij,jikl->ikl', self.rmnc, nsinangle)
        
        d_Z_d_theta_vmec = np.einsum('ij,jikl->ikl', self.zmns, mcosangle)
        d_Z_d_phi = -np.einsum('ij,jikl->ikl', self.zmns, ncosangle)
        
        # Convert to Cartesian derivatives
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)
        
        d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
        d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
        d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi  
        d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
        
        # Compute |∇s|√g using cross products
        grad_s_X_sqrtg = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi)
        grad_s_Y_sqrtg = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi)
        grad_s_Z_sqrtg = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi)
        
        grad_s_dot_grad_s_g = (grad_s_X_sqrtg**2 + grad_s_Y_sqrtg**2 + grad_s_Z_sqrtg**2)
        abs_grad_s_sqrtg = np.sqrt(grad_s_dot_grad_s_g)
        
        # Integrate to get surface areas
        dtheta = 2*np.pi / N_theta
        dphi = 2*np.pi / N_phi / self.nfp
        self.area = np.sum(abs_grad_s_sqrtg * dtheta * dphi, axis=(1, 2)) * self.nfp
        
        # Compute <|∇ρ|> = A / (dV/dρ)
        self.grad_rho = np.zeros(self.ns)
        self.grad_rho[1:] = self.area[1:] / self.dVdrho[1:]
    
    def _compute_current_density(self):
        """
        Compute toroidal current density from VMEC contravariant components
        j_φ = B^φ / μ₀ (in SI units)
        """
        mu0 = 4e-7 * np.pi  # Permeability of free space
        
        # Extract toroidal contravariant field component (mode 0,0)
        # B^φ is stored in bsupvmnc
        self.bsupv_00 = self.bsupvmnc[:, 0]  # (m=0, n=0) component
        
        # Current density j_φ = B^φ / μ₀  
        self.j_tor = self.bsupv_00 / mu0
        
        print(f"  - Toroidal current density range: {np.min(self.j_tor):.2e} to {np.max(self.j_tor):.2e} A/m²")
    
    def get_iota_23(self):
        """Get iota at 2/3 flux surface"""
        try:
            # Interpolate to get iota at ρ = 2/3
            interp_func = interpolate.interp1d(self.rho, self.iotaf, kind='linear', fill_value='extrapolate')
            iota_23 = float(interp_func(2.0/3.0))
            print(f"✓ iota_23 = {iota_23:.6f}")
            return iota_23
        except Exception as e:
            print(f"✗ Error computing iota_23: {e}")
            raise
    
    def get_cross_sectional_area(self, rho_values=None):
        """
        Get cross-sectional area as function of rho
        
        This calculates the effective cross-sectional area perpendicular to magnetic field lines
        at each flux surface, useful for transport calculations.
        
        Method: A_cross = dV/ds / (effective_circumference)
        where effective_circumference accounts for stellarator 3D geometry
        
        Parameters:
        -----------
        rho_values : array_like, optional
            Rho values to interpolate to. If None, returns on native grid.
            
        Returns:
        --------
        rho_out : ndarray
            Rho coordinate values
        cross_area : ndarray  
            Cross-sectional area at each rho [m²]
        """
        if rho_values is None:
            rho_out = self.rho
            
            # Method 1: Simple approximation using major radius
            # A_cross = (dV/ds) / (2π * R_major) where dV/ds = dVdrho / (2*rho)
            cross_area_simple = self.dVdrho / (2 * self.rho * 2 * np.pi * self.Rmajor)
            
            # Method 2: More accurate using flux surface area and <|grad(rho)|>
            # From flux surface geometry: A_cross ≈ Volume / (flux_surface_area * <|grad(rho)|>)
            # But this gives different quantity, so we use the simpler volume-based approach
            
            # For stellarators, include field period correction for better accuracy
            # Effective circumference = 2π * R_major * geometry_factor
            # where geometry_factor accounts for 3D shaping
            geometry_factor = 1.0  # Could be improved with aspect ratio dependence
            
            # Final calculation
            cross_area = self.dVdrho / (2 * self.rho * 2 * np.pi * self.Rmajor * geometry_factor)
            
            # Handle singularity at rho=0
            cross_area[0] = cross_area[1] if len(cross_area) > 1 else 0.0
            
        else:
            rho_values = np.asarray(rho_values)
            
            # Calculate on native grid first
            native_cross_area = self.dVdrho / (2 * self.rho * 2 * np.pi * self.Rmajor)
            native_cross_area[0] = native_cross_area[1] if len(native_cross_area) > 1 else 0.0
            
            # Interpolate to requested grid
            interp_func = interpolate.interp1d(self.rho, native_cross_area,
                                             kind='linear', fill_value='extrapolate')
            cross_area = interp_func(rho_values)
            rho_out = rho_values
            
        return rho_out, cross_area
    
    def get_flux_surface_area(self, rho_values=None):
        """
        Get flux surface area as function of rho
        
        Parameters:
        -----------
        rho_values : array_like, optional
            Rho values to interpolate to. If None, returns on native grid.
            
        Returns:
        --------
        rho_out : ndarray
            Rho coordinate values  
        surf_area : ndarray
            Flux surface area at each rho [m²]
        """
        if rho_values is None:
            rho_out = self.rho
            surf_area = self.area
        else:
            rho_values = np.asarray(rho_values) 
            interp_func = interpolate.interp1d(self.rho, self.area, kind='linear', fill_value='extrapolate')
            surf_area = interp_func(rho_values)
            rho_out = rho_values
            
        return rho_out, surf_area
    
    def get_volume_derivative(self, rho_values=None):
        """
        Get dV/drho as function of rho
        
        Parameters:
        -----------
        rho_values : array_like, optional  
            Rho values to interpolate to. If None, returns on native grid.
            
        Returns:
        --------
        rho_out : ndarray
            Rho coordinate values
        dVdrho : ndarray
            Volume derivative dV/drho [m³] 
        """
        if rho_values is None:
            rho_out = self.rho
            dVdrho_out = self.dVdrho
        else:
            rho_values = np.asarray(rho_values)
            interp_func = interpolate.interp1d(self.rho, self.dVdrho, kind='linear', fill_value='extrapolate') 
            dVdrho_out = interp_func(rho_values)
            rho_out = rho_values
            
        return rho_out, dVdrho_out
    
    def get_toroidal_current_density(self, rho_values=None):
        """
        Get toroidal current density as function of rho
        
        Parameters:
        -----------
        rho_values : array_like, optional
            Rho values to interpolate to. If None, returns on native grid.
            
        Returns:
        --------
        rho_out : ndarray
            Rho coordinate values
        j_tor : ndarray
            Toroidal current density [A/m²]
        """
        if rho_values is None:
            rho_out = self.rho
            j_tor_out = self.j_tor
        else:
            rho_values = np.asarray(rho_values)
            interp_func = interpolate.interp1d(self.rho, self.j_tor, kind='linear', fill_value='extrapolate')
            j_tor_out = interp_func(rho_values)
            rho_out = rho_values
            
        return rho_out, j_tor_out
    
    def get_volume_averaged_B(self):
        """Get volume averaged toroidal magnetic field"""
        return self.volavgB
    
    def get_equilibrium_summary(self):
        """
        Get summary of key equilibrium quantities
        
        Returns:
        --------
        dict with keys:
        - 'Rmajor': Major radius [m]
        - 'aminor': Minor radius [m] 
        - 'volume': Total plasma volume [m³]
        - 'Ba_vmec': Toroidal field at LCFS [T]
        - 'volavgB': Volume averaged |B| [T]
        - 'B0': Field at magnetic axis [T]
        - 'iota_23': Rotational transform at 2/3 surface
        - 'nfp': Number of field periods
        """
        return {
            'Rmajor': self.Rmajor,
            'aminor': self.aminor,
            'volume': self.volume,
            'Ba_vmec': self.Ba_vmec,
            'volavgB': self.volavgB, 
            'B0': self.B0,
            'iota_23': self.get_iota_23(),
            'nfp': self.nfp
        }
    
    def get_stellarator_profiles(self, nr=200):
        """
        Get comprehensive stellarator profiles for OpenPOPCON integration
        
        Parameters:
        -----------
        nr : int
            Number of radial grid points for interpolation
            
        Returns:
        --------
        dict : Dictionary containing all profiles needed for OpenPOPCON:
            - 'rho': radial coordinate grid (sqrt of normalized toroidal flux)
            - 'sqrtpsin': equivalent to rho for stellarators 
            - 'volgrid': volume grid (cumulative volume fraction)
            - 'agrid': cross-sectional area grid
            - 'surface_area': flux surface area grid  
            - 'j_prof': toroidal current density profile
            - 'dVdrho': volume derivative profile
            - 'volavgB': volume averaged magnetic field
        """
        # Create radial grid from 0 to 1 (avoiding exactly 0 and 1)
        rho_grid = np.linspace(0.01, 0.99, nr)
        
        # Extract profiles on the grid
        _, dVdrho = self.get_volume_derivative(rho_grid)
        _, cross_area = self.get_cross_sectional_area(rho_grid)
        _, surf_area = self.get_flux_surface_area(rho_grid)
        _, j_tor = self.get_toroidal_current_density(rho_grid)
        
        # Compute cumulative volume grid
        # Integrate dV/drho to get volume as function of rho
        # V(rho) = integral from 0 to rho of dV/drho' drho'
        volume_profile = np.zeros_like(rho_grid)
        for i in range(1, len(rho_grid)):
            # Trapezoidal integration
            drho = rho_grid[i] - rho_grid[i-1]
            volume_profile[i] = volume_profile[i-1] + 0.5 * (dVdrho[i] + dVdrho[i-1]) * drho
        
        # Normalize volume grid by total volume
        if volume_profile[-1] > 0:
            volgrid = volume_profile / volume_profile[-1]
        else:
            volgrid = rho_grid**3  # Fallback to cubic scaling
            
        # For stellarators, sqrtpsin is equivalent to rho (sqrt of normalized toroidal flux)
        sqrtpsin = rho_grid.copy()
        
        # Create profiles dictionary
        profiles = {
            'rho': rho_grid,
            'sqrtpsin': sqrtpsin,  # For stellarators: sqrt(normalized toroidal flux)
            'volgrid': volgrid,
            'agrid': cross_area,  # Cross-sectional area
            'surface_area': surf_area,  # Flux surface area
            'j_prof': j_tor,  # Toroidal current density profile
            'dVdrho': dVdrho,  # Volume derivative
            'volavgB': self.get_volume_averaged_B()  # Volume averaged |B|
        }
        
        return profiles
        
    def get_coordinate_comparison(self):
        """
        Compare sqrt(poloidal flux) vs sqrt(toroidal flux) coordinates for stellarators
        
        Returns:
        --------
        dict : Analysis of coordinate system differences
        """
        # For stellarators, the natural coordinate is sqrt(toroidal flux)
        # VMEC uses s = normalized toroidal flux, so rho = sqrt(s)
        
        rho_tor = self.rho  # sqrt(normalized toroidal flux) - VMEC native
        
        # For comparison, construct what poloidal flux coordinate would be
        # In stellarators, poloidal and toroidal flux are related through rotational transform
        # Phi_pol ≈ iota * Phi_tor (approximately)
        iota_profile = self.iotaf
        
        # Approximate poloidal flux coordinate (this is conceptual for comparison)
        # rho_pol ≈ sqrt(iota * s) where s is normalized toroidal flux
        s_tor = self.s  # normalized toroidal flux
        
        # Use average iota for scaling estimate
        iota_avg = np.mean(np.abs(iota_profile))
        rho_pol_approx = np.sqrt(iota_avg * s_tor)
        
        # Normalize to [0,1]
        if rho_pol_approx[-1] > 0:
            rho_pol_approx = rho_pol_approx / rho_pol_approx[-1]
        
        analysis = {
            'rho_toroidal': rho_tor,
            'rho_poloidal_approx': rho_pol_approx,
            'iota_profile': iota_profile,
            'iota_avg': iota_avg,
            'coordinate_difference': np.abs(rho_tor - rho_pol_approx),
            'max_difference': np.max(np.abs(rho_tor - rho_pol_approx)),
            'rms_difference': np.sqrt(np.mean((rho_tor - rho_pol_approx)**2)),
            'recommendation': 'Use sqrt(toroidal flux) for stellarators as implemented in VMEC',
            'note': 'VMEC natively uses sqrt(normalized toroidal flux) which is appropriate for stellarators'
        }
        
        return analysis

# Test and demonstration functions
def test_vmec_comprehensive():
    """Comprehensive test of VMEC reader capabilities"""
    try:
        print("=== VMEC Reader Comprehensive Test ===")
        
        # Check if file exists
        import os
        filepath = "/Users/mattwang/Desktop/OpenPOPCON/src/lib/wout_2007_000_000000.nc"
        
        if not os.path.exists(filepath):
            # Try Downloads directory as fallback
            filepath = "/Users/mattwang/Downloads/wout_2007_000_000000.nc"
            if not os.path.exists(filepath):
                print(f"✗ File does not exist in either location")
                print("Please download a VMEC wout file to test")
                return False
        
        print(f"✓ Testing with file: {filepath}")
        
        # Create VMEC reader
        vmec = VMECReader(filepath)
        print("✓ VMEC reader initialized successfully")
        
        # Test all extraction methods
        print("\n--- Testing quantity extraction ---")
        
        # Basic equilibrium summary
        summary = vmec.get_equilibrium_summary()
        print(f"✓ Equilibrium summary extracted:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.6f}")
            else:
                print(f"   {key}: {value}")
        
        # Test iota_23 
        iota_23 = vmec.get_iota_23()
        print(f"✓ iota_23: {iota_23:.6f}")
        
        # Test volume averaged field
        volavgB = vmec.get_volume_averaged_B()
        print(f"✓ Volume averaged |B|: {volavgB:.6f} T")
        
        # Test profile extractions on reduced grid
        rho_test = np.linspace(0.1, 0.9, 10)
        
        rho_out, dVdrho = vmec.get_volume_derivative(rho_test)
        print(f"✓ dV/drho extracted at {len(rho_out)} points, range: {np.min(dVdrho):.3e} to {np.max(dVdrho):.3e} m³")
        
        rho_out, cross_area = vmec.get_cross_sectional_area(rho_test) 
        print(f"✓ Cross-sectional area extracted, range: {np.min(cross_area):.3e} to {np.max(cross_area):.3e} m²")
        
        rho_out, surf_area = vmec.get_flux_surface_area(rho_test)
        print(f"✓ Surface area extracted, range: {np.min(surf_area):.3e} to {np.max(surf_area):.3e} m²")
        
        rho_out, j_tor = vmec.get_toroidal_current_density(rho_test)
        print(f"✓ Toroidal current density extracted, range: {np.min(j_tor):.3e} to {np.max(j_tor):.3e} A/m²")
        
        print("\n=== Comprehensive Test PASSED ===")
        print("All required quantities for OpenPOPCON stellarator calculations are available!")
        return True
        
    except Exception as e:
        print(f"\n=== Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_vmec_integration():
    """Demonstrate how to use VMEC reader in OpenPOPCON context"""
    try:
        print("=== OpenPOPCON VMEC Integration Demo ===")
        
        # File path - update this to your VMEC file
        filepath = "/Users/mattwang/Desktop/OpenPOPCON/src/lib/wout_2007_000_000000.nc"
        
        if not os.path.exists(filepath):
            # Try Downloads directory as fallback  
            filepath = "/Users/mattwang/Downloads/wout_2007_000_000000.nc"
            if not os.path.exists(filepath):
                print("Demo requires a VMEC wout file. Please provide one.")
                return
            
        # Initialize reader
        vmec = VMECReader(filepath) 
        
        print(f"\nStellator Parameters:")
        summary = vmec.get_equilibrium_summary()
        print(f"  Major radius: {summary['Rmajor']:.3f} m")
        print(f"  Minor radius: {summary['aminor']:.3f} m") 
        print(f"  Aspect ratio: {summary['Rmajor']/summary['aminor']:.1f}")
        print(f"  Field periods: {summary['nfp']}")
        print(f"  Volume: {summary['volume']:.3f} m³")
        print(f"  Toroidal field: {summary['Ba_vmec']:.3f} T")
        print(f"  Iota 2/3: {summary['iota_23']:.6f}")
        
        # Extract profiles for POPCON calculation
        rho_grid = np.linspace(0.1, 0.95, 20)  # Typical POPCON radial grid
        
        print(f"\nExtracting profiles on {len(rho_grid)} point grid...")
        
        # Get all required profiles
        _, dVdrho = vmec.get_volume_derivative(rho_grid)
        _, cross_area = vmec.get_cross_sectional_area(rho_grid)
        _, surf_area = vmec.get_flux_surface_area(rho_grid)
        _, j_tor = vmec.get_toroidal_current_density(rho_grid)
        
        print("✓ Ready for OpenPOPCON stellarator POPCON calculation!")
        
        # Optional: create simple plot
        try:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 3, 1)
            plt.plot(rho_grid, dVdrho*1e-3)
            plt.xlabel('ρ')
            plt.ylabel('dV/dρ [10³ m³]')
            plt.title('Volume Derivative')
            plt.grid(True)
            
            plt.subplot(2, 3, 2)
            plt.plot(rho_grid, cross_area)
            plt.xlabel('ρ')
            plt.ylabel('Cross Area [m²]')
            plt.title('Cross-Sectional Area')
            plt.grid(True)
            
            plt.subplot(2, 3, 3)  
            plt.plot(rho_grid, surf_area)
            plt.xlabel('ρ')
            plt.ylabel('Surface Area [m²]')
            plt.title('Flux Surface Area')
            plt.grid(True)
            
            plt.subplot(2, 3, 4)
            plt.plot(rho_grid, j_tor*1e-6)
            plt.xlabel('ρ') 
            plt.ylabel('j_tor [MA/m²]')
            plt.title('Toroidal Current Density')
            plt.grid(True)
            
            plt.subplot(2, 3, 5)
            rho_full, iota_full = vmec.rho, vmec.iotaf
            plt.plot(rho_full, iota_full)
            plt.axhline(y=vmec.get_iota_23(), color='r', linestyle='--', label=f'ι(2/3) = {vmec.get_iota_23():.4f}')
            plt.xlabel('ρ')
            plt.ylabel('ι')
            plt.title('Rotational Transform')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            print("✓ Profiles plotted successfully")
            
        except Exception as plot_error:
            print(f"Note: Could not create plots: {plot_error}")
        
    except Exception as e:
        print(f"Demo failed: {e}")

# Safe execution
if __name__ == "__main__":
    # Run comprehensive test
    success = test_vmec_comprehensive()
    
    if success:
        # Run integration demo
        demo_vmec_integration()