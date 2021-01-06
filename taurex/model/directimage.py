from .emission import EmissionModel
from taurex.constants import PI
import numpy as np

class DirectImageModel(EmissionModel):
    """
    A forward model for direct imaging of exo-planets

    Parameters
    ----------

    planet: :class:`~taurex.data.planet.Planet`, optional
        Planet model, default planet is Jupiter

    star: :class:`~taurex.data.stellar.star.Star`, optional
        Star model, default star is Sun-like

    pressure_profile: :class:`~taurex.data.profiles.pressure.pressureprofile.PressureProfile`, optional
        Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
        and ``atm_max_pressure``

    temperature_profile: :class:`~taurex.data.profiles.temperature.tprofile.TemperatureProfile`, optional
        Temperature model, default is an :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
        profile at 1500 K

    chemistry: :class:`~taurex.data.profiles.chemistry.chemistry.Chemistry`, optional
        Chemistry model, default is
        :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry` with
        ``H2O`` and ``CH4``

    nlayers: int, optional
        Number of layers. Used if ``pressure_profile`` is not defined.

    atm_min_pressure: float, optional
        Pressure at TOA. Used if ``pressure_profile`` is not defined.

    atm_max_pressure: float, optional
        Pressure at BOA. Used if ``pressure_profile`` is not defined.

    ngauss: int, optional
        Number of gaussian quadrature points, default = 4

    """

    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 ngauss=4,
                 linear_scaling=[],
                 linear_regions=[]
                 ):
        super().__init__(planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure,
                         ngauss=ngauss)

        self._linear_scale = linear_scaling
        self._linear_regions = linear_regions

    def collect_linear_scaling_terms(self):

        bounds = [1.0, 2.0]
        for idx, val in enumerate(self._linear_scale):
            point_num = idx+1
            param_name = 'scale_factor_{}'.format(point_num)
            param_latex = '$S_{}$'.format(point_num)

            def read_point(self, idx=idx):
                return self._linear_scale[idx]

            def write_point(self, value, idx=idx):
                self._linear_scale[idx] = value

            fget_point = read_point
            fset_point = write_point
            self.debug('FGet_location %s', fget_point)
            default_fit = False
            self.add_fittable_param(param_name, param_latex, fget_point,
                                    fset_point, 'log', default_fit, bounds)

    def compute_final_flux(self, f_total):
        star_distance_meters = self._star.distance*3.08567758e16

        SDR = pow((star_distance_meters/3.08567758e16), 2)
        SDR = 1.0
        planet_radius = self._planet.fullRadius

        return((f_total * (planet_radius**2) * 2.0 * PI) /
               (4 * PI * (star_distance_meters**2))) * SDR


    def model(self, wngrid=None, cutoff_grid=True):
        native_grid, absorp, tau, extra = super().model(wngrid, cutoff_grid)

        linear_regions = np.array([0]+self._linear_regions)

        wlgrid = 10000/native_grid
        final_scale = np.zeros_like(native_grid)
        for i in range(1,linear_regions.shape[0],1):
            x = i-1
            value_to_set = self._linear_scale[x]
            filter_wn = (wlgrid >= linear_regions[x]) & (wlgrid  < linear_regions[i])
            print(filter_wn)
            if np.any(filter_wn):
                print(final_scale[filter_wn])
                final_scale[filter_wn] = value_to_set
        absorp*=final_scale

        return native_grid, absorp, tau, extra



