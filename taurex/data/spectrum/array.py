from .spectrum import BaseSpectrum
import numpy as np

class ArraySpectrum(BaseSpectrum):
    """
    Loads an observed spectrum from an array and computes bin
    edges and bin widths. Spectrum shape(nbins, 3-4) with 3-4 columns with ordering:
        1. wavelength
        2. spectral data
        3. error
        4. (optional) bin width

    If no bin width is present then they are computed.

    Parameters
    -----------
    filename: string
        Path to observed spectrum file.

    """

    def __init__(self, spectrum):
        super().__init__(self.__class__.__name__)

        self._obs_spectrum = spectrum
        self._bin_widths = None
        self._bin_edges = None

        self._sort_spectrum()
        self._process_spectrum()

    def _sort_spectrum(self):
        self._obs_spectrum = self._obs_spectrum[self._obs_spectrum[:, 0].argsort(axis=0)[::-1]]

    def _process_spectrum(self):
        """
        Seperates out the observed data, error, grid and binwidths
        from raw file array. If bin widths are not present then they are
        calculated here
        """
        if self.rawData.shape[1] == 4:
            self._bin_widths = self._obs_spectrum[:, 3]
            obs_wl = self.wavelengthGrid[::-1]
            obs_bw = self.binWidths[::-1]

            bin_edges = np.zeros(shape=(len(self.binWidths)*2,))

            bin_edges[0::2] = obs_wl - obs_bw/2
            bin_edges[1::2] = obs_wl + obs_bw/2
            # bin_edges[-1] = obs_wl[-1]-obs_bw[-1]/2.

            self._bin_edges = bin_edges[::-1]
        else:
            self.manual_binning()

    @property
    def rawData(self):
        """Data read from file"""
        return self._obs_spectrum

    @property
    def spectrum(self):
        """The spectrum itself"""
        return self._obs_spectrum[:, 1]

    @property
    def wavelengthGrid(self):
        """Wavelength grid in microns"""
        return self.rawData[:, 0]

    @property
    def wavenumberGrid(self):
        """Wavenumber grid in cm-1"""
        return 10000/self.wavelengthGrid

    @property
    def binEdges(self):
        """ Bin edges"""
        return 10000/self._bin_edges
    @property
    def binWidths(self):
        """bin widths"""
        return 10000/(self.wavelengthGrid-self._bin_widths/2) - 10000/(self.wavelengthGrid+self._bin_widths/2)

    @property
    def errorBar(self):
        """ Error bars for the spectrum"""
        return self.rawData[:, 2]

    def manual_binning(self):
        """
        Performs the calculation of bin edges when none are present
        """
        bin_edges = []
        wl_grid = self.wavelengthGrid

        bin_edges.append(wl_grid[0]-(wl_grid[1]-wl_grid[0])/2)
        for i in range(wl_grid.shape[0]-1):
            bin_edges.append(wl_grid[i]+(wl_grid[i+1]-wl_grid[i])/2.0)
        bin_edges.append((wl_grid[-1]-wl_grid[-2])/2.0 + wl_grid[-1])
        self._bin_edges = np.array(bin_edges)
        self._bin_widths = np.abs(np.diff(self._bin_edges))
