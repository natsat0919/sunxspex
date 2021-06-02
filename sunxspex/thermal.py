import copy

import astropy.units as u
import numpy as np
from scipy import interpolate, stats
from sunpy.data import manager

from sunxspex.io import load_chianti_continuum, load_chianti_lines_lite, load_xray_abundances

doc_string_params = """
Parameters
----------
energy_edges: `astropy.units.Quantity`
    The edges of the energy bins in a 1D N+1 quantity.

temperature: `astropy.units.Quantity`
    The temperature of the plasma.
    Can be scalar or 1D of any length. If not scalar, the flux for each temperature
    will be calculated. The first dimension of the output flux will correspond
    to temperature.

emission_measure: `astropy.units.Quantity`
    The emission measure of the plasma at each temperature.
    Must be same length as temperature or scalar.

abundance_type: `str` (optional)
    Abundance type to use.  Options are:
        1. cosmic
        2. sun_coronal - default abundance
        3. sun_coronal_ext
        4. sun_hybrid
        5. sun_hybrid_ext
        6. sun_photospheric
        7. mewe_cosmic
        8. mewe_solar
    The values for each abundance type is stored in the global
    variable DEFAULT_ABUNDANCES which is generated by `setup_default_abundances`
    function. To load different default values for each abundance type,
    see the docstring of that function.

relative_abundances: `tuple` of `tuples` of (`int`, `float`) (optional)
    The relative abundances of different elements as a fraction of their
    default abundances defined by abundance_type.
    Each tuple represents the values for a given element.
    The first entry represents the atomic number of the element.
    The second entry represents the axis represents the fraction by which the
    element's default abundance should be scaled.

observer_distance: `astropy.units.Quantity` (Optional)
    The distance between the source and the observer.
    Default=1 AU.

Returns
-------
flux: `astropy.units.Quantity`
    The photon flux as a function of temperature and energy.
"""

def setup_continuum_parameters(filename=None):
    """
    Define continuum intensities as a function of temperature.

    Intensities are set as global variables and used in
    calculation of spectra by other functions in this module. They are in
    units of per volume emission measure at source, i.e. they must be
    divided by 4 * pi R**2 to be converted to physical values where
    R**2 is observer distance.

    Intensities are derived from output from the CHIANTI atomic physics database.
    The default CHIANTI data used here are collected from
    `https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v71.sav`.
    This includes contributions from thermal bremsstrahlung and two-photon interactions.
    To use a different file, provide the URL/file location via the filename kwarg,
    e.g. to include only thermal bremsstrahlung, set the filename kwarg to
    'https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_cont_1_250_v70_no2photon.sav'

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the CHIANTI IDL save file to be used.
    """
    global _CONTINUUM_GRID
    if filename:
        with manager.override_file("chianti_continuum", uri=filename):
            cont_info = load_chianti_continuum()
    else:
        cont_info = load_chianti_continuum()
    _CONTINUUM_GRID = {}
    _CONTINUUM_GRID["abundance index"] = cont_info.element_index.data
    _CONTINUUM_GRID["sorted abundance index"] = np.sort(_CONTINUUM_GRID["abundance index"])

    T_grid = (cont_info.temperature.data * cont_info.attrs["units"]["temperature"]).to(u.K)
    _CONTINUUM_GRID["log10T"] = np.log10(T_grid.value)
    _CONTINUUM_GRID["T_keV"] = T_grid.to_value(u.keV, equivalencies=u.temperature_energy())

    wavelength = cont_info.wavelength.data * cont_info.attrs["units"]["wavelength"]
    dwave_AA = (cont_info.attrs["wavelength_edges"][1:] -
                cont_info.attrs["wavelength_edges"][:-1]).to_value(u.AA)
    _CONTINUUM_GRID["E_keV"] = wavelength.to_value(u.keV, equivalencies=u.spectral())
    _CONTINUUM_GRID["energy bin widths keV"] = (
        _CONTINUUM_GRID["E_keV"] * dwave_AA / wavelength.to_value(u.AA))

    _CONTINUUM_GRID["intensity"] = cont_info.data
    _CONTINUUM_GRID["intensity unit"] = cont_info.attrs["units"]["data"]
    _CONTINUUM_GRID["intensity description"] = (
        "Intensity is stored as photons per keV per unit emission measure at the source.  "
        "It (and its unit) therefore must be multipled by emission measure and "
        "divided by 4 * pi * observer_distance**2 to get observed values.")

    # Define valid input temperature and energy ranges
    _define_valid_input_ranges(_CONTINUUM_GRID["E_keV"], T_grid.value, "_LINE_GRID")


def setup_line_parameters(filename=None):
    """Define line intensities as a function of temperature for calculating line emission.

    Line intensities are set as global variables and used in the
    calculation of spectra by other functions in this module. They are in
    units of per unit emission measure at source, i.e. they must be
    divided by 4 pi R**2 (where R is the observer distance) and
    multiplied by emission measure to be converted to physical values at the observer.

    Line intensities are derived from output from the CHIANTI atomic
    physics database. The default CHIANTI data used here is collected from
    `https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/chianti_lines_1_10_v71.sav`.
    To use a different file, provide the URL/file location via the filename kwarg.

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the CHIANTI IDL save file to be used.
    """
    global _LINE_GRID
    if filename:
        with manager.override_file("chianti_lines", uri=filename):
            line_info = load_chianti_lines_lite()
    else:
        line_info = load_chianti_lines_lite()
    _LINE_GRID = {}
    _LINE_GRID["intensity"] = np.array(line_info.data)
    _LINE_GRID["intensity unit"] = line_info.attrs["units"]["data"]
    _LINE_GRID["intensity description"] = (
        "Intensity is stored as photons per unit emission measure at the source.  "
        "It (and its unit) therefore must be multipled by emission measure and "
        "divided by 4 * pi * observer_distance**2 to get observed values.")
    _LINE_GRID["line_peaks_keV"] = (
            line_info.peak_energy.data * line_info.attrs["units"]["peak_energy"]).to_value(
                u.keV, equivalencies=u.spectral())
    _LINE_GRID["log10T"] = line_info.logT.data
    _LINE_GRID["abundance index"] = line_info.attrs["element_index"]
    _LINE_GRID["line atomic numbers"] = line_info.atomic_number.data

    # Define valid input temperature and energy ranges
    _define_valid_input_ranges(_LINE_GRID["line_peaks_keV"], 10**_LINE_GRID["log10T"],
                               "_CONTINUUM_GRID")


def _define_valid_input_ranges(local_grid_E_keV, local_grid_T_K, other_grid_name):
    global VALID_ENERGY_RANGE_KEV, VALID_TEMPERATURE_RANGE_K
    if other_grid_name in globals():
        VALID_ENERGY_RANGE_KEV = (
            min(_CONTINUUM_GRID["E_keV"].min(), _LINE_GRID["line_peaks_keV"].min()),
            max(_CONTINUUM_GRID["E_keV"].max(), _LINE_GRID["line_peaks_keV"].max()))
        VALID_TEMPERATURE_RANGE_K = (
            min(10**_CONTINUUM_GRID["log10T"].min(), 10**_LINE_GRID["log10T"].min()),
            max(10**_CONTINUUM_GRID["log10T"].max(), _LINE_GRID["log10T"].max()))
    else:
        VALID_ENERGY_RANGE_KEV = (local_grid_E_keV.min(), local_grid_E_keV.max())
        VALID_TEMPERATURE_RANGE_K = (local_grid_T_K.min(), local_grid_T_K.max())


def setup_default_abundances(filename=None):
    """
    Read default abundance values into global variable.

    By default, data is read from the following file:
    https://hesperia.gsfc.nasa.gov/ssw/packages/xray/dbase/chianti/xray_abun_file.genx
    To load data from a different file, see Notes section.

    Parameters
    ----------
    filename: `str` (optional)
        URL or file location of the .genx abundance file to be used.
    """
    global DEFAULT_ABUNDANCES
    if filename:
        with manager.override_file("xray_abundance", uri=filename):
            DEFAULT_ABUNDANCES = load_xray_abundances()
    else:
        DEFAULT_ABUNDANCES = load_xray_abundances()

# Read line, continuum and abundance data into global variables.
setup_continuum_parameters()
setup_line_parameters()
setup_default_abundances()
DEFAULT_ABUNDANCE_TYPE = "sun_coronal_ext"

@u.quantity_input(energy_edges=u.keV,
                  temperature=u.K,
                  emission_measure=(u.cm**(-3), u.cm**(-5)),
                  observer_distance=u.cm)
def thermal_emission(energy_edges,
                     temperature,
                     emission_measure,
                     abundance_type=DEFAULT_ABUNDANCE_TYPE,
                     relative_abundances=None,
                     observer_distance=(1*u.AU).to(u.cm)):
    f"""Calculate the thermal X-ray spectrum (lines + continuum) from the solar atmosphere.

    The flux is calculated as a function of temperature and emission measure.
    Which continuum mechanisms are included --- free-free, free-bound, or two-photon --- are
    determined by the file from which the continuum parameters are loaded.
    To change the file used, see the setup_continuum_parameters() function.

    {doc_string_params}"""
    energy_edges_keV, temperature_K = _sanitize_inputs(energy_edges, temperature)
    # Calculate abundances
    abundances = _calculate_abundances(abundance_type, relative_abundances)
    # Calculate fluxes.
    continuum_flux = _continuum_emission(energy_edges_keV, temperature_K, abundances)
    line_flux = _line_emission(energy_edges_keV, temperature_K, abundances)
    flux = ((continuum_flux + line_flux) * emission_measure /
                (4 * np.pi * observer_distance**2))
    if temperature.isscalar and emission_measure.isscalar:
        flux = flux[0]
    return flux


@u.quantity_input(energy_edges=u.keV,
                  temperature=u.K,
                  emission_measure=(u.cm**(-3), u.cm**(-5)),
                  observer_distance=u.cm)
def continuum_emission(energy_edges,
              temperature,
              emission_measure,
              abundance_type=DEFAULT_ABUNDANCE_TYPE,
              relative_abundances=None,
              observer_distance=(1*u.AU).to(u.cm)):
    f"""Calculate the thermal X-ray continuum emission from the solar atmosphere.

    The emission is calculated as a function of temperature and emission measure.
    Which continuum mechanisms are included --- free-free, free-bound, or two-photon --- are
    determined by the file from which the comtinuum parameters are loaded.
    To change the file used, see the setup_continuum_parameters() function.

    {doc_string_params}"""
    # Convert inputs to known units and confirm they are within range.
    energy_edges_keV, temperature_K = _sanitize_inputs(energy_edges, temperature)
    # Calculate abundances
    abundances = _calculate_abundances(abundance_type, relative_abundances)
    # Calculate flux.
    flux = _continuum_emission(energy_edges_keV, temperature_K, abundances)
    flux *= emission_measure / (4 * np.pi * observer_distance**2)
    if temperature.isscalar and emission_measure.isscalar:
        flux = flux[0]
    return flux


def _continuum_emission(energy_edges_keV, temperature_K, abundances):
    """
    Calculates emission-measure-normalized X-ray continuum spectrum at the source.

    Output must be multiplied by emission measure and divided by 4*pi*observer_distance**2
    to get physical values.
    Which continuum mechanisms are included --- free-free, free-bound, or two-photon --- are
    determined by the file from which the comtinuum parameters are loaded.
    To change the file used, see the setup_continuum_parameters() function.

    Parameters
    ----------
    energy_edges_keV: 1-D array-like
        Boundaries of contiguous spectral bins in units on keV.

    temperature_K: 1-D array-like
        The temperature(s) of the plasma in unit of K.  Must not be a scalar.

    abundances: 1-D `numpy.array` of same length a DEFAULT_ABUNDANCES.
        The abundances for the all the elements.
    """
    # Handle inputs and derive some useful parameters from them
    log10T_in = np.log10(temperature_K)
    T_in_keV = temperature_K / 11604518  # Convert temperature from K to keV.
    # Get energy bins centers based on geometric mean.
    energy_gmean_keV = stats.gmean(np.vstack((energy_edges_keV[:-1], energy_edges_keV[1:])))

    # Mask Unwanted Abundances
    abundance_mask = np.zeros(len(abundances))
    abundance_mask[_CONTINUUM_GRID["abundance index"]] = 1.
    abundances *= abundance_mask

    #####  Calculate Continuum Intensity Summed Over All Elements
    #####  For Each Temperature as a function of Energy/Wavelength ######
    # Before looping over temperatures, let's perform the calculations that are
    # used over again in the for loop.
    # 1. If many temperatures are input, convolve intensity grid with abundances for all
    # temperatures here.  If only a few temperatures are input, do this step only
    # when looping over input temperatures.  This minimizes computation.
    n_tband = 3
    n_t_grid = len(_CONTINUUM_GRID["log10T"])
    n_temperature_K = len(temperature_K)
    n_thresh = n_temperature_K * n_tband
    if n_thresh >= n_t_grid:
        intensity_per_em_at_source_allT = np.zeros(_CONTINUUM_GRID["intensity"].shape[1:])
        for i in range(0, n_t_grid):
            intensity_per_em_at_source_allT[i] = np.matmul(
                abundances[_CONTINUUM_GRID["sorted abundance index"]],
                _CONTINUUM_GRID["intensity"][:, i])
    # 2. Add dummy axes to energy and temperature grid arrays for later vectorized operations.
    repeat_E_grid = _CONTINUUM_GRID["E_keV"][np.newaxis, :]
    repeat_T_grid = _CONTINUUM_GRID["T_keV"][:, np.newaxis]
    dE_grid_keV = _CONTINUUM_GRID["energy bin widths keV"][np.newaxis, :]
    # 3. Identify the indices of the temperature bins containing each input temperature and
    # the bins above and below them.  For each input temperature, these three bins will
    # act as a temperature band over which we'll interpolate the continuum emission.
    selt = np.digitize(log10T_in, _CONTINUUM_GRID["log10T"]) - 1
    tband_idx = selt[:, np.newaxis] + np.arange(n_tband)[np.newaxis, :]

    # Finally, loop over input temperatures and calculate continuum emission for each.
    flux = np.zeros((n_temperature_K, len(energy_gmean_keV)))
    for j, logt in enumerate(log10T_in):
        # If not already done above, calculate continuum intensity summed over
        # all elements as a function of energy/wavelength over the temperature band.
        if n_thresh < n_t_grid:
            element_intensities_per_em_at_source = _CONTINUUM_GRID["intensity"][:, tband_idx[j]]
            intensity_per_em_at_source = np.zeros(element_intensities_per_em_at_source.shape[1:])
            for i in range(0, n_tband):
                intensity_per_em_at_source[i] = np.matmul(
                    abundances[_CONTINUUM_GRID["sorted abundance index"]],
                    element_intensities_per_em_at_source[:, i])
        else:
            intensity_per_em_at_source = intensity_per_em_at_source_allT[tband_idx[j]]

        ##### Calculate Continuum Intensity at Input Temperature  ######
        ##### Do this by interpolating the normalized temperature component
        ##### of the intensity grid to input temperature(s) and then rescaling.
        # Calculate normalized temperature component of the intensity grid.
        exponent = (repeat_E_grid / repeat_T_grid[tband_idx[j]])
        exponential = np.exp(np.clip(exponent, None, 80))
        gaunt = intensity_per_em_at_source / dE_grid_keV * exponential
        # Interpolate the normalized temperature component of the intensity grid the the
        # input temperature.
        flux[j] = _interpolate_continuum_intensities(
            gaunt, _CONTINUUM_GRID["log10T"][tband_idx[j]], _CONTINUUM_GRID["E_keV"], energy_gmean_keV, logt)
    # Rescale the interpolated intensity.
    flux = flux * np.exp(-(energy_gmean_keV[np.newaxis, :] / T_in_keV[:, np.newaxis]))

    # Put intensity into correct units.
    return flux * _CONTINUUM_GRID["intensity unit"]


@u.quantity_input(energy_edges=u.keV,
                  temperature=u.K,
                  emission_measure=(u.cm**(-3), u.cm**(-5)),
                  observer_distance=u.cm)
def line_emission(energy_edges,
         temperature,
         emission_measure,
         abundance_type=DEFAULT_ABUNDANCE_TYPE,
         relative_abundances=None,
         observer_distance=(1*u.AU).to(u.cm)):
    """
    Calculate thermal line emission from the solar corona.

    {docstring_params}"""
    # Convert inputs to known units and confirm they are within range.
    energy_edges_keV, temperature_K = _sanitize_inputs(energy_edges, temperature)
    # Calculate abundances
    abundances = _calculate_abundances(abundance_type, relative_abundances)

    flux = _line_emission(energy_edges_keV, temperature_K, abundances)
    flux *= emission_measure / (4 * np.pi * observer_distance**2)
    if temperature.isscalar and emission_measure.isscalar:
        flux = flux[0]
    return flux


def _line_emission(energy_edges_keV, temperature_K, abundances):
    """
    Calculates emission-measure-normalized X-ray line spectrum at the source.

    Output must be multiplied by emission measure and divided by 4*pi*observer_distance**2
    to get physical values.

    Parameters
    ----------
    energy_edges_keV: 1-D array-like
        Boundaries of contiguous spectral bins in units on keV.

    temperature_K: 1-D array-like
        The temperature(s) of the plasma in unit of K.  Must not be a scalar.

    abundances: 1-D `numpy.array` of same length a DEFAULT_ABUNDANCES.
        The abundances for the all the elements.
    """
    n_energy_bins = len(energy_edges_keV)-1
    n_temperatures = len(temperature_K)

    # Find indices of lines within user input energy range.
    energy_roi_indices = np.logical_and(_LINE_GRID["line_peaks_keV"] >= energy_edges_keV.min(),
                                        _LINE_GRID["line_peaks_keV"] <= energy_edges_keV.max())
    n_energy_roi_indices = energy_roi_indices.sum()
    # If there are emission lines within the energy range of interest, compile spectrum.
    if n_energy_roi_indices > 0:
        # Mask Unwanted Abundances
        abundance_mask = np.zeros(len(abundances))
        abundance_mask[_LINE_GRID["abundance index"]] = 1.
        abundances *= abundance_mask
        # Extract only the lines within the energy range of interest.
        line_abundances = abundances[_LINE_GRID["line atomic numbers"][energy_roi_indices] - 2]
        # Above magic number of of -2 is comprised of:
        # a -1 to account for the fact that index is atomic number -1, and
        # another -1 because abundance index is offset from abundance index by 1.

        ##### Calculate Line Intensities within the Input Energy Range #####
        # Calculate abundance-normalized intensity of each line in energy range of
        # interest as a function of energy and temperature.
        line_intensity_grid = _LINE_GRID["intensity"][energy_roi_indices]
        line_intensities = _calculate_abundance_normalized_line_intensities(
            np.log10(temperature_K), line_intensity_grid, _LINE_GRID["log10T"])
        # Scale line intensities by abundances to get true line intensities.
        line_intensities *= line_abundances

        ##### Weight Line Emission So Peak Energies Maintained Within Input Energy Binning #####
        # Split emission of each line between nearest neighboring spectral bins in
        # proportion such that the line centroids appear at the correct energy
        # when averaged over neighboring bins.
        # This has the effect of appearing to double the number of lines as regards
        # the dimensionality of the line_intensities array.
        line_peaks_keV = _LINE_GRID["line_peaks_keV"][energy_roi_indices]
        split_line_intensities, line_spectrum_bins = _weight_emission_bins_to_line_centroid(
            line_peaks_keV, energy_edges_keV, line_intensities)

        #### Calculate Flux #####
        # Use binned_statistic to determine which spectral bins contain
        # components of line emission and sum over those line components
        # to get the total emission is each spectral bin.
        flux = stats.binned_statistic(line_spectrum_bins, split_line_intensities,
                                      "sum", n_energy_bins, (0, n_energy_bins-1)).statistic
    else:
        flux = np.zeros((n_temperatures, n_energy_bins))

    # Scale flux by observer distance, emission measure and spectral bin width
    # and put into correct units.
    energy_bin_widths = (energy_edges_keV[1:] - energy_edges_keV[:-1]) * u.keV
    flux = (flux * _LINE_GRID["intensity unit"] / energy_bin_widths)

    return flux


def _interpolate_continuum_intensities(data_grid, log10T_grid, energy_grid_keV, energy_keV, log10T):
    # Determine valid range based on limits of intensity grid's spectral extent
    # and the normalized temperature component of intensity.
    n_tband = len(log10T_grid)
    vrange, = np.where(data_grid[0] > 0)
    for i in range(1, n_tband):
        vrange_i, = np.where(data_grid[i] > 0)
        if len(vrange) < len(vrange_i):
            vrange = vrange_i
    data_grid = data_grid[:, vrange]
    energy_grid_keV = energy_grid_keV[vrange]
    energy_idx, = np.where(energy_keV < energy_grid_keV.max())

    # Interpolate temperature component of intensity and derive continuum intensity.
    flux = np.zeros(energy_keV.shape)
    if len(energy_idx) > 0:
        energy_keV = energy_keV[energy_idx]
        cont0 = interpolate.interp1d(energy_grid_keV, data_grid[0])(energy_keV)
        cont1 = interpolate.interp1d(energy_grid_keV, data_grid[1])(energy_keV)
        cont2 = interpolate.interp1d(energy_grid_keV, data_grid[2])(energy_keV)
        # Calculate the continuum intensity as the weighted geometric mean
        # of the interpolated values across the temperature band of the
        # temperature component of intensity.
        logelog10T = np.log(log10T)
        x0, x1, x2 = np.log(log10T_grid)
        flux[energy_idx]  = np.exp(
            np.log(cont0) * (logelog10T - x1) * (logelog10T - x2) / ((x0 - x1) * (x0 - x2)) +
            np.log(cont1) * (logelog10T - x0) * (logelog10T - x2) / ((x1 - x0) * (x1 - x2)) +
            np.log(cont2) * (logelog10T - x0) * (logelog10T - x1) / ((x2 - x0) * (x2 - x1)) )
    return flux


def _calculate_abundance_normalized_line_intensities(logT, data_grid, line_logT_bins):
    """
    Calculates normalized line intensities at a given temperature using interpolation.

    Given a 2D array, say of line intensities, as a function of two parameters,
    say energy and log10(temperature), and a log10(temperature) value,
    interpolate the line intensities over the temperature axis and
    extract the intensities as a function of energy at the input temperature.

    Note that strictly speaking the code is agnostic to the physical properties
    of the axes and values in the array. All the matters is that data_grid
    is interpolated over the 2nd axis and the input value also corresponds to
    somewhere along that same axis. That value does not have to exactly correspond to
    the value of a column in the grid. This is accounted for by the interpolation.

    Parameters
    ----------
    logT: 1D `numpy.ndarray` of `float`.
        The input value along the 2nd axis at which the line intensities are desired.
        If multiple values given, the calculation is done for each and the
        output array has an extra dimension.

    data_grid: 2D `numpy.ndarray`
        Some property, e.g. line intensity, as function two parameters,
        e.g. energy (0th dimension) and log10(temperature in kelvin) (1st dimension).

    line_logT_bins: 1D `numpy.ndarray`
        The value along the 2nd axis at which the data are required,
        say a value of log10(temperature in kelvin).

    Returns
    -------
    interpolated_data: 1D or 2D `numpy.ndarray`
        The line intensities as a function of energy (1st dimension) at
        each of the input temperatures (0th dimension).
        Note that unlike the input line intensity table, energy here is the 0th axis.
        If there is only one input temperature, interpolated_data is 1D.

    """
    # Ensure input temperatures are in an array to consistent manipulation.
    n_temperatures = len(logT)

    # Get bins in which input temperatures belong.
    temperature_bins = np.digitize(logT, line_logT_bins)-1

    # For each input "temperature", interpolate the grid over the 2nd axis
    # using the bins corresponding to the input "temperature" and the two neighboring bins.
    # This will result in a function giving the data as a function of the 1st axis,
    # say energy, at the input temperature to sub-temperature bin resolution.
    interpolated_data = np.zeros((n_temperatures, data_grid.shape[0]))
    for i in range(n_temperatures):
        # Identify the "temperature" bin to which the input "temperature"
        # corresponds and its two nearest neighbors.
        indx = temperature_bins[i]-1+np.arange(3)
        # Interpolate the 2nd axis to produce a function that gives the data
        # as a function of 1st axis, say energy, at a given value along the 2nd axis,
        # say "temperature".
        get_intensities_at_logT = interpolate.interp1d(line_logT_bins[indx], data_grid[:, indx], kind="quadratic")
        # Use function to get interpolated_data as a function of the first axis at
        # the input value along the 2nd axis,
        # e.g. line intensities as a function of energy at a given temperature.
        interpolated_data[i, :] = get_intensities_at_logT(logT[i]).squeeze()[:]

    return interpolated_data


def _weight_emission_bins_to_line_centroid(line_peaks_keV, energy_edges_keV, line_intensities):
    """
    Split emission between neighboring energy bins such that averaged energy is the line peak.

    Given the peak energies of the lines and a set of the energy bin edges:
    1. Find the bins into which each of the lines belong.
    2. Calculate distance between the line peak energy and the
    center of the bin to which it corresponds as a fraction of the distance between
    the bin center the center of the next closest bin to the line peak energy.
    3. Assign the above fraction of the line intensity to the neighboring bin and
    the rest of the energy to the original bin.
    4. Add the neighboring bins to the array of bins containing positive emission.

    Parameters
    ----------
    line_peaks_keV: 1D `numpy.ndarray`
        The energy of the line peaks in keV.

    energy_peak_keV: 1D `numpy.ndarray`
        The edges of adjacent energy bins.
        Length must be n+1 where n is the number of energy bins.
        These energy bins may be referred to as 'spectrum energy bins' in comments.

    line_intensities: 2D `numpy.ndarray`
        The amplitude of the line peaks.
        The last dimension represents intensities of each line in line_peaks_keV while
        the first dimension represents the intensities as a function of another parameter,
        e.g. temperature.
        These intensities are the ones divided between neighboring bins as described above.

    Returns
    -------
    new_line_intensities: 2D `numpy.ndarray`
        The weighted line intensities including neigboring component for each line weighted
        such that total emission is the same, but the energy of each line averaged over the
        energy_edge_keV bins is the same as the actual line energy.

    new_iline: `numpy.ndarray`
        Indices of the spectrum energy bins to which emission from each line corresponds.
        This includes indices of the neighboring bin emission components.

    """
    # Get widths and centers of the spectrum energy bins.
    energy_bin_widths = energy_edges_keV[1:] - energy_edges_keV[:-1]
    energy_centers = energy_edges_keV[:-1] + energy_bin_widths/2
    energy_center_diffs = energy_centers[1:] - energy_centers[:-1]

    # For each line, find the index of the spectrum energy bin to which it corresponds.
    iline = np.digitize(line_peaks_keV, energy_edges_keV) - 1

    # Get the difference between each line energy and
    # the center of the spectrum energy bin to which is corresponds.
    line_deviations_keV = line_peaks_keV - energy_centers[iline]
    # Get the indices of the lines which are above and below their bin center.
    neg_deviation_indices, = np.where(line_deviations_keV < 0)
    pos_deviation_indices, = np.where(line_deviations_keV >= 0)
    # Discard bin indices at the edge of the spectral range if they should
    # be shared with a bin outside the energy range.
    neg_deviation_indices = neg_deviation_indices[np.where(iline[neg_deviation_indices] > 0)[0]]
    pos_deviation_indices = pos_deviation_indices[
        np.where(iline[pos_deviation_indices] <= (len(energy_edges_keV)-2))[0]]

    # Split line emission between the spectrum energy bin containing the line peak and
    # the nearest neighboring bin based on the proximity of the line energy to
    # the center of the spectrum bin.
    # Treat lines which are above and below the bin center separately as slightly
    # different indexing is required.
    new_line_intensities = copy.deepcopy(line_intensities)
    new_iline = copy.deepcopy(iline)
    if len(neg_deviation_indices) > 0:
        neg_line_intensities, neg_neighbor_intensities, neg_neighbor_iline = _weight_emission_bins(
            line_deviations_keV, neg_deviation_indices,
            energy_center_diffs, line_intensities, iline, negative_deviations=True)
        # Combine new line and neighboring bin intensities and indices into common arrays.
        new_line_intensities[:, neg_deviation_indices] = neg_line_intensities
        new_line_intensities = np.concatenate((new_line_intensities, neg_neighbor_intensities), axis=-1)
        new_iline = np.concatenate((new_iline, neg_neighbor_iline))

    if len(pos_deviation_indices) > 0:
        pos_line_intensities, pos_neighbor_intensities, pos_neighbor_iline = _weight_emission_bins(
            line_deviations_keV, pos_deviation_indices,
            energy_center_diffs, line_intensities, iline, negative_deviations=False)
        # Combine new line and neighboring bin intensities and indices into common arrays.
        new_line_intensities[:, pos_deviation_indices] = pos_line_intensities
        new_line_intensities = np.concatenate(
            (new_line_intensities, pos_neighbor_intensities), axis=-1)
        new_iline = np.concatenate((new_iline, pos_neighbor_iline))

    # Order new_line_intensities so neighboring intensities are next
    # to those containing the line peaks.
    ordd = np.argsort(new_iline)
    new_iline = new_iline[ordd]
    for i in range(new_line_intensities.shape[0]):
        new_line_intensities[i, :] = new_line_intensities[i, ordd]

    return new_line_intensities, new_iline


def _weight_emission_bins(line_deviations_keV, deviation_indices,
                          energy_center_diffs, line_intensities, iline,
                          negative_deviations=True):
    if negative_deviations is True:
        if not np.all(line_deviations_keV[deviation_indices] < 0):
            raise ValueError(
                "As negative_deviations is True, can only handle "
                "lines whose energy < energy bin center, "
                "i.e. all line_deviations_keV must be negative.")
        a = -1
        b = -1
    else:
        if not np.all(line_deviations_keV[deviation_indices] >= 0):
            raise ValueError(
                "As negative_deviations is not True, can only handle "
                "lines whose energy >= energy bin center, "
                "i.e. all line_deviations_keV must be positive.")
        a = 0
        b = 1

    # Calculate difference between line energy and the spectrum bin center as a
    # fraction of the distance between the spectrum bin center and the
    # center of the nearest neighboring bin.
    wghts = np.absolute(line_deviations_keV[deviation_indices]) / energy_center_diffs[iline[deviation_indices+a]]
    # Tile/replicate wghts through the other dimension of line_intensities.
    wghts = np.tile(wghts, tuple([line_intensities.shape[0]] + [1] * wghts.ndim))

    # Weight line intensitites.
    # Weight emission in the bin containing the line intensity by 1-wght,
    # since by definition wght < 0.5 and
    # add the line intensity weighted by wght to the nearest neighbor bin.
    # This will mean the intensity integrated over the two bins is the
    # same as the original intensity, but the intensity-weighted peak energy
    # is the same as the original line peak energy even with different spectrum energy binning.
    new_line_intensities = line_intensities[:, deviation_indices] * (1-wghts)
    neighbor_intensities = line_intensities[:, deviation_indices] * wghts
    neighbor_iline = iline[deviation_indices]+b

    return new_line_intensities, neighbor_intensities, neighbor_iline


def _sanitize_inputs(energy_edges, temperature):
    # Convert inputs to known units and confirm they are within range.
    energy_edges_keV = energy_edges.to_value(u.keV)
    temperature_K = temperature.to_value(u.K)
    if temperature.isscalar:
        temperature_K = np.array([temperature_K])
    # Confirm inputs are within valid ranges
    invalid_range_message = lambda parameter, min_, max_, unit: \
        f"Invalid {parameter} input. All input {parameter} values must be in range {min_} -- {max_} {unit}"
    if (energy_edges_keV.min() < VALID_ENERGY_RANGE_KEV[0] or
        energy_edges_keV.max() > VALID_ENERGY_RANGE_KEV[1]):
        raise ValueError(invalid_range_message("energy", VALID_ENERGY_RANGE_KEV[0],
                                               VALID_ENERGY_RANGE_KEV[1], "keV"))
    if (temperature_K.min() < VALID_TEMPERATURE_RANGE_K[0] or
        temperature_K.max() > VALID_TEMPERATURE_RANGE_K[1]):
        raise ValueError(invalid_range_message("energy", VALID_TEMPERATURE_RANGE_K[0] * 1e-6,
                                               VALID_TEMPERATURE_RANGE_K[0] * 1e-6, "MK"))
    return energy_edges_keV, temperature_K


def _calculate_abundances(abundance_type, relative_abundances):
    abundances = DEFAULT_ABUNDANCES[abundance_type].data
    if relative_abundances:
        # Convert input relative abundances to array where
        # first axis is atomic number, i.e == index + 1
        # Second axis is relative abundance value.
        rel_abund_array = np.array(relative_abundances).T
        # Confirm relative abundances are for valid elements and positive.
        min_abundance_z = DEFAULT_ABUNDANCES["atomic number"].min()
        max_abundance_z = DEFAULT_ABUNDANCES["atomic number"].max()
        if (rel_abund_array[0].min() < min_abundance_z or
            rel_abund_array[0].max() > max_abundance_z):
            raise ValueError("Relative abundances can only be set for elements with "
                             f"atomic numbers in range {min_abundance_z} -- {min_abundance_z}")
        if rel_abund_array[1].min() < 0:
            raise ValueError("Relative abundances cannot be negative.")
        rel_idx = np.rint(rel_abund_array[0]).astype(int) - 1
        rel_abund_values = np.ones(len(abundances))
        rel_abund_values[rel_idx] = rel_abund_array[1]
        abundances *= rel_abund_values
    return abundances
