"""
The following code is used to make the SRM/counts data in consistent units from NuSTAR spectral data.
"""

import numpy as np
from astropy.time import Time, TimeDelta

from . import io

__all__ = ["flux_cts_spec", "_load_rmf"]


def flux_cts_spec(counts, counts_err, bin_size, exposure, tstart, tstop):
    """ Takes a .pha file and returns plotting information.

    Parameters
    ----------
    file : Str
            String for the .pha file of the spectrum under investigation.

    Returns
    -------
    The count rate per keV (cts), its error (cts_err) and corresponding time
    """

    #_, counts, syst_err, stat_err, exposure, tstart, tstop = io._read_xsm_pha_file(file)

    counts_shape = np.shape(counts)

    cts = np.zeros(counts_shape)

    cts_err = np.zeros(counts_shape)

    for i in range(counts_shape[0]):

        cts[i, :] = counts[i, :] / bin_size/ exposure[i]  # now in cts keV^-1 s^-1

        cts_err[i, :] = counts_err[i, :] / bin_size / exposure[i]
        #cts[i, :] = counts[i, :][count_bins_mask] / bin_size/ exposure[i]  # now in cts keV^-1 s^-1
        #cts_err[i, :] = counts_err[i, :][count_bins_mask] / bin_size / exposure[i]

    #time
    tstart_mjd=57754.0+tstart/86400.0
    tstop_mjd=57754.0+tstop/86400.0

    tstart_utc = Time(tstart_mjd, format='mjd', scale='utc')
    tstart_utc.format = 'isot'

    tstop_utc = Time(tstop_mjd, format='mjd', scale='utc')
    tstop_utc.format = 'isot'

    time_bins = np.concatenate((tstart_utc[:, None], tstop_utc[:, None]), axis=1)

    return cts, cts_err, time_bins


def _load_rmf(rmf_file):
        """ Extracts all information, mainly the redistribution matrix ([counts/photon]) from a given RMF file.

        Parameters
        ----------
        rmf_file : string
                The file path and name of the RMF file.

        Returns
        -------
        The lower/higher photon bin edges (e_lo_rmf, e_hi_rmf), the number of counts channels activated by each photon channel (ngrp),
        starting indices of the count channel groups (fchan), number counts channels from each starting index (nchan), the coresponding
        counts/photon value for each count and photon entry (matrix), and the redistribution matrix (redist_m: with rows of photon channels,
        columns of counts channels, and in the units of counts/photon).
        """

        count_bin_lo, count_bin_hi, photon_bin_lo, photon_bin_hi, matrix = io._read_xsm_rmf_file(rmf_file)

        #filter bin edges
        count_bins = np.stack((count_bin_lo, count_bin_hi), axis=-1)
        photon_bins = np.stack((photon_bin_lo, photon_bin_hi), axis=-1)

        #Restric count bins from 1 -15 keV, look at their IDL code 
        #count_bins_mask = (np.all(count_bins <= 15, axis = 1)&np.all(count_bins >= 1, axis = 1))
        #count_bins = count_bins[count_bins_mask, :]
        count_bins_mask = (np.all(count_bins <= 15, axis = 1)&np.all(count_bins >= 1.0, axis = 1))
        count_bins = count_bins[count_bins_mask, :]

        #Restric photon bins from 1 keV, look at their IDL code
        photon_bins_mask = np.all(photon_bins >= 1.0, axis = 1)
        photon_bins = photon_bins[photon_bins_mask, :]


        redist_m = np.zeros((len(matrix), len(matrix[0])))

        # We just copy each row from the redistribution matrix into an numpy matrix which will be our matrix
        for r in range(len(matrix)):
                redist_m[r, :] = matrix[r]
                
        #Now slice the matrix based on the energy conditions from the photon and count bins
        redist_m = redist_m[:, count_bins_mask]
        redist_m = redist_m[photon_bins_mask, :]

        return count_bins, photon_bins, count_bins_mask, photon_bins_mask, matrix, redist_m