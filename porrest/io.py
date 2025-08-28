"""This module contains functions to read different data formats."""
from typing import Literal
import os
from scipy.io import loadmat
import pyfar as pf
import numpy as np
import scipy.io as spio


def read_ita(
        fname: str,
        data_type: Literal['data' , 'signal'] = 'data',
    ) -> tuple[pf.Signal | pf.TimeData | pf.FrequencyData, dict]:
    """Read a *.ita file.

    Note that coordinate information stored in the *.ita file is not
    returned due to unsupported MATLAB class structures in Python.

    Parameters
    ----------
    fname : str
        The filename.
    data_type : Literal, optional
        The return type of data. Can be 'data' or 'signal'. Default is 'data'.

    Returns
    -------
    data : pyfar.Signal or pyfar.TimeData, pyfar.FrequencyData
        The data contained in the *.ita file.
    user_data : dict
        Additional user data contained in the ``userData`` field of the *.ita
        file.
    """
    matfile = loadmat(
        os.path.join(fname),
        struct_as_record=False, squeeze_me=True, appendmat=False)
    mfiledata = matfile['ITA_TOOLBOX_AUDIO_OBJECT']

    if data_type == 'signal':
        fft_norm = 'none' if mfiledata.signalType == 'energy' else 'rms'
        data = pf.Signal(
            mfiledata.data.T.copy(), mfiledata.samplingRate,
            domain=mfiledata.domain, fft_norm=fft_norm,
            comment=mfiledata.comment)
    elif data_type == 'data':
        domain = mfiledata.domain
        if domain == 'time':
            times = np.arange(mfiledata.data.shape[0])/mfiledata.samplingRate
            data = pf.TimeData(
                mfiledata.data.T.copy(), times,
                comment=mfiledata.comment)
        elif domain == 'freq':
            freqs = np.linspace(
                0, mfiledata.samplingRate/2, mfiledata.data.shape[0])
            data = pf.FrequencyData(
                mfiledata.data.T.copy(), freqs,
                comment=mfiledata.comment)

    if hasattr(mfiledata, 'userData'):
        user_data = _todict(mfiledata.userData)
    else:
        user_data = {}

    user_data['samplingRate'] = mfiledata.samplingRate

    return data, user_data


def _todict(matobj):
    """Recursive construction of nested dictionaries from matobjects.
    """
    output_dictionary = {}
    for fieldname in matobj._fieldnames:
        elem = matobj.__dict__[fieldname]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            output_dictionary[fieldname] = _todict(elem)
        else:
            output_dictionary[fieldname] = elem
    return output_dictionary
