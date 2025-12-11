"""
Load OPS peak files from binary format and output as a dictionary of
numpy arrays.
"""

import numpy as np
import struct
import os
import warnings


def load_peak_file(filename: str) -> dict:
    """
    Load OPS peak file in binary format.

    Parameters
    ----------
    filename : str
        Path to the binary peak file.

    Returns
    -------
    dict
        A dictionary containing:

        - 'peak_height' : ndarray of int32
            Peak height in digitizer bins
        - 'peak_width' : ndarray of int32
            Peak width in unknown units
        - 'microseconds_since_previous_peak' : ndarray of int32
            Microseconds since previous peak
        - 'peak_time' : ndarray of float64
            Seconds since 1970 (epoch time)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    filesize = os.path.getsize(filename)
    init_size = int(np.ceil(filesize / 3))

    # Pre-allocate arrays
    peak = np.full(init_size, np.nan)
    width = np.full(init_size, np.nan)
    dt = np.full(init_size, np.nan)
    peak_time = np.full(init_size, np.nan)

    c = 0
    bytes_read = 0
    num_elements = 3

    with open(filename, "rb") as fd:
        while bytes_read < filesize:
            # Read array length (int32)
            array_len_bytes = fd.read(4)
            if len(array_len_bytes) < 4:
                break
            array_len = struct.unpack("<i", array_len_bytes)[0]

            # Read BBB seconds (double)
            bbb_seconds_bytes = fd.read(8)
            if len(bbb_seconds_bytes) < 8:
                break
            bbb_seconds = struct.unpack("<d", bbb_seconds_bytes)[0]

            bytes_read += 12

            # Read peak data (num_elements x array_len int32 values)
            data_size = num_elements * array_len * 4
            data_bytes = fd.read(data_size)
            if len(data_bytes) < data_size:
                break

            bytes_read += data_size

            # Unpack all int32 values
            peak_data = np.array(
                struct.unpack(f"<{num_elements * array_len}i", data_bytes)
            )
            peak_data = peak_data.reshape((array_len, num_elements))

            if peak_data.shape[1] < 3:
                warnings.warn("Data format unexpected: less than 3 columns")
                break

            # Calculate cumulative time
            mirco_seconds_from_start = np.cumsum(peak_data[:, 2])

            # Store data
            idx_slice = slice(c, c + array_len)
            peak[idx_slice] = peak_data[:, 0]
            width[idx_slice] = peak_data[:, 1]
            dt[idx_slice] = peak_data[:, 2]
            peak_time[idx_slice] = bbb_seconds + mirco_seconds_from_start / 1e6

            c += array_len

    # Trim to actual size
    return {
        "peak_height": peak[:c].astype(np.int32),
        "peak_width": width[:c].astype(np.int32),
        "microseconds_since_previous_peak": dt[:c].astype(np.int32),
        "peak_time": peak_time[:c].astype(np.float64),
    }
