# Configuration File for Filterbank Design

This configuration file, written in TOML format, is used to set up parameters for a scientific script that designs filterbanks. The following sections explain each configuration option and how to use it.

## Table of Contents
1. [Input/Output](#inputoutput)
2. [Filterbank Configuration](#filterbank-configuration)
   - [Resonance Frequencies](#resonance-frequencies)
   - [Filter Parameters](#filter-parameters)
   - [Transmission Lines](#transmission-lines)
3. [Frequency Range](#frequency-range)

---

## Input/Output

### `[input_output]`

Defines the folders used for input and output data.

- **`input_data_folder`**: *(string)* Path to the directory containing input data.
- **`output_data_folder`**: *(string)* Path to the directory where output data will be saved.

Example:
```toml
[input_output]
input_data_folder = "/path/to/input"
output_data_folder = "/path/to/output"
```

## Filterbank Configuration

### `[filterbank]`

This section defines the main parameters of the filterbank, including resonance frequencies, quality factors, and the properties of associated transmission lines.

### Resonance Frequencies

Specify either a list of specific resonance frequencies (`f0`) **or** a range of resonance frequencies (`f0_min` and `f0_max`).

- **`f0`**: *(array of floats)* A list of resonance frequencies in Hz.
- **`f0_min`**: *(float)* Minimum resonance frequency in Hz.
- **`f0_max`**: *(float)* Maximum resonance frequency in Hz.

**Note**: Use either `f0` or both `f0_min` and `f0_max`, but not both options simultaneously.

Example:
```toml
# Option 1: Specify list of resonance frequencies
# f0 = [100.0, 200.0, 300.0]

# Option 2: Specify range of resonance frequencies
f0_min = 100.0
f0_max = 500.0
```

### Filter Parameters

Provide exactly **two out of these three parameters** to define the filter characteristics:

- **`Ql`**: *(float)* Loaded Q-factor.
- **`R`**: *(float)* Spectral resolution.
- **`OS`**: *(float)* Oversampling factor.

**Note**: Only two of these values should be specified.

Example:
```toml
Ql = 50.0  # loaded Q-factor
R = 10.0   # spectral resolution
# OS = 2.0  # oversampling factor (commented out to use Ql and R)
```

### Transmission Lines

Each transmission line is defined in a nested table within `[filterbank.transmissionlines]`. There are exactly three required transmission lines:

1. **`signal`**
2. **`resonator`**
3. **`MKID`**

Each transmission line includes:

- **`Z0`**: *(float)* Characteristic impedance in ohms.
- **`eps_eff`**: *(float)* Effective dielectric constant.
- **`Qi`**: *(float, optional)* Loss factor.

Example:
```toml
[filterbank.transmissionlines.signal]
Z0 = 50.0
eps_eff = 3.0
# Qi = 1000.0

[filterbank.transmissionlines.resonator]
Z0 = 75.0
eps_eff = 2.5
# Qi = 1500.0

[filterbank.transmissionlines.MKID]
Z0 = 60.0
eps_eff = 2.8
# Qi = 2000.0
```

## Frequency Range

### `[frequency_range]`

Defines the frequency range over which calculations will be performed.

- **`f_min`**: *(float)* Minimum frequency in Hz.
- **`f_max`**: *(float)* Maximum frequency in Hz.
- **`nf`**: *(int)* Number of points in the frequency range.

Example:
```toml
[frequency_range]
f_min = 90.0
f_max = 600.0
nf = 1000
```

---

## Example Complete Configuration

```toml
[input_output]
input_data_folder = "/path/to/input"
output_data_folder = "/path/to/output"

[filterbank]
f0_min = 100.0
f0_max = 500.0
Ql = 50.0
R = 10.0

[filterbank.transmissionlines.signal]
Z0 = 50.0
eps_eff = 3.0
# Qi = 1000.0

[filterbank.transmissionlines.resonator]
Z0 = 75.0
eps_eff = 2.5
# Qi = 1500.0

[filterbank.transmissionlines.MKID]
Z0 = 60.0
eps_eff = 2.8
# Qi = 2000.0

[frequency_range]
f_min = 90.0
f_max = 600.0
nf = 1000
```

## Notes

- Ensure that exactly two of `Ql`, `R`, and `OS` are specified in the `[filterbank]` section.
- Only one of `f0` (list) or the `f0_min` and `f0_max` range should be defined in `[filterbank]`.
- Optional parameters like `Qi` for each transmission line can be omitted if not required.
