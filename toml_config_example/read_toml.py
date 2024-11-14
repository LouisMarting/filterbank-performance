import toml

# Load the TOML configuration file
config = toml.load("config.toml")

# Accessing the configuration data
input_data_folder = config["input_output"]["input_data_folder"]
output_data_folder = config["input_output"]["output_data_folder"]

# Accessing filterbank parameters
filterbank = config["filterbank"]
f0 = filterbank.get("f0", None)
f0_min = filterbank.get("f0_min", None)
f0_max = filterbank.get("f0_max", None)
Ql = filterbank.get("Ql", None)
R = filterbank.get("R", None)
OS = filterbank.get("OS", None)

# Accessing each transmission line's parameters
transmissionlines = filterbank["transmissionlines"]

signal_line = transmissionlines["signal"]
signal_Z0 = signal_line["Z0"]
signal_eps_eff = signal_line["eps_eff"]
signal_Qi = signal_line.get("Qi", None)  # optional Qi

resonator_line = transmissionlines["resonator"]
resonator_Z0 = resonator_line["Z0"]
resonator_eps_eff = resonator_line["eps_eff"]
resonator_Qi = resonator_line.get("Qi", None)  # optional Qi

MKID_line = transmissionlines["MKID"]
MKID_Z0 = MKID_line["Z0"]
MKID_eps_eff = MKID_line["eps_eff"]
MKID_Qi = MKID_line.get("Qi", None)  # optional Qi

# Accessing frequency range parameters
frequency_range = config["frequency_range"]
f_min = frequency_range["f_min"]
f_max = frequency_range["f_max"]
nf = frequency_range["nf"]

# Print configuration (optional)
print("Input/Output Configuration:")
print("Input Folder:", input_data_folder)
print("Output Folder:", output_data_folder)
print("\nFilterbank Configuration:")
print("f0:", f0)
print("f0_min:", f0_min)
print("f0_max:", f0_max)
print("Ql:", Ql)
print("R:", R)
print("OS:", OS)

print("\nTransmission Lines:")
print("Signal Line - Z0:", signal_Z0, ", eps_eff:", signal_eps_eff, ", Qi:", signal_Qi)
print("Resonator Line - Z0:", resonator_Z0, ", eps_eff:", resonator_eps_eff, ", Qi:", resonator_Qi)
print("MKID Line - Z0:", MKID_Z0, ", eps_eff:", MKID_eps_eff, ", Qi:", MKID_Qi)

print("\nFrequency Range:")
print("f_min:", f_min)
print("f_max:", f_max)
print("nf:", nf)
