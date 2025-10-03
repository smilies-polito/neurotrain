import torch
import numpy as np


####################
# SPIKER FUNCTIONS #
####################

def fixed_point( value, fp_dec, bitwidth):
	"""
	Convert a value to a fixed-point representation.
	"""
	quant = value * 2**fp_dec		            # Move as much fractional part as you want to the integer part
	return saturated_int(quant, bitwidth)	# Remove the residual fractional part and saturate the value 

def saturated_int( value, bitwidth):
	"""
	Convert a value to a saturated integer with the given bitwidth.
	"""
	return saturate(to_int(value), bitwidth)

def saturate( value, bitwidth):
	"""
	Saturate the value to fit in the range of a signed integer with the given bitwidth
	"""
	if type(value).__module__ == np.__name__ or type(value).__module__ == torch.__name__:	# If it's a numpy array or a torch tensor
		# Print the values out of range for debugging
		if type(value).__module__ == np.__name__:
			out_of_range = np.where((value > 2**(bitwidth-1)-1) | (value < -2**(bitwidth-1)))
			if out_of_range[0].size > 0:
				print(f"Values out of range in numpy array: {value[out_of_range]}")
		elif type(value).__module__ == torch.__name__:
			out_of_range = torch.where((value > 2**(bitwidth-1)-1) | (value < -2**(bitwidth-1)))
			if out_of_range[0].numel() > 0:
				print(f"Values out of range in torch tensor: {value[out_of_range]}")
				print(f"Ranges valid: [{-2**(bitwidth-1)}, {2**(bitwidth-1)-1}]")
		# Saturate the values
		value[value > 2**(bitwidth-1)-1] = 2**(bitwidth-1)-1	# Saturate the maximum value exploiting numpy or torch broadcasting
		value[value < -2**(bitwidth-1)] = -2**(bitwidth-1)		# Saturate the minimum value exploiting numpy or torch broadcasting
		return value.float()
	else: 								# If it's a standard Python number
		if value > 2**(bitwidth-1)-1:
			value = 2**(bitwidth-1)-1	# Saturate the maximum value
		elif value < -2**(bitwidth-1):
			value = -2**(bitwidth-1)	# Saturate the minimum value
		return float(value)

def to_int( value):
	"""
	Convert a value to an integer, preserving the type if it's a numpy or torch tensor
	"""
	if type(value).__module__ == np.__name__:		# If it's a numpy array
		quant = value.astype(int).astype(float)
	elif type(value).__module__ == torch.__name__:	# If it's a torch tensor
		quant = value.type(torch.int64).float()
	else:
		quant = float(int(value))					# If it's a standard Python number
	return quant


####################
# CUSTOM FUNCTIONS #
####################

def check_range(tensor, bitwidth, name=""):
    # Allowed integer range
	qmin = -2**(bitwidth-1)
	qmax =  2**(bitwidth-1)-1
	# Get min and max values of the tensor
	mn = tensor.min().item()
	mx = tensor.max().item()
	# Check if the tensor values are within the allowed range
	assert mn >= qmin and mx <= qmax, (
        f"{name} out of range: [{mx}, {mn}] "
        f"should be [{qmax}, {qmin}]"
    )

def clamp_int_(t: torch.Tensor, bitwidth: int):
	qmin = -(1 << (bitwidth - 1))
	qmax =  (1 << (bitwidth - 1)) - 1
	t.floor_()                    # remove decimal part in-place
	t.clamp_(qmin, qmax)          # in-place, keeps autograd history intact
	return t
