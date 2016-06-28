from ctypes import byref, cdll, c_int
import ctypes
import numpy as np
import cython

def runModel(params):
	print len(params)
	lualib = ctypes.CDLL("/home/lili/torch/install/lib/libluajit.so", mode=ctypes.RTLD_GLOBAL)
	l = cdll.LoadLibrary('HPOptim/libcluaf.so')

	l.computeCost.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]


	arr_a = (ctypes.c_char_p * len(params))()
	arr_b = (ctypes.c_float * len(params))()


	i = 0
	for name, param in params.iteritems():
		arr_a[i] = name
		arr_b[i] = param
		print(arr_a[i])
		print(arr_b[i])
		i = i + 1
		print i 

	result = (ctypes.c_float)()

 

	l.computeCost(arr_a,arr_b,ctypes.c_int(len(params)),result)

	print 'the evaluation value is'
	print result.value
	return result.value

def main(job_id, params):
    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    return runModel(params)
