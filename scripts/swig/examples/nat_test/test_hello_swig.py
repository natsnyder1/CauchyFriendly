import numpy as np 
import ctypes


import hello_swig as hs 
import numpy as np 
import ctypes as ct 

def test_numpy():
    n = 10
    arr_sum = hs.get_array_sum(np.arange(n, dtype=np.float64))
    y = np.zeros(n, dtype=np.float64)
    hs.double_it(np.arange(n), y) # inplace return
    X = hs.fill_array_cumsum(5) # returns output array
    W, Z, foo, bar = hs.return_multiple_things(n) # returns two numpy arrays and two integer values
    foo = 9

# Function Callback to C from Python we know works 
def py_callback(i, s):
    print( 'py_callback(%d, %s)'%(i, s) )

def test_py_callback():
    py_callback_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
    f = py_callback_type(py_callback)
    f_ptr = ctypes.cast(f, ctypes.c_void_p).value
    hs.use_callback(f_ptr)

class CPoint(ct.Structure):
    _fields_ = [("x", ct.c_int),
                ("y", ct.c_int),
                ("z", ct.c_int)]    

def py_callback2( PointStruct ):
    print("HELLO FROM PYTHON CALLBACK!")
    PointStruct.contents.x = 8
    PointStruct.contents.y = 9
    PointStruct.contents.z = 10


def test_struct_callback():
    py_callback_type = ct.CFUNCTYPE(None, ct.POINTER(CPoint))
    f = py_callback_type(py_callback2)
    f_ptr = ctypes.cast(f, ctypes.c_void_p).value
    hs.initialize_foo_system(5, f_ptr)

class CDynUpdateLight(ct.Structure):

    _fields_ = [("n", ct.c_int),
                ("pncc", ct.c_int),
                ("p", ct.c_int),
                ("Phi", ct.POINTER(ct.c_double)),
                ("Gamma", ct.POINTER(ct.c_double)),
                ("H", ct.POINTER(ct.c_double)),
                ("beta", ct.POINTER(ct.c_double)),
                ("gamma", ct.POINTER(ct.c_double)),
                ("x", ct.POINTER(ct.c_double)),
                ("step", ct.c_int),
                ("other", ct.c_void_p)
                ]

def py_callback3( CDynUpdateLightStruct ):
    duc = CDynUpdateLightStruct
    print("Python says: Doing the big test!")
    print("n is: {}, pncc is: {}, p is: {}".format(duc.contents.n, duc.contents.pncc, duc.contents.p))
    duc.contents.Phi[0] = 1.5
    duc.contents.Phi[2] = 3.5
    duc.contents.Phi[4] = 5.5
    duc.contents.Phi[6] = 7.5

    duc.contents.Gamma[0] = 1.5
    duc.contents.Gamma[1] = 3.5
    duc.contents.Gamma[2] = 5.5

def test_struct_of_arrays_callback():
    py_callback_type = ct.CFUNCTYPE(None, ct.POINTER(CDynUpdateLight))
    f = py_callback_type(py_callback3)
    f_ptr = ctypes.cast(f, ctypes.c_void_p).value
    x = np.array([1.0,2.0,3.0,4.0])
    hs.initialize_NL_system(4, 2, 1, x, f_ptr)

def test_multiple_callback_handoff():
    
    py_callback_type1 = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)
    f1 = py_callback_type1(py_callback)
    f_ptr1 = ctypes.cast(f1, ctypes.c_void_p).value
    
    py_callback_type3 = ct.CFUNCTYPE(None, ct.POINTER(CDynUpdateLight))
    f3 = py_callback_type3(py_callback3)
    f_ptr3 = ctypes.cast(f3, ctypes.c_void_p).value

    x = np.array([1.0,2.0,3.0,4.0])
    #hs.initialize_NL_system_v2(4, 2, 1, x, f_ptr1, f_ptr3)
    hs.initialize_NL_system_v2(4,2,1,x,f_ptr1,f_ptr3)

    foo = 9



if __name__ == "__main__":
    test_numpy()
    test_py_callback()
    test_struct_callback()
    test_struct_of_arrays_callback()
    test_multiple_callback_handoff()