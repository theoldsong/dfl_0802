import os

import numpy as np


os.environ['PYOPENCL_NO_CACHE'] = '1'
import pyopencl as cl

class Tensor4D:
    def __init__(self):
        pass


class Layer:
    def __init__(self):
        pass

class Input2D(Layer):
    def __init__(self, ch, h, w):
        super().__init__()
        self.ch = ch
        self.h = h
        self.w = w

class Conv2D(Layer):
    def __init__(self, kernel_size, strides, padding):
        super().__init__()


    def forward(self, inp_t):
        pass

    def backward(self):
        pass

    def compile_program(self):
        #\#define K_SIZE {self.kernel_size}

        prg_s = \
"""
__kernel void sum(__global const float* inp, __global const float* K, __global float* outp)
{
    int idx = get_global_id(0);
    
    
    
    
}
"""
        self.cl_prg = cl.Program(ctx, prg_s).build()


def main():

    

    #.get_devices()[1].get_info( cl.device_info.MAX_MEM_ALLOC_SIZE)

    device = cl.get_platforms()[0].get_devices()[1]

    ctx = cl.Context(devices=[device])
    ctx_q = cl.CommandQueue(ctx)

    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=512*1024*1024)
    #ev = cl.enqueue_fill_buffer(ctx_q, buf, np.array([1], dtype=np.float32), 0, 4*1024)

    buf_np = np.zeros( (1024,), dtype=np.float32 )
    cl.enqueue_copy(ctx_q, buf_np, buf )

    prg_s = \
"""
__kernel void conv2d(__global const float* inp, __global const float* K, __global float* outp)
{
    int gid = get_global_id(0);
    
}
"""
    cl_prg = cl.Program(ctx, prg_s).build()

    """

    3,16,3,3
    1,3,64,64
    """

    prg_test_s = \
"""
__kernel void test(__global float* outp)
{
    int v = 4;
    
    outp[0] = (float) (v >= 5);    
}
"""
    cl_prg_test = cl.Program(ctx, prg_test_s).build()
    
    
    buf_test = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=4)
    buf_test_np = np.zeros( (1,), dtype=np.float32 )
    cl_prg_test.test(ctx_q, (1,), None, buf_test)
    cl.enqueue_copy(ctx_q, buf_test_np, buf_test )
    print(buf_test_np)
    import code
    code.interact(local=dict(globals(), **locals()))
    #===================================================


if __name__ == "__main__":
    main()
