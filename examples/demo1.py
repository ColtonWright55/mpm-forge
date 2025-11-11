import numpy as np
import warp as wp

wp.init()


@wp.kernel
def simple_kernel(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), c: wp.array(dtype=float)):

    # get thread index
    tid = wp.tid()

    # load two vec3s
    x = a[tid]
    y = b[tid]

    # compute the dot product between vectors
    r = wp.dot(x, y)

    # write result back to memory
    c[tid] = r


a = wp.array(np.arange(10), dtype=wp.float32)
b = wp.array(np.arange(10), dtype=wp.float32)
# multiply a by 2 and add these two arrays together element-wise
c = 2.0 * a + b
# multiply c by 10.0 in-place
c *= 10.0
print(c)


# wp.launch(kernel=simple_kernel, # kernel to launch
#           dim=1024,             # number of threads
#           inputs=[a, b, c],     # parameters
#           device="cuda")        # execution device
