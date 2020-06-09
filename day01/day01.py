import numpy as np

arr1 = np.array([1, 2, 3])
print(arr1)
arr2 = np.ones(3)
print(arr2)
arr3 = np.zeros(3)
print(arr3)
arr4 = np.random.random(3)
print(arr4)

a = np.array([0, 1, 2, 3])
print(type(a))
#a1 = np.array(0, 1, 2, 3) # error
a2 = np.array((0, 1, 2, 3))
print(a2)
c = np.array([3, '4.5'])
print(c)

c1 = np.arange(6)
print(c1)

# -2 to 1, step 0.5
d = np.arange(-2, 1, 0.5)
print(d)

# 0-2, 5 pieces, including 2
e = np.linspace(0, 2, 5)
print(e)

f = np.ones([2, 3])
print(f)
g = np.zeros([2, 3])
print(g)

h = np.eye(2)
print(h)

arr_eye = np.array([1, 2, 3])
print(arr_eye)
eye_matrix = np.diag(arr_eye)
print(eye_matrix)

d_arr = np.array([1, 2, 3])
print(d_arr.ndim)
print(d_arr.shape)
print(d_arr.size)
print(d_arr.dtype)

# Advance numpy
data = np.array([1, 2])
print(data)
ones = np.ones(2)
print(ones)

result = data + ones;
print(result)
result1 = data - ones
print(result1)
result2 = data * data
print(result2)
result3 = data / data
print(result3)

ret1 = data * 1.6
print(ret1)

ret2 = data ** 2
print(ret2)

ret3 = np.sin(data)
print(ret3)

ret4 = np.exp(data)
print(ret4)

ret5 = np.sqrt(data)
print(ret5)

data_a = np.array([20, 30, 40, 50])
data_a <= 35
print(data_a)

a_a = [20, 30 ,40 ,50]
b_b = list(range(2, 6))

ab = a_a + b_b
print(ab)
#a_b = a_a - b_b # errorv
#print(a_b)
ret_a = a_a * 2
print(ret_a)
#a_a / b_b # error

data_index = np.array([1, 2, 3])
print(data_index[0])
print(data_index[1])
print(data_index[0:2])
print(data_index[1:])
print(data_index.max())
print(data_index.min())
print(data_index.sum())

two_dim_arr = np.array([[1, 2], [3, 4]])
print(two_dim_arr)
two_dim_ones = np.ones((3, 2))
print(two_dim_ones)
two_dim_zeros = np.zeros((3, 2))
print(two_dim_zeros)
two_dim_randoms = np.random.random((3, 2))
print(two_dim_randoms)

data_arr_op = np.array([[1, 2], [3, 4]])
data_arr_ones = np.ones((2, 2))
new_result = data_arr_op + data_arr_ones
print(new_result)

ones_row = np.ones(2)
print(ones_row)

data_2_3 = np.array([[1, 2], [3, 4], [5, 6]])
print(data_2_3)
op_result = data_2_3 + ones_row
print(op_result)

data_0_1 = data_2_3[0, 1]
print(data_0_1)
data_1_3 = data_2_3[1:3]
print(data_1_3)
data_0_2_0 = data_2_3[0:2, 0]
print(data_0_2_0)

data_t = data_2_3.transpose()
print(data_t)

data_1_6 = np.arange(1, 7)
print(data_1_6)
data_re_2_3 = data_1_6.reshape(2, 3)
print(data_re_2_3)
data_re_3_2 = data_1_6.reshape(3, 2)
print(data_re_3_2)
data_re_3_miner_1 = data_1_6.reshape(3, -1)
print(data_re_3_miner_1)
data_1 = data_1_6.ravel()
print(data_1)

data_arr_high = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(data_arr_high)

data_ones_high = np.ones((2, 4, 3))
print(data_ones_high)

data_zeros_high = np.zeros((2, 4, 3))
print(data_zeros_high)

data_random_high = np.random.random((2, 4, 3))
print(data_random_high)
print(data_random_high.ndim)
print(data_random_high.shape)