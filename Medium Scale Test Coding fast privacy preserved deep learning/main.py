import P0
import P1
import numpy as np
import time

def convolution(p0,p1,convolution_x0,convolution_x1):#二维卷积层
    p0.padding()
    p1.padding()

    p0.expand(p0.feature_map,convolution_x0)
    p1.expand(p1.feature_map,convolution_x1)
    
    p0.compute_e_f(p0.expansion_feature_map,p0.expansion_convolution)
    p1.compute_e_f(p1.expansion_feature_map,p1.expansion_convolution)
    
    p0.compute_z()
    p1.compute_z()
    p0.sum_convolution()
    p1.sum_convolution()

def convolution_3d(p0,p1,convolution_x0,convolution_x1):#三维卷积层
    p0.convolution_3d_init(convolution_x0)
    p1.convolution_3d_init(convolution_x1)
    p0.padding_3d()
    p1.padding_3d()

    p0.expand_3d(p0.feature_map,convolution_x0)
    p1.expand_3d(p1.feature_map,convolution_x1)
    
    p0.compute_e_f(p0.expansion_feature_map,p0.expansion_convolution)
    p1.compute_e_f(p1.expansion_feature_map,p1.expansion_convolution)
    
    p0.compute_z()
    p1.compute_z()
    p0.sum_convolution_3d()
    p1.sum_convolution_3d()

def relu(p0,p1,matrix_x0,matrix_x1):
    shape = matrix_x0.shape
    matrix_x0 = matrix_x0.reshape(-1,1)
    matrix_x1 = matrix_x1.reshape(-1,1)
    p0.compute_e_f(matrix_x0,p0.compare_triple[0:matrix_x0.shape[0],0].reshape(-1,1))
    p1.compute_e_f(matrix_x1,p1.compare_triple[0:matrix_x1.shape[0],0].reshape(-1,1))
    p0.compute_z()#计算z
    p1.compute_z()
    p0.output_z()#输出中间乘法的z
    p1.output_z()
    p0.add_z()#在本地还原真正的z
    p1.add_z()
    p0.init_s_mul()#初始化乘法共享的符号
    p1.init_s_mul()
    #将乘法共享转换为加法共享
    p0.init_rand()#生成随机数，作为乘法共享的值的加法共享值
    p1.init_rand()
    p0.get_rand()#读取对方发来的随机数，作为乘法共享的值的加法共享值
    p1.get_rand()
    #计算z1+z2=<[s]0>*<[s]1>
    p0.compute_e_f(p0.s_mul_0_add_0,p0.s_mul_1_add_0)
    p1.compute_e_f(p1.s_mul_0_add_1,p1.s_mul_1_add_1)
    p0.compute_z()#计算z
    p1.compute_z()

    p0.set_s(shape)#还原s,p0p1获得了自己的加法共享的s
    p1.set_s(shape)
    p0.set_c()#初始化c，当s=-1时，c=0，s=1时，c=1 
    p1.set_c()
    
    p0.compute_e_f(matrix_x0,p0.c)
    p1.compute_e_f(matrix_x1,p1.c)
    p0.compute_z()#计算z
    p1.compute_z()

    p0.construct_feature_map(shape)
    p1.construct_feature_map(shape)

def pooling(p0,p1,matrix_x0,matrix_x1):
    shape = matrix_x0.shape
    p0.init_pooling_matrix(matrix_x0)
    p1.init_pooling_matrix(matrix_x1)
    compare(p0,p1,p0.pooling_00_01,p1.pooling_00_01)
    
    p0.pooling_00_01_s = p0.z.reshape(-1,1)
    p1.pooling_00_01_s = p1.z.reshape(-1,1)
    p0.pooling_01_00_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_01_00_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_00_01_s_flag = p0.flag 
    p1.pooling_00_01_s_flag = p1.flag
    p0.pooling_01_00_s_flag = p0.flag
    p1.pooling_01_00_s_flag = p1.flag
    compare(p0,p1,p0.pooling_00_10,p1.pooling_00_10)
    
    p0.pooling_00_10_s = p0.z.reshape(-1,1)
    p1.pooling_00_10_s = p1.z.reshape(-1,1)
    p0.pooling_10_00_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_10_00_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_00_10_s_flag = p0.flag
    p1.pooling_00_10_s_flag = p1.flag
    p0.pooling_10_00_s_flag = p0.flag
    p1.pooling_10_00_s_flag = p1.flag

    compare(p0,p1,p0.pooling_00_11,p1.pooling_00_11)
    p0.pooling_00_11_s = p0.z.reshape(-1,1)
    p1.pooling_00_11_s = p1.z.reshape(-1,1)
    p0.pooling_11_00_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_11_00_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_00_11_s_flag = p0.flag
    p1.pooling_00_11_s_flag = p1.flag
    p0.pooling_11_00_s_flag = p0.flag
    p1.pooling_11_00_s_flag = p1.flag


    compare(p0,p1,p0.pooling_01_10,p1.pooling_01_10)
    p0.pooling_01_10_s = p0.z.reshape(-1,1)
    p1.pooling_01_10_s = p1.z.reshape(-1,1)
    p0.pooling_10_01_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_10_01_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_01_10_s_flag = p0.flag
    p1.pooling_01_10_s_flag = p1.flag
    p0.pooling_10_01_s_flag = p0.flag
    p1.pooling_10_01_s_flag = p1.flag

    compare(p0,p1,p0.pooling_01_11,p1.pooling_01_11)
    p0.pooling_01_11_s = p0.z.reshape(-1,1)
    p1.pooling_01_11_s = p1.z.reshape(-1,1)
    p0.pooling_11_01_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_11_01_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_01_11_s_flag = p0.flag
    p1.pooling_01_11_s_flag = p1.flag
    p0.pooling_11_01_s_flag = p0.flag
    p1.pooling_11_01_s_flag = p1.flag

    compare(p0,p1,p0.pooling_10_11,p1.pooling_10_11)
    p0.pooling_10_11_s = p0.z.reshape(-1,1)
    p1.pooling_10_11_s = p1.z.reshape(-1,1)
    p0.pooling_11_10_s = (p0.z*(-1)).reshape(-1,1)
    p1.pooling_11_10_s = (p1.z*(-1)).reshape(-1,1)
    p0.pooling_10_11_s_flag = p0.flag
    p1.pooling_10_11_s_flag = p1.flag
    p0.pooling_11_10_s_flag = p0.flag
    p1.pooling_11_10_s_flag = p1.flag

    p0.flag_00 = p0.pooling_00_01_s_flag + p0.pooling_00_10_s_flag + p0.pooling_00_11_s_flag
    p0.flag_01 = p0.pooling_01_00_s_flag + p0.pooling_01_10_s_flag + p0.pooling_01_11_s_flag
    p0.flag_10 = p0.pooling_10_00_s_flag + p0.pooling_10_01_s_flag + p0.pooling_10_11_s_flag
    p0.flag_11 = p0.pooling_11_00_s_flag + p0.pooling_11_01_s_flag + p0.pooling_11_10_s_flag
    # print(p0.flag_00[14+5-1,0])
    # print(p0.flag_01[14+5-1,0])
    # print(p0.flag_10[14+5-1,0])
    # print(p0.flag_11[14+5-1,0])
    p0.flag_00 = p0.flag_00*p0.flag_00*p0.flag_00/3+p0.flag_00*p0.flag_00*(-2)+p0.flag_00*5/3+8
    p0.flag_01 = p0.flag_01*p0.flag_01*p0.flag_01/3+p0.flag_01*p0.flag_01*(-2)+p0.flag_01*5/3+8
    p0.flag_10 = p0.flag_10*p0.flag_10*p0.flag_10/3+p0.flag_10*p0.flag_10*(-2)+p0.flag_10*5/3+8
    p0.flag_11 = p0.flag_11*p0.flag_11*p0.flag_11/3+p0.flag_11*p0.flag_11*(-2)+p0.flag_11*5/3+8
    
    p1.flag_00 = p1.pooling_00_01_s_flag + p1.pooling_00_10_s_flag + p1.pooling_00_11_s_flag
    p1.flag_01 = p1.pooling_01_00_s_flag + p1.pooling_01_10_s_flag + p1.pooling_01_11_s_flag
    p1.flag_10 = p1.pooling_10_00_s_flag + p1.pooling_10_01_s_flag + p1.pooling_10_11_s_flag
    p1.flag_11 = p1.pooling_11_00_s_flag + p1.pooling_11_01_s_flag + p1.pooling_11_10_s_flag
    
   
    p1.flag_00 = p1.flag_00*p1.flag_00*p1.flag_00/3+p1.flag_00*p1.flag_00*(-2)+p1.flag_00*5/3+8
    p1.flag_01 = p1.flag_01*p1.flag_01*p1.flag_01/3+p1.flag_01*p1.flag_01*(-2)+p1.flag_01*5/3+8
    p1.flag_10 = p1.flag_10*p1.flag_10*p1.flag_10/3+p1.flag_10*p1.flag_10*(-2)+p1.flag_10*5/3+8
    p1.flag_11 = p1.flag_11*p1.flag_11*p1.flag_11/3+p1.flag_11*p1.flag_11*(-2)+p1.flag_11*5/3+8
    # print(p1.flag_00[14+5-1,0])
    # print(p1.flag_01[14+5-1,0])
    # print(p1.flag_10[14+5-1,0])
    # print(p1.flag_11[14+5-1,0])
    #先计算00是不是最大的

    multiple(p0,p1,p0.pooling_00_01_s+1,p0.pooling_00_10_s+1,p1.pooling_00_01_s,p1.pooling_00_10_s)
    p0.temp_s = p0.z.reshape(-1,1)
    p1.temp_s = p1.z.reshape(-1,1)
    multiple(p0,p1,p0.temp_s,p0.pooling_00_11_s+1,p1.temp_s,p1.pooling_00_11_s)
    p0.temp_s = p0.z.reshape(-1,1)/p0.flag_00
    p1.temp_s = p1.z.reshape(-1,1)/p1.flag_00
    multiple(p0,p1,p0.temp_s,p0.pooling_00.reshape(-1,1),p1.temp_s,p1.pooling_00.reshape(-1,1))
    p0.temp = p0.z.reshape(-1,1)
    p1.temp = p1.z.reshape(-1,1)

    # print(p0.temp+p1.temp)
    #再计算01
    multiple(p0,p1,p0.pooling_01_00_s+1,p0.pooling_01_10_s+1,p1.pooling_01_00_s,p1.pooling_01_10_s)
    p0.temp_s = p0.z.reshape(-1,1)
    p1.temp_s = p1.z.reshape(-1,1)
    multiple(p0,p1,p0.temp_s,p0.pooling_01_11_s+1,p1.temp_s,p1.pooling_01_11_s)
    p0.temp_s = p0.z.reshape(-1,1)/p0.flag_01
    p1.temp_s = p1.z.reshape(-1,1)/p1.flag_01
    multiple(p0,p1,p0.temp_s,p0.pooling_01.reshape(-1,1),p1.temp_s,p1.pooling_01.reshape(-1,1))
    p0.temp += p0.z.reshape(-1,1)
    p1.temp += p1.z.reshape(-1,1)
    # print(p0.temp+p1.temp)

    #再计算10
    multiple(p0,p1,p0.pooling_10_00_s+1,p0.pooling_10_01_s+1,p1.pooling_10_00_s,p1.pooling_10_01_s)
    p0.temp_s = p0.z.reshape(-1,1)
    p1.temp_s = p1.z.reshape(-1,1)
    multiple(p0,p1,p0.temp_s,p0.pooling_10_11_s+1,p1.temp_s,p1.pooling_10_11_s)
    p0.temp_s = p0.z.reshape(-1,1)/p0.flag_10
    p1.temp_s = p1.z.reshape(-1,1)/p1.flag_10
    multiple(p0,p1,p0.temp_s,p0.pooling_10.reshape(-1,1),p1.temp_s,p1.pooling_10.reshape(-1,1))
    p0.temp += p0.z.reshape(-1,1)
    p1.temp += p1.z.reshape(-1,1)
    # print(p0.temp+p1.temp)

    #再计算11
    multiple(p0,p1,p0.pooling_11_00_s+1,p0.pooling_11_01_s+1,p1.pooling_11_00_s,p1.pooling_11_01_s)
    p0.temp_s = p0.z.reshape(-1,1)
    p1.temp_s = p1.z.reshape(-1,1)
    multiple(p0,p1,p0.temp_s,p0.pooling_11_10_s+1,p1.temp_s,p1.pooling_11_10_s)
    p0.temp_s = p0.z.reshape(-1,1)/p0.flag_11
    p1.temp_s = p1.z.reshape(-1,1)/p1.flag_11
    multiple(p0,p1,p0.temp_s,p0.pooling_11.reshape(-1,1),p1.temp_s,p1.pooling_11.reshape(-1,1))
    p0.temp += p0.z.reshape(-1,1)
    p1.temp += p1.z.reshape(-1,1)
    
    p0.feature_map = p0.temp.reshape(shape[0],int(shape[1]/2),int(shape[2]/2))
    p1.feature_map = p1.temp.reshape(shape[0],int(shape[1]/2),int(shape[2]/2))
    
def compare(p0,p1,matrix_x0,matrix_x1):
    #shape = matrix_x0.shape
    matrix_x0 = matrix_x0.reshape(-1,1)
    matrix_x1 = matrix_x1.reshape(-1,1)
    p0.compute_e_f(matrix_x0,p0.compare_triple[0:matrix_x0.shape[0],0].reshape(-1,1))
    p1.compute_e_f(matrix_x1,p1.compare_triple[0:matrix_x1.shape[0],0].reshape(-1,1))
    p0.compute_z()#计算z
    p1.compute_z()
    p0.output_z()#输出中间乘法的z
    p1.output_z()
    p0.add_z()#在本地还原真正的z
    p1.add_z()
    p0.init_s_mul()#初始化乘法共享的符号
    p1.init_s_mul()
    #将乘法共享转换为加法共享
    p0.init_rand()#生成随机数，作为乘法共享的值的加法共享值
    p1.init_rand()
    p0.get_rand()#读取对方发来的随机数，作为乘法共享的值的加法共享值
    p1.get_rand()
    #计算z1+z2=<[s]0>*<[s]1>
    p0.compute_e_f(p0.s_mul_0_add_0,p0.s_mul_1_add_0)
    p1.compute_e_f(p1.s_mul_0_add_1,p1.s_mul_1_add_1)
    p0.compute_z()#计算z
    p1.compute_z()

def multiple(p0,p1,matrix_x0,matrix_y0,matrix_x1,matrix_y1):
    p0.compute_e_f(matrix_x0,matrix_y0)
    p1.compute_e_f(matrix_x1,matrix_y1)
    p0.compute_z()
    p1.compute_z()

def fully_connection(p0,p1):
    p0.expansion_fully_connection()
    p1.expansion_fully_connection()

    p0.compute_e_f(p0.expansion_fully_connection_feature_map,p0.expansion_fully_connection_fc_weight)
    p1.compute_e_f(p1.expansion_fully_connection_feature_map,p1.expansion_fully_connection_fc_weight)
    p0.compute_z()
    p1.compute_z()

    p0.sum_convolution_fully_connection()
    p1.sum_convolution_fully_connection()

p0 = P0.P0() 
p1 = P1.P1()

start = time.perf_counter()

convolution(p0,p1,p0.convolution_weight,p1.convolution_weight)

# print(p0.feature_map+p1.feature_map)
# print((p0.feature_map+p1.feature_map)[0,:,:])

# print((p0.feature_map+p1.feature_map).shape)

pooling(p0,p1,p0.feature_map,p1.feature_map)

# print((p0.feature_map+p1.feature_map)[31,:,:])

# print((p0.feature_map+p1.feature_map).shape)



relu(p0,p1,p0.feature_map,p1.feature_map)

# print((p0.feature_map+p1.feature_map)[31,:,:])
print(p0.feature_map.shape)

# print((p0.feature_map+p1.feature_map).shape)
# print((p0.feature_map+p1.feature_map).dtype)
convolution_3d(p0,p1,p0.convolution_weight_3,p1.convolution_weight_3)
# print((p0.expansion_feature_map+p1.expansion_feature_map).shape)
# print((p0.expansion_convolution+p1.expansion_convolution).shape)

pooling(p0,p1,p0.feature_map,p1.feature_map)
relu(p0,p1,p0.feature_map,p1.feature_map)
print((p0.feature_map+p1.feature_map).shape)

# print((p0.feature_map+p1.feature_map)[63,:,:])

# print(np.matmul((p0.fc_weight+p1.fc_weight),(p0.feature_map+p1.feature_map).reshape(-1,1)))

fully_connection(p0,p1)
print((p0.feature_map+p1.feature_map))

elapsed = (time.perf_counter() - start)
print("Time used:",elapsed)
print("Number of compare_triple:",p0.new_compare_number)
print("Number of multiplication_number:",p0.new_multiplication_number)