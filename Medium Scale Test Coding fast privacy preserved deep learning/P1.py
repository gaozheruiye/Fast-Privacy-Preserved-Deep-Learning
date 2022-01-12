import numpy as np
# 导入xlrd库
import xlrd
import xlwt


class P1:  # 创建Circle类
    def __init__(self):  # 初始化

        # 导入CT图像
        self.CT1 = np.load("./CT1.npy")
        # 导入卷积核权重矩阵
        self.convolution_weight = np.load("./convolution1_weight.npy")
        # 导入卷积核偏置
        self.convolution_bias = np.load("./convolution1_bias.npy")
        # 这里非常粗暴，直接将self.convolution_col置为5
        self.convolution_row = 5
        self.convolution_col = 5
        self.convolution_depth = 32
        self.convolution_number = self.convolution_row*self.convolution_col
        ######################################################################
        # 导入第三层卷积核权重矩阵
        self.convolution_weight_3 = np.load("./convolution1_weight_3.npy")
        # 导入第三层卷积核偏置
        self.convolution_bias_3 = np.load("./convolution1_bias_3.npy")

        # 导入全连接层权重矩阵(10,3136)
        self.fc_weight = np.load("./fc_weight_1.npy")
        # 导入全连接层bias(10,)
        self.fc_bias = np.load("./fc_bias_1.npy")

        # 导入比较三元组
        self.compare_triple = np.load("./compare_triple1.npy")
        # 导入乘法三元组
        self.multiplication_triple = np.load("./multiplication_triple1.npy")
        # 初始化特征图的长和宽

        self.feature_map = self.CT1
        self.feature_map_row = self.feature_map.shape[0]
        self.feature_map_col = self.feature_map.shape[1]
        self.compare_number = 0
        self.multiplication_number = 0  # 记录使用到第几个乘法三元组

    def padding(self):
        self.feature_map_row = self.feature_map_row + 4
        self.feature_map_col = self.feature_map_col + 4
        temp = np.zeros((self.feature_map_row, self.feature_map_col))
        temp[2:-2, 2:-2] = self.feature_map
        self.feature_map = temp

    def padding_3d(self):
        self.feature_map_row = self.feature_map_row + 4
        self.feature_map_col = self.feature_map_col + 4
        temp = np.zeros(
            (self.feature_map.shape[0], self.feature_map_row, self.feature_map_col))
        temp[:, 2:-2, 2:-2] = self.feature_map
        self.feature_map = temp

    def expand(self, matrix, convolution_matrix):

        # 初始化卷积展开
        self.expansion_feature_map = np.zeros(((self.feature_map_row-self.convolution_col+1)*(
            self.feature_map_col-self.convolution_col+1)*self.convolution_number, 1))
        self.expansion_convolution = np.zeros((self.convolution_number, 1))

        # 展开图像
        for x in range((self.feature_map_row-self.convolution_col+1)):
            for y in range((self.feature_map_col-self.convolution_col+1)):
                for i in range(self.convolution_col):
                    for j in range(self.convolution_col):
                        self.expansion_feature_map[x*(self.feature_map_col-self.convolution_col+1)*self.convolution_number +
                                                   y*self.convolution_number+i*self.convolution_col+j, 0] = matrix[x+i, y+j]

        temp = self.expansion_feature_map
        for i in range(self.convolution_depth-1):
            self.expansion_feature_map = np.vstack(
                [self.expansion_feature_map, temp])

        # 展开卷积核

        for i in range(self.convolution_row):
            for j in range(self.convolution_col):
                self.expansion_convolution[i*self.convolution_col +
                                           j] = convolution_matrix[i, j]

        temp = self.expansion_convolution
        for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
            self.expansion_convolution = np.vstack(
                [self.expansion_convolution, temp])

        for k in range(self.convolution_depth-1):
            temp_expansion_convolution = np.zeros((self.convolution_number, 1))
            for i in range(self.convolution_row):
                for j in range(self.convolution_col):
                    temp_expansion_convolution[i*self.convolution_col +
                                               j] = convolution_matrix[i+self.convolution_col*k+self.convolution_col, j]
            temp = temp_expansion_convolution
            for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
                temp = np.vstack(
                    [temp_expansion_convolution, temp])
            self.expansion_convolution = np.vstack(
                [self.expansion_convolution, temp])
        # print(self.expansion_convolution.shape)

    def convolution_3d_init(self, convolution_x):
        self.convolution_depth = convolution_x.shape[0]
        self.convolution_channel = convolution_x.shape[1]

    def expand_3d(self, matrix, convolution_matrix):
        # 初始化卷积展开
        self.expansion_feature_map = np.zeros(((self.feature_map_row-self.convolution_col+1)*(
            self.feature_map_col-self.convolution_col+1)*self.convolution_number, 1))
        self.expansion_convolution = np.zeros((self.convolution_number, 1))
        # print(self.expansion_feature_map.shape)#(4900, 1)
        # 展开特征图

        for x in range((self.feature_map_row-self.convolution_col+1)):
            for y in range((self.feature_map_col-self.convolution_col+1)):
                for i in range(self.convolution_col):
                    for j in range(self.convolution_col):
                        self.expansion_feature_map[x*(self.feature_map_col-self.convolution_col+1)*self.convolution_number +
                                                   y*self.convolution_number+i*self.convolution_col+j, 0] = matrix[0, x+i, y+j]
        # print(self.expansion_feature_map.shape)#(4900, 1)
        for r in range(self.feature_map_channel-1):
            temp_expansion_feature_map = np.zeros(((self.feature_map_row-self.convolution_col+1)*(
                self.feature_map_col-self.convolution_col+1)*self.convolution_number, 1))
            for x in range((self.feature_map_row-self.convolution_col+1)):
                for y in range((self.feature_map_col-self.convolution_col+1)):
                    for i in range(self.convolution_col):
                        for j in range(self.convolution_col):
                            temp_expansion_feature_map[x*(self.feature_map_col-self.convolution_col+1)*self.convolution_number +
                                                       y*self.convolution_number+i*self.convolution_col+j, 0] = matrix[r+1, x+i, y+j]
            # print(temp_expansion_feature_map.shape)
            self.expansion_feature_map = np.vstack(
                [self.expansion_feature_map, temp_expansion_feature_map])
        temp = self.expansion_feature_map
        for i in range(self.convolution_depth-1):
            self.expansion_feature_map = np.vstack(
                [self.expansion_feature_map, temp])
        # print(self.expansion_feature_map.shape)#(10035200, 1)

        # 展开卷积核
        for i in range(self.convolution_row):
            for j in range(self.convolution_col):
                self.expansion_convolution[i*self.convolution_col +
                                           j] = convolution_matrix[0, 0, i, j]
        temp = self.expansion_convolution
        for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
            self.expansion_convolution = np.vstack(
                [self.expansion_convolution, temp])
        # print(self.expansion_convolution.shape)  # 4900
        for k in range(self.convolution_channel-1):
            temp_expansion_convolution = np.zeros((self.convolution_number, 1))
            for i in range(self.convolution_row):
                for j in range(self.convolution_col):
                    temp_expansion_convolution[i*self.convolution_col +
                                               j] = convolution_matrix[0, k+1, i, j]
            temp = temp_expansion_convolution
            for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
                temp = np.vstack(
                    [temp_expansion_convolution, temp])
            self.expansion_convolution = np.vstack(
                [self.expansion_convolution, temp])
        # print(self.expansion_convolution.shape)  #(156800, 1)

        for r in range(self.convolution_depth-1):
            new_temp_expansion_convolution = np.zeros(
                (self.convolution_number, 1))
            for i in range(self.convolution_row):
                for j in range(self.convolution_col):
                    new_temp_expansion_convolution[i*self.convolution_col +
                                                   j] = convolution_matrix[r+1, 0, i, j]
            temp = new_temp_expansion_convolution
            for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
                new_temp_expansion_convolution = np.vstack(
                    [new_temp_expansion_convolution, temp])

            # print(new_temp_expansion_convolution.shape)  # 4900
            for k in range(self.convolution_channel-1):
                temp_expansion_convolution = np.zeros(
                    (self.convolution_number, 1))
                for i in range(self.convolution_row):
                    for j in range(self.convolution_col):
                        temp_expansion_convolution[i*self.convolution_col +
                                                   j] = convolution_matrix[r+1, k+1, i, j]
                temp = temp_expansion_convolution
                for i in range((self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1)-1):
                    temp = np.vstack(
                        [temp_expansion_convolution, temp])
                new_temp_expansion_convolution = np.vstack(
                    [new_temp_expansion_convolution, temp])
            # print(new_temp_expansion_convolution.shape, temp_expansion_convolution.shape)
            self.expansion_convolution = np.vstack(
                [self.expansion_convolution, new_temp_expansion_convolution])
        # print(self.expansion_convolution.shape)  # (156800, 1)

    def compute_e_f(self, matrix_x, matrix_y):  # 计算e=x-a,f=y-b要求：
        matrix_e = np.zeros(matrix_x.shape)
        matrix_f = np.zeros(matrix_y.shape)
        for i in range(matrix_x.shape[0]):
            matrix_e[i, 0] = matrix_x[i, 0] - \
                self.multiplication_triple[self.multiplication_number+i, 0]

        for i in range(matrix_x.shape[0]):
            matrix_f[i, 0] = matrix_y[i, 0] - \
                self.multiplication_triple[self.multiplication_number+i, 1]
        matrix_e_f = np.hstack((matrix_e, matrix_f))
        # 保存matrix_e_f0
        np.save("./matrix_e_f1.npy", matrix_e_f)
        self.matrix_e_f1 = matrix_e_f

    def compute_z(self):  # 乘法中计算z
        matrix_e_f0 = np.load("./matrix_e_f0.npy")
        matrix_e_f = np.zeros(matrix_e_f0.shape)
        matrix_e_f = self.matrix_e_f1 + matrix_e_f0
        self.z = matrix_e_f[:, 1]*matrix_e_f[:, 0] + matrix_e_f[:, 1]*self.multiplication_triple[self.multiplication_number:self.multiplication_number+matrix_e_f0.shape[0], 0] + \
            matrix_e_f[:, 0]*self.multiplication_triple[self.multiplication_number:self.multiplication_number+matrix_e_f0.shape[0], 1] + \
            self.multiplication_triple[self.multiplication_number:
                                       self.multiplication_number+matrix_e_f0.shape[0], 2]

        # self.multiplication_number += matrix_e_f0.shape[0]
        return self.z

    def output_z(self):
        np.save("./z1.npy", self.z)

    def add_z(self):
        z0 = np.load("./z0.npy")
        self.z = self.z.reshape(-1, 1)
        z0 = z0.reshape(-1, 1)
        self.z = self.z+z0

    def init_s_mul(self):
        self.s_mul = np.zeros(self.z.shape)
        self.flag = np.zeros(self.z.shape)
        # mask=self.z<0.001
        # self.flag[mask]=1
        # self.s_mul[mask]=0
        # self.s_mul[((self.compare_triple[:, 1]*self.z[:, 0])>0)&(~mask),0]=1
        # self.s_mul[((self.compare_triple[:, 1]*self.z[:, 0])<0)&(~mask),0]=-1

        self.s_mul = np.zeros(self.z.shape)
        self.flag = np.zeros(self.z.shape)
        for i in range(self.s_mul.shape[0]):
            if abs(self.z[i, 0]) < 0.00000001:
                self.flag[i, 0] = 1
                self.s_mul[i, 0] = 0
            elif self.compare_triple[i, 1]*self.z[i, 0] > 0:
                self.s_mul[i, 0] = 1
            elif self.compare_triple[i, 1]*self.z[i, 0] < 0:
                self.s_mul[i, 0] = -1

    def init_rand(self):
        rand = np.random.randint(-20, 20, (self.s_mul.shape[0], 1))
        self.s_mul_1_add_1 = self.s_mul-rand
        np.save("./s_mul_1_rand.npy", rand)

    def get_rand(self):
        self.s_mul_0_add_1 = np.load("./s_mul_0_rand.npy")
        self.s_mul_0_add_1 = self.s_mul_0_add_1.reshape(-1, 1)

    def set_s(self, shape):
        self.s = self.z.reshape(shape)
        self.flag = self.flag.reshape(shape)

    def set_c(self):
        self.c = np.zeros(self.s.shape)
        for k in range(self.s.shape[0]):
            for i in range(self.s.shape[1]):
                for j in range(self.s.shape[2]):
                    self.c[k, i, j] = self.s[k, i, j]/2+0.25
                    if self.flag[k, i, j] == 1:
                        self.c[k, i, j] = 0
        self.c = self.c.reshape(-1, 1)

    def construct_feature_map(self, shape):
        self.feature_map = self.z.reshape(shape)
        self.feature_map_channel = self.feature_map.shape[0]
        self.feature_map_row = self.feature_map.shape[1]
        self.feature_map_col = self.feature_map.shape[2]

    def sum_convolution(self):  # 计算卷积层后的特征图
        self.feature_map = np.zeros(
            (self.convolution_depth, self.feature_map_col-self.convolution_col+1, self.feature_map_row-self.convolution_col+1))
        for k in range(self.convolution_depth):
            for i in range(self.feature_map_row-self.convolution_col+1):
                for j in range(self.feature_map_col-self.convolution_col+1):
                    temp = 0
                    for x in range(self.convolution_col):
                        for y in range(self.convolution_col):
                            temp += self.z[k * (self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1) * self.convolution_number + (
                                i*(self.feature_map_row-self.convolution_col+1)+j) * self.convolution_number+x*self.convolution_col+y]
                    self.feature_map[k, i, j] = temp + \
                        self.convolution_bias[k, 0]
        self.feature_map_channel = self.feature_map.shape[0]
        self.feature_map_row = self.feature_map.shape[1]
        self.feature_map_col = self.feature_map.shape[2]

    def sum_convolution_3d(self):  # 计算卷积层后的特征图
        self.feature_map = np.zeros(
            (self.convolution_depth, self.feature_map_col-self.convolution_col+1, self.feature_map_row-self.convolution_col+1))
        for k in range(self.convolution_depth):
            for i in range(self.feature_map_row-self.convolution_col+1):
                for j in range(self.feature_map_col-self.convolution_col+1):
                    temp = 0
                    for r in range(self.feature_map_channel):
                        for x in range(self.convolution_col):
                            for y in range(self.convolution_col):
                                temp += self.z[k*(self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1) * self.convolution_number*self.feature_map_channel+r * (self.feature_map_row-self.convolution_col+1)*(self.feature_map_col-self.convolution_col+1) * self.convolution_number + (
                                    i*(self.feature_map_row-self.convolution_col+1)+j) * self.convolution_number+x*self.convolution_col+y]
                    self.feature_map[k, i, j] = temp + self.convolution_bias_3[k, 0]
        self.feature_map_channel = self.feature_map.shape[0]
        self.feature_map_row = self.feature_map.shape[1]
        self.feature_map_col = self.feature_map.shape[2]

    def init_pooling_matrix(self, matrix_x0):
        depth = matrix_x0.shape[0]
        row = int(matrix_x0.shape[1]/2)
        col = int(matrix_x0.shape[2]/2)

        self.pooling_00 = np.zeros((depth, row, col))
        for k in range(depth):
            for i in range(row):
                for j in range(col):
                    self.pooling_00[k, i, j] = self.feature_map[k, 2*i, 2*j]
        self.pooling_01 = np.zeros((depth, row, col))
        for k in range(depth):
            for i in range(row):
                for j in range(col):
                    self.pooling_01[k, i, j] = self.feature_map[k, 2*i, 2*j+1]
        self.pooling_10 = np.zeros((depth, row, col))
        for k in range(depth):
            for i in range(row):
                for j in range(col):
                    self.pooling_10[k, i, j] = self.feature_map[k, 2*i+1, 2*j]
        self.pooling_11 = np.zeros((depth, row, col))
        for k in range(depth):
            for i in range(row):
                for j in range(col):
                    self.pooling_11[k, i,
                                    j] = self.feature_map[k, 2*i+1, 2*j+1]

        self.pooling_00_01 = self.pooling_00-self.pooling_01
        self.pooling_00_10 = self.pooling_00-self.pooling_10
        self.pooling_00_11 = self.pooling_00-self.pooling_11
        self.pooling_01_10 = self.pooling_01-self.pooling_10
        self.pooling_01_11 = self.pooling_01-self.pooling_11
        self.pooling_10_11 = self.pooling_10-self.pooling_11

        self.pooling_00_01_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_00_10_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_00_11_s = np.zeros(self.pooling_00_01.shape)

        self.pooling_01_00_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_01_10_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_01_11_s = np.zeros(self.pooling_00_01.shape)

        self.pooling_10_00_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_10_01_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_10_11_s = np.zeros(self.pooling_00_01.shape)

        self.pooling_11_00_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_11_01_s = np.zeros(self.pooling_00_01.shape)
        self.pooling_11_10_s = np.zeros(self.pooling_00_01.shape)

        self.temp_s = 0
        self.temp = 0
        self.flag_00 = np.zeros(self.pooling_00_01.shape)
        self.flag_01 = np.zeros(self.pooling_00_01.shape)
        self.flag_10 = np.zeros(self.pooling_00_01.shape)
        self.flag_11 = np.zeros(self.pooling_00_01.shape)

    def expansion_fully_connection(self):
        self.expansion_fully_connection_feature_map = self.feature_map.reshape((-1,1))
        temp = self.expansion_fully_connection_feature_map
        for i in range(self.fc_weight.shape[0]-1):
            self.expansion_fully_connection_feature_map = np.vstack(
                [self.expansion_fully_connection_feature_map, temp])
        self.expansion_fully_connection_fc_weight = self.fc_weight.reshape((-1,1))

    def sum_convolution_fully_connection(self):
        self.feature_map = np.zeros((self.fc_weight.shape[0],1))
        for i in range(self.fc_weight.shape[0]):
            for j in range(self.fc_weight.shape[1]):
                self.feature_map[i,0] += self.z[i*self.fc_weight.shape[1]+j]
            self.feature_map[i,0] += self.fc_bias[i]