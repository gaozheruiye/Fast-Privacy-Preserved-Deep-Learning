import numpy as np
# 导入xlrd库
import xlrd
import xlwt


class Company:
    def __init__(self):
        # convolution_bias_ = xlrd.open_workbook(r'.\convolution_bias.xls')
        # table = convolution_bias_.sheet_by_name('Sheet1')
        # row = table.nrows  # 行数
        # col = table.ncols  # 列数
        # self.convolution_bias_row = row
        # self.convolution_bias_col = col
        # self.convolution_bias = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
        # for x in range(col):
        #     cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        #     self.convolution_bias[:, x] = cols  # 按列把数据存进矩阵中
        # convolution0_weight_ = xlrd.open_workbook(r'.\convolution_weight.xls')
        # table = convolution0_weight_.sheet_by_name('Sheet1')
        # row = table.nrows  # 行数
        # col = table.ncols  # 列数
        # self.convolution0_weight_row = row
        # self.convolution0_weight_col = col
        # self.convolution0_weight = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
        # for x in range(col):
        #     cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        #     self.convolution0_weight[:, x] = cols  # 按列把数据存进矩阵中

        # 加载第二层的bias和weight
        convolution_bias__3 = xlrd.open_workbook(r'.\convolution_bias_3.xls')
        table = convolution_bias__3.sheet_by_name('Sheet1')
        row = table.nrows  # 行数
        col = table.ncols  # 列数
        self.convolution_bias_row_3 = row
        self.convolution_bias_col_3 = col
        self.convolution_bias_3 = np.zeros(
            (row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
        for x in range(col):
            cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
            self.convolution_bias_3[:, x] = cols  # 按列把数据存进矩阵中

        self.convolution_weight_3 = np.load("./convolution_weight_3.npy")

    def ran_(self):
        # convolution_bias0 = np.random.randint(-100,100,(self.convolution_bias_row,self.convolution_bias_col))
        # convolution_bias1 = self.convolution_bias-convolution_bias0
        # print(convolution_bias0,convolution_bias1)
        # np.save("./convolution0_bias.npy",convolution_bias0)
        # np.save("./convolution1_bias.npy",convolution_bias1)

        # convolution0_weight0 = np.random.randint(-100,100,(self.convolution0_weight_row,self.convolution0_weight_col))
        # convolution0_weight1 = self.convolution0_weight-convolution0_weight0
        # print(convolution0_weight0+convolution0_weight1)
        # np.save("./convolution0_weight.npy",convolution0_weight0)
        # np.save("./convolution1_weight.npy",convolution0_weight1)

        convolution_bias_3_0 = np.random.randint(
            -100, 100, self.convolution_bias_3.shape)
        convolution_bias_3_1 = self.convolution_bias_3-convolution_bias_3_0
        print(convolution_bias_3_0.shape)
        np.save("./convolution0_bias_3.npy", convolution_bias_3_0)
        np.save("./convolution1_bias_3.npy", convolution_bias_3_1)

        convolution_weight_3_0 = np.random.randint(
            -100, 100, self.convolution_weight_3.shape)
        convolution_weight_3_1 = self.convolution_weight_3-convolution_weight_3_0
        print(convolution_weight_3_0.shape)
        np.save("./convolution0_weight_3.npy", convolution_weight_3_0)
        np.save("./convolution1_weight_3.npy", convolution_weight_3_1)


company = Company()
company.ran_()
