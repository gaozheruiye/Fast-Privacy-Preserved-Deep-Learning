import numpy as np
# 导入xlrd库
import xlrd
import xlwt

class Hospital:
    def __init__(self):
        CT_ = xlrd.open_workbook(r'.\CT.xls')
        table = CT_.sheet_by_name('Sheet1')
        row = table.nrows  # 行数
        col = table.ncols  # 列数
        self.CT_row = row
        self.CT_col = col
        self.CT = np.zeros((row, col))  # 生成一个nrows行ncols列，且元素均为0的初始矩阵
        for x in range(col):
            cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
            self.CT[:, x] = cols  # 按列把数据存进矩阵中
    
    def ran_(self):
        CT0 = np.random.randint(-100,100,(self.CT_row,self.CT_col))
        CT1 = self.CT-CT0
        print(CT0+CT1)
        np.save("./CT0.npy",CT0)
        np.save("./CT1.npy",CT1)
        # # 创建工作簿
        # workBook0 = xlwt.Workbook("UTF-8")
        # # 创建工作表
        # oneWorkSheet0 = workBook0.add_sheet("Sheet1")

        # # 写入数据(行, 列, 数据)
        # for x in range(self.CT_row):
        #     for j in range(self.CT_col):
        #         oneWorkSheet0.write(x, j, str(CT0[x, j]))
        # # 保存数据,入参为文件路径
        # workBook0.save("./CT0.xls")

        # # 创建工作簿
        # workBook0 = xlwt.Workbook("UTF-8")
        # # 创建工作表
        # oneWorkSheet0 = workBook0.add_sheet("Sheet1")

        # # 写入数据(行, 列, 数据)
        # for x in range(self.CT_row):
        #     for j in range(self.CT_col):
        #         oneWorkSheet0.write(x, j, str(CT1[x, j]))
        # # 保存数据,入参为文件路径
        # workBook0.save("./CT1.xls")

hospital = Hospital()
hospital.ran_()