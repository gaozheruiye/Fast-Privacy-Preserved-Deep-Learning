import numpy as np
import xlwt
class TTP:  # 创建类
    def __init__(self,n,CTmin,CTmax,MTmin,MTmax): # 初始化
        self.n = n
        self.CTmin = CTmin
        self.CTmax = CTmax
        self.MTmin = MTmin
        self.MTmax = MTmax

    def randomCT(self): 
        Random_Comparison_matrix_column_p = np.random.randint(self.CTmin,self.CTmax,(self.n,1))
        Random_Comparison_matrix_column_q = np.random.randint(self.CTmin,self.CTmax,(self.n,1))

        Random_Comparison_matrix_column_u = Random_Comparison_matrix_column_p*Random_Comparison_matrix_column_q 
        Random_Comparison_matrix_column_u0 = np.random.randint(-100,100,(self.n,1))
        Random_Comparison_matrix_column_u1 = Random_Comparison_matrix_column_u-Random_Comparison_matrix_column_u0

        Random_Comparison_matrix_TTP = np.hstack((Random_Comparison_matrix_column_p,Random_Comparison_matrix_column_q,Random_Comparison_matrix_column_u))
        Random_Comparison_matrix_0 = np.hstack((Random_Comparison_matrix_column_u0,Random_Comparison_matrix_column_p))
        Random_Comparison_matrix_1 = np.hstack((Random_Comparison_matrix_column_u1,Random_Comparison_matrix_column_q))
        
        print(Random_Comparison_matrix_TTP)
        # Random_Comparison_matrix_0=Random_Comparison_matrix_0.astype(np.float16)
        # Random_Comparison_matrix_1=Random_Comparison_matrix_1.astype(np.float16)
        print(Random_Comparison_matrix_0)
        print(Random_Comparison_matrix_1)
        np.save("./compare_triple0.npy",Random_Comparison_matrix_0)
        np.save("./compare_triple1.npy",Random_Comparison_matrix_1)

        

    def randomMT(self):
        Random_Multi_Triple_matrix_column_a = np.random.randint(self.MTmin,self.MTmax,(self.n,1))
        Random_Multi_Triple_matrix_column_b = np.random.randint(self.MTmin,self.MTmax,(self.n,1))
        Random_Multi_Triple_matrix_column_c = Random_Multi_Triple_matrix_column_a*Random_Multi_Triple_matrix_column_b

        Random_Multi_Triple_matrix_column_a0 = np.random.randint(self.MTmin,self.MTmax,(self.n,1))
        Random_Multi_Triple_matrix_column_b0 = np.random.randint(self.MTmin,self.MTmax,(self.n,1))
        Random_Multi_Triple_matrix_column_c0 = np.random.randint(-100,100,(self.n,1))

        Random_Multi_Triple_matrix_column_a1 = Random_Multi_Triple_matrix_column_a-Random_Multi_Triple_matrix_column_a0
        Random_Multi_Triple_matrix_column_b1 = Random_Multi_Triple_matrix_column_b-Random_Multi_Triple_matrix_column_b0
        Random_Multi_Triple_matrix_column_c1 = Random_Multi_Triple_matrix_column_c-Random_Multi_Triple_matrix_column_c0

        Random_Multi_Triple_matrix_TTP = np.hstack((Random_Multi_Triple_matrix_column_a,Random_Multi_Triple_matrix_column_b,Random_Multi_Triple_matrix_column_c))
        Random_Multi_Triple_matrix_0 = np.hstack((Random_Multi_Triple_matrix_column_a0,Random_Multi_Triple_matrix_column_b0,Random_Multi_Triple_matrix_column_c0))
        Random_Multi_Triple_matrix_1 = np.hstack((Random_Multi_Triple_matrix_column_a1,Random_Multi_Triple_matrix_column_b1,Random_Multi_Triple_matrix_column_c1))


        print(Random_Multi_Triple_matrix_TTP)
        print(Random_Multi_Triple_matrix_0)
        print(Random_Multi_Triple_matrix_1)
        np.save("./multiplication_triple0.npy",Random_Multi_Triple_matrix_0)
        np.save("./multiplication_triple1.npy",Random_Multi_Triple_matrix_1)
        

ttp = TTP(20000000,1,20,1,20)  # 创建实例时直接给定实例属性，self不算在内

ttp.randomCT()
ttp.randomMT()