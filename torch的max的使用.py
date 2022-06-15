import torch
# 二维
# a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
# print(a)
# #1.不指定行列
#     #这个返回值不是元组
# max= torch.max(a)
# print(type(max))  #<class 'torch.Tensor'>
# print(max)       #tensor(65)
# # print(max[0])  #报错，因为返回值不是元组，不能使用切片

# #2.指定列
# max= torch.max(a,0)
# print(type(max))  #<class 'torch.return_types.max'>
# print(max)        #torch.return_types.max(values=tensor([ 2, 65, 62, 54]),indices=tensor([1, 2, 0, 0]))
# print(max[0])     #tensor([ 2, 65, 62, 54])  第一个返回值是值
# print(max[1])     #tensor([1, 2, 0, 0])      第二个返回值是index

#-----------------------------------三维------------------------

a = torch.tensor([[[1,5,62,54], [2,6,2,6], [2,65,2,6]],
                    [[2,5,63,58], [8,9,10,6], [89,56,22,6]]])
                    # tensor([[[ 1,  5, 62, 54],
                    #          [ 2,  6,  2,  6],
                    #          [ 2, 65,  2,  6]],

                    #         [[ 2,  5, 63, 58],
                    #          [ 8,  9, 10,  6],
                    #          [89, 56, 22,  6]]])
print(a)
max= torch.max(a,0)
print(max)        #torch.return_types.max(
                                            # values=tensor([[ 2,  5, 63, 58],
                                            #         [ 8,  9, 10,  6],
                                            #         [89, 65, 22,  6]]),
                                            # indices=tensor([[1, 0, 1, 1],
                                            #         [1, 1, 1, 0],
                                            #         [1, 0, 1, 0]]))
print(max[0])     #tensor([[ 2,  5, 63, 58],
                            # [ 8,  9, 10,  6],
                            # [89, 65, 22,  6]])
print(max[1])  
                    # tensor([[1, 0, 1, 1],
                    #         [1, 1, 1, 0],
                    #         [1, 0, 1, 0]])
                       