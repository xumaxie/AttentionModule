import torch 
a = torch.tensor([[1, 2]])
a2=torch.tensor([[1],[2]])
a3=torch.tensor([[1,2],[3,4]])
b = torch.tensor([[4, 9], [5, 8]])

print(a*b)
print(a2*b)
print(a3*b)

print(torch.mul(a,b))
print(torch.mul(a2,b))
print(torch.mul(a3,b))
# print(a*b==b*a)
# print(a2*b==b*a2)
# print(a3*b==b*a3)
                    # tensor([[True, True],
                    #         [True, True]])
                    # tensor([[True, True],
                    #         [True, True]])
                    # tensor([[True, True],
                    #         [True, True]])