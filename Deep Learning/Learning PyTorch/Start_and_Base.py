import torch


# region random matrix for dimension 1
x1 = torch.empty(1)
print(x1)
# endregion

# region random matrix for dimension 2
x2 = torch.zeros(2, 2, dtype=torch.int32)
print(x2)
print(x2.size())
# endregion

# region random matrix for dimension 3
x3 = torch.randn(3, 3, 3, dtype=torch.double)
print(x3)
print(x3.size())
# endregion

# region random matrix for dimension 4
x4 = torch.randn(2, 2, 2, 3, dtype=torch.float16)
print(x4)
print(x4.size())
# endregion


# region creating a manual tensor
x5 = torch.tensor([2, 3])
print(x5)
print(x5.size())
# endregion


if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    print(x)





