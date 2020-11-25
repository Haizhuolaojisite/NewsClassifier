import numpy as np
import pandas as pd

# dataset is given below
dataset = [[101, 31, "M", 310, 518, 677, 1505],
           [102, 53, "M", 356, 548, 629, 1520],
           [106, 26, "M", 387, 605, 553, 1545],
           [113, 15, "U", 301, 508, 524, 1333],
           [115, 21, "U", 376, 627, 647, 1638],
           [120, 10, "M", 289, 482, 682, 1453],
           [121, 22, "U", 402, 674, 518, 1594],
           [110, 47, "M", 319, 536, 518, 1594],
           [122, 13, "U", 297, 502, 590, 1403],
           [130, 19, "U", 355, 583, 684, 1622],
           [131, 72, "U", 265, 471, 544, 1280],
           [133, 65, "M", 286, 496, 677, 1460],
           [125, 16, "M", 347, 568, 518, 1433],
           [141, 28, "M", 416, 688, 561, 1665]
           ]

# INPUT [uncomment & modify if required]

# input = []
# result1 = []
# for i in dataset:
#     if i[1]<18 and i[2]=='M':
#         input.append(i[0])
#         input.append(2)
#         result1.append('U')
#     if (i[3]+i[4]+i[5]!=i[6]):
#         c = i[3]+i[4]+i[5]
#         input.append(i[0])
#         input.append(6)
#         result1.append(c)
df = pd.DataFrame(dataset)


def checkdata(n, col):
    result = []
    n = int(n)
    col = int(col)
    rowidx = df.index[df[0] == n].tolist()[0]
    if col == 3:
        if df[rowidx, 1] < 18 and df[rowidx, 2] == 'M':
            result.append('U')

    if col == 7:
        s = df[rowidx, 3] + df[rowidx, 4] + df[rowidx, 5]
        t = df[rowidx, 6]
        if s != t:
            result.append(s)

    return result