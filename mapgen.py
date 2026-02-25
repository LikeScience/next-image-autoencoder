import sys
import random
dir = "maps"

height, width, mapv_type = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

wallprob=0.8

if mapv_type[:5] != "color":
    mapv = [[1 for j in range(width)] for i in range(height)]
else:
    mapv = [[3 for j in range(width)] for i in range(height)]

for i in range(height):
    mapv[i][0] = 2
    mapv[i][width-1] = 2

for i in range(width):
    mapv[0][i] = 2
    mapv[height-1][i] = 2

if mapv_type[-5:] == "walls":
    for i in range(1,height-1):
        for j in range(1,width-1):
            if random.random() > wallprob and (i > 1 or j > 1):
                mapv[i][j] = 2

if mapv_type[:5] == "color":
    mapc = [[5 for j in range(width)] for i in range(height)]
    for i in range(1,height-1):
        for j in range(1,width-1):
            if (mapv[i][j] != 2):
                mapc[i][j] = random.choice([0,1])


with open(f"{dir}/{width}x{height}_{mapv_type}.txt", "w") as file:
    for row in mapv:
        file.write(" ".join(map(str, row)) + "\n")

with open(f"{dir}/colors/{width}x{height}_{mapv_type}.txt", "w") as file:
    for row in mapc:
        file.write(" ".join(map(str, row)) + "\n")


