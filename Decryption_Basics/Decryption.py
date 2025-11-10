import math

n = 50381
e = 11

sq_n = math.sqrt(n)

p = 0
q = 0

for k in range(2, int(sq_n)):
    if n % k == 0:
        p = k
        break
q = n // p

d = []
phi = (p-1) * (q-1)

for k in range(2, (p-1) * (q-1)):
    if e*k % phi == 1:
        d.append(k)

print("n = " + str(n))
print("e = " + str(e))
print("p = " + str(p))
print("q = " + str(q))
print("d = ")
print(d)
