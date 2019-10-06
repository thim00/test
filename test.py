import numpy as np

test = "Hello world!!"
test_caps = test.upper()
print(test)
print(test_caps)


numbers = [2, 4, 8]

for number in numbers:
    print(number)


sieve = [True] * 101
for i in range(2,100):
    if sieve[i]:
        print(i)
        for j in range(i*i, 100, i):
            sieve[j] = False


test_data = [""] * 9
print(test_data)