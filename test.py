def generate_number():
    i = 0
    a = [i + 1 for i in range(1000)]
    return a


data = generate_number()
temp = data[-100:]

print(data)
print(temp[-1])
print(len(temp))
