count = 5

def recursion(value):
    if value == 0:
        return
    else:
        for i in range(value):
            storage = [recursion(i)]

