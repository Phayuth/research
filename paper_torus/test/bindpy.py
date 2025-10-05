import sys

sys.path.append("../build/")
import my_module

result = my_module.add(5, 3)
print(f"The result is: {result}")


qa = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
qb = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
path = my_module.plan(qa, qb)
print(path)
