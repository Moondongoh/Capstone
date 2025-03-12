import numpy as np

# random_state=4와 같은 효과
np.random.seed(4)
print("*********************************************************")
print
print(np.random.rand(3))  # 실행할 때마다 같은 결과

np.random.seed(4)
print("*********************************************************")
print
print(np.random.rand(3))  # 위와 같은 결과가 나옴

np.random.seed(None)
print("*********************************************************")
print
print(np.random.rand(3))  # 위와 같은 결과가 나옴

np.random.seed(None)
print("*********************************************************")
print
print(np.random.rand(3))  # 위와 같은 결과가 나옴

np.random.seed(42)
print("*********************************************************")
print
print(np.random.rand(3))  # 위와 같은 결과가 나옴

# random_state=42와 같은 효과
np.random.seed(42)
print("*********************************************************")
print(np.random.rand(3))  # 다른 패턴이 나옴
