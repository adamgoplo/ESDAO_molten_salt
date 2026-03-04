import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

print("NumPy OK, pi =", np.pi)
print("CoolProp OK, woda h =", CP.PropsSI("H", "T", 300, "P", 1e5, "Water"))

T = np.linspace(200, 400, 50)
plt.plot(T, T)
plt.xlabel("T [C]")
plt.ylabel("T [C]")
plt.title("Test matplotlib")
plt.show()
