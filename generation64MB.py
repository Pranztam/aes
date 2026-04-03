import os

size = 32 * 1024 * 1024
data = os.urandom(size)
with open("input.bin", "wb") as f:
    f.write(data)

print(f"Generated {size} bytes")
