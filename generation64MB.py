import os

SIZE_MB = 32
size = SIZE_MB * 1024 * 1024

data = os.urandom(size)

with open("input.bin", "wb") as f:
    f.write(data)

print(f"Generated {size} bytes (multiple of 16)")
