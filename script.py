import math
import sys
from os import rename


import requests

print(sys.version)

# check virtual environment you're using
print(sys.executable)


# def greet(who_to_greet):
#     greeting = "Hello, {}".format(who_to_greet)
#     return greeting


r = requests.get("https://www.youtube.com/watch?v=06I63_p-2A4")
print(r.status_code)

# name = input("Your Name? ")
# print("Hello, ", name)