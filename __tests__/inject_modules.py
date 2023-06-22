import sys
import os

current = os.path.dirname(os.path.relpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
