import os
import sys

# get current directory and append to path
# this allows everything inside perf4sight to access anything else 
# within perf4sight regardless of where perf4sight is
currDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currDir)

from create_network_dataset import create_dataset
from fingerprinting.fingerprint import fingerprint_device
from profiling.network_wise_profiling import profile_network
