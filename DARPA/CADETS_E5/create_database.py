import os
import re
import torch
from tqdm import tqdm
import hashlib


from config import *
from kairos_utils import *

filelist = [
 'ta1-cadets-1-e5-official-2.bin.100.json',
 'ta1-cadets-1-e5-official-2.bin.100.json.1',
 'ta1-cadets-1-e5-official-2.bin.100.json.2',
 'ta1-cadets-1-e5-official-2.bin.101.json',
 'ta1-cadets-1-e5-official-2.bin.101.json.1',
 'ta1-cadets-1-e5-official-2.bin.101.json.2',
 'ta1-cadets-1-e5-official-2.bin.102.json',
 'ta1-cadets-1-e5-official-2.bin.102.json.1',
 'ta1-cadets-1-e5-official-2.bin.102.json.2',
 'ta1-cadets-1-e5-official-2.bin.103.json',
 'ta1-cadets-1-e5-official-2.bin.103.json.1',
 'ta1-cadets-1-e5-official-2.bin.103.json.2',
 'ta1-cadets-1-e5-official-2.bin.104.json',
 'ta1-cadets-1-e5-official-2.bin.104.json.1',
 'ta1-cadets-1-e5-official-2.bin.104.json.2',
 'ta1-cadets-1-e5-official-2.bin.105.json',
 'ta1-cadets-1-e5-official-2.bin.105.json.1',
 'ta1-cadets-1-e5-official-2.bin.105.json.2',
 'ta1-cadets-1-e5-official-2.bin.106.json',
 'ta1-cadets-1-e5-official-2.bin.106.json.1',
 'ta1-cadets-1-e5-official-2.bin.106.json.2',
 'ta1-cadets-1-e5-official-2.bin.107.json',
 'ta1-cadets-1-e5-official-2.bin.107.json.1',
 'ta1-cadets-1-e5-official-2.bin.107.json.2',
 'ta1-cadets-1-e5-official-2.bin.108.json',
 'ta1-cadets-1-e5-official-2.bin.108.json.1',
 'ta1-cadets-1-e5-official-2.bin.108.json.2',
 'ta1-cadets-1-e5-official-2.bin.109.json',
 'ta1-cadets-1-e5-official-2.bin.109.json.1',
 'ta1-cadets-1-e5-official-2.bin.109.json.2',
 'ta1-cadets-1-e5-official-2.bin.10.json',
 'ta1-cadets-1-e5-official-2.bin.10.json.1',
 'ta1-cadets-1-e5-official-2.bin.10.json.2',
 'ta1-cadets-1-e5-official-2.bin.110.json',
 'ta1-cadets-1-e5-official-2.bin.110.json.1',
 'ta1-cadets-1-e5-official-2.bin.110.json.2',
 'ta1-cadets-1-e5-official-2.bin.111.json',
 'ta1-cadets-1-e5-official-2.bin.111.json.1',
 'ta1-cadets-1-e5-official-2.bin.111.json.2',
 'ta1-cadets-1-e5-official-2.bin.112.json',
 'ta1-cadets-1-e5-official-2.bin.112.json.1',
 'ta1-cadets-1-e5-official-2.bin.112.json.2',
 'ta1-cadets-1-e5-official-2.bin.113.json',
 'ta1-cadets-1-e5-official-2.bin.113.json.1',
 'ta1-cadets-1-e5-official-2.bin.113.json.2',
 'ta1-cadets-1-e5-official-2.bin.114.json',
 'ta1-cadets-1-e5-official-2.bin.114.json.1',
 'ta1-cadets-1-e5-official-2.bin.114.json.2',
 'ta1-cadets-1-e5-official-2.bin.115.json',
 'ta1-cadets-1-e5-official-2.bin.115.json.1',
 'ta1-cadets-1-e5-official-2.bin.115.json.2',
 'ta1-cadets-1-e5-official-2.bin.116.json',
 'ta1-cadets-1-e5-official-2.bin.116.json.1',
 'ta1-cadets-1-e5-official-2.bin.116.json.2',
 'ta1-cadets-1-e5-official-2.bin.117.json',
 'ta1-cadets-1-e5-official-2.bin.117.json.1',
 'ta1-cadets-1-e5-official-2.bin.117.json.2',
 'ta1-cadets-1-e5-official-2.bin.118.json',
 'ta1-cadets-1-e5-official-2.bin.118.json.1',
 'ta1-cadets-1-e5-official-2.bin.118.json.2',
 'ta1-cadets-1-e5-official-2.bin.119.json',
 'ta1-cadets-1-e5-official-2.bin.119.json.1',
 'ta1-cadets-1-e5-official-2.bin.119.json.2',
 'ta1-cadets-1-e5-official-2.bin.11.json',
 'ta1-cadets-1-e5-official-2.bin.11.json.1',
 'ta1-cadets-1-e5-official-2.bin.11.json.2',
 'ta1-cadets-1-e5-official-2.bin.120.json',
 'ta1-cadets-1-e5-official-2.bin.120.json.1',
 'ta1-cadets-1-e5-official-2.bin.120.json.2',
 'ta1-cadets-1-e5-official-2.bin.121.json',
 'ta1-cadets-1-e5-official-2.bin.121.json.1',
 'ta1-cadets-1-e5-official-2.bin.12.json',
 'ta1-cadets-1-e5-official-2.bin.12.json.1',
 'ta1-cadets-1-e5-official-2.bin.12.json.2',
 'ta1-cadets-1-e5-official-2.bin.13.json',
 'ta1-cadets-1-e5-official-2.bin.13.json.1',
 'ta1-cadets-1-e5-official-2.bin.13.json.2',
 'ta1-cadets-1-e5-official-2.bin.14.json',
 'ta1-cadets-1-e5-official-2.bin.14.json.1',
 'ta1-cadets-1-e5-official-2.bin.14.json.2',
 'ta1-cadets-1-e5-official-2.bin.15.json',
 'ta1-cadets-1-e5-official-2.bin.15.json.1',
 'ta1-cadets-1-e5-official-2.bin.15.json.2',
 'ta1-cadets-1-e5-official-2.bin.16.json',
 'ta1-cadets-1-e5-official-2.bin.16.json.1',
 'ta1-cadets-1-e5-official-2.bin.16.json.2',
 'ta1-cadets-1-e5-official-2.bin.17.json',
 'ta1-cadets-1-e5-official-2.bin.17.json.1',
 'ta1-cadets-1-e5-official-2.bin.17.json.2',
 'ta1-cadets-1-e5-official-2.bin.18.json',
 'ta1-cadets-1-e5-official-2.bin.18.json.1',
 'ta1-cadets-1-e5-official-2.bin.18.json.2',
 'ta1-cadets-1-e5-official-2.bin.19.json',
 'ta1-cadets-1-e5-official-2.bin.19.json.1',
 'ta1-cadets-1-e5-official-2.bin.19.json.2',
 'ta1-cadets-1-e5-official-2.bin.1.json',
 'ta1-cadets-1-e5-official-2.bin.1.json.1',
 'ta1-cadets-1-e5-official-2.bin.1.json.2',
 'ta1-cadets-1-e5-official-2.bin.20.json',
 'ta1-cadets-1-e5-official-2.bin.20.json.1',
 'ta1-cadets-1-e5-official-2.bin.20.json.2',
 'ta1-cadets-1-e5-official-2.bin.21.json',
 'ta1-cadets-1-e5-official-2.bin.21.json.1',
 'ta1-cadets-1-e5-official-2.bin.21.json.2',
 'ta1-cadets-1-e5-official-2.bin.22.json',
 'ta1-cadets-1-e5-official-2.bin.22.json.1',
 'ta1-cadets-1-e5-official-2.bin.22.json.2',
 'ta1-cadets-1-e5-official-2.bin.23.json',
 'ta1-cadets-1-e5-official-2.bin.23.json.1',
 'ta1-cadets-1-e5-official-2.bin.23.json.2',
 'ta1-cadets-1-e5-official-2.bin.24.json',
 'ta1-cadets-1-e5-official-2.bin.24.json.1',
 'ta1-cadets-1-e5-official-2.bin.24.json.2',
 'ta1-cadets-1-e5-official-2.bin.25.json',
 'ta1-cadets-1-e5-official-2.bin.25.json.1',
 'ta1-cadets-1-e5-official-2.bin.25.json.2',
 'ta1-cadets-1-e5-official-2.bin.26.json',
 'ta1-cadets-1-e5-official-2.bin.26.json.1',
 'ta1-cadets-1-e5-official-2.bin.26.json.2',
 'ta1-cadets-1-e5-official-2.bin.27.json',
 'ta1-cadets-1-e5-official-2.bin.27.json.1',
 'ta1-cadets-1-e5-official-2.bin.27.json.2',
 'ta1-cadets-1-e5-official-2.bin.28.json',
 'ta1-cadets-1-e5-official-2.bin.28.json.1',
 'ta1-cadets-1-e5-official-2.bin.28.json.2',
 'ta1-cadets-1-e5-official-2.bin.29.json',
 'ta1-cadets-1-e5-official-2.bin.29.json.1',
 'ta1-cadets-1-e5-official-2.bin.29.json.2',
 'ta1-cadets-1-e5-official-2.bin.2.json',
 'ta1-cadets-1-e5-official-2.bin.2.json.1',
 'ta1-cadets-1-e5-official-2.bin.2.json.2',
 'ta1-cadets-1-e5-official-2.bin.30.json',
 'ta1-cadets-1-e5-official-2.bin.30.json.1',
 'ta1-cadets-1-e5-official-2.bin.30.json.2',
 'ta1-cadets-1-e5-official-2.bin.31.json',
 'ta1-cadets-1-e5-official-2.bin.31.json.1',
 'ta1-cadets-1-e5-official-2.bin.31.json.2',
 'ta1-cadets-1-e5-official-2.bin.32.json',
 'ta1-cadets-1-e5-official-2.bin.32.json.1',
 'ta1-cadets-1-e5-official-2.bin.32.json.2',
 'ta1-cadets-1-e5-official-2.bin.33.json',
 'ta1-cadets-1-e5-official-2.bin.33.json.1',
 'ta1-cadets-1-e5-official-2.bin.33.json.2',
 'ta1-cadets-1-e5-official-2.bin.34.json',
 'ta1-cadets-1-e5-official-2.bin.34.json.1',
 'ta1-cadets-1-e5-official-2.bin.34.json.2',
 'ta1-cadets-1-e5-official-2.bin.35.json',
 'ta1-cadets-1-e5-official-2.bin.35.json.1',
 'ta1-cadets-1-e5-official-2.bin.35.json.2',
 'ta1-cadets-1-e5-official-2.bin.36.json',
 'ta1-cadets-1-e5-official-2.bin.36.json.1',
 'ta1-cadets-1-e5-official-2.bin.36.json.2',
 'ta1-cadets-1-e5-official-2.bin.37.json',
 'ta1-cadets-1-e5-official-2.bin.37.json.1',
 'ta1-cadets-1-e5-official-2.bin.37.json.2',
 'ta1-cadets-1-e5-official-2.bin.38.json',
 'ta1-cadets-1-e5-official-2.bin.38.json.1',
 'ta1-cadets-1-e5-official-2.bin.38.json.2',
 'ta1-cadets-1-e5-official-2.bin.39.json',
 'ta1-cadets-1-e5-official-2.bin.39.json.1',
 'ta1-cadets-1-e5-official-2.bin.39.json.2',
 'ta1-cadets-1-e5-official-2.bin.3.json',
 'ta1-cadets-1-e5-official-2.bin.3.json.1',
 'ta1-cadets-1-e5-official-2.bin.3.json.2',
 'ta1-cadets-1-e5-official-2.bin.40.json',
 'ta1-cadets-1-e5-official-2.bin.40.json.1',
 'ta1-cadets-1-e5-official-2.bin.40.json.2',
 'ta1-cadets-1-e5-official-2.bin.41.json',
 'ta1-cadets-1-e5-official-2.bin.41.json.1',
 'ta1-cadets-1-e5-official-2.bin.41.json.2',
 'ta1-cadets-1-e5-official-2.bin.42.json',
 'ta1-cadets-1-e5-official-2.bin.42.json.1',
 'ta1-cadets-1-e5-official-2.bin.42.json.2',
 'ta1-cadets-1-e5-official-2.bin.43.json',
 'ta1-cadets-1-e5-official-2.bin.43.json.1',
 'ta1-cadets-1-e5-official-2.bin.43.json.2',
 'ta1-cadets-1-e5-official-2.bin.44.json',
 'ta1-cadets-1-e5-official-2.bin.44.json.1',
 'ta1-cadets-1-e5-official-2.bin.44.json.2',
 'ta1-cadets-1-e5-official-2.bin.45.json',
 'ta1-cadets-1-e5-official-2.bin.45.json.1',
 'ta1-cadets-1-e5-official-2.bin.45.json.2',
 'ta1-cadets-1-e5-official-2.bin.46.json',
 'ta1-cadets-1-e5-official-2.bin.46.json.1',
 'ta1-cadets-1-e5-official-2.bin.46.json.2',
 'ta1-cadets-1-e5-official-2.bin.47.json',
 'ta1-cadets-1-e5-official-2.bin.47.json.1',
 'ta1-cadets-1-e5-official-2.bin.47.json.2',
 'ta1-cadets-1-e5-official-2.bin.48.json',
 'ta1-cadets-1-e5-official-2.bin.48.json.1',
 'ta1-cadets-1-e5-official-2.bin.48.json.2',
 'ta1-cadets-1-e5-official-2.bin.49.json',
 'ta1-cadets-1-e5-official-2.bin.49.json.1',
 'ta1-cadets-1-e5-official-2.bin.49.json.2',
 'ta1-cadets-1-e5-official-2.bin.4.json',
 'ta1-cadets-1-e5-official-2.bin.4.json.1',
 'ta1-cadets-1-e5-official-2.bin.4.json.2',
 'ta1-cadets-1-e5-official-2.bin.50.json',
 'ta1-cadets-1-e5-official-2.bin.50.json.1',
 'ta1-cadets-1-e5-official-2.bin.50.json.2',
 'ta1-cadets-1-e5-official-2.bin.51.json',
 'ta1-cadets-1-e5-official-2.bin.51.json.1',
 'ta1-cadets-1-e5-official-2.bin.51.json.2',
 'ta1-cadets-1-e5-official-2.bin.52.json',
 'ta1-cadets-1-e5-official-2.bin.52.json.1',
 'ta1-cadets-1-e5-official-2.bin.52.json.2',
 'ta1-cadets-1-e5-official-2.bin.53.json',
 'ta1-cadets-1-e5-official-2.bin.53.json.1',
 'ta1-cadets-1-e5-official-2.bin.53.json.2',
 'ta1-cadets-1-e5-official-2.bin.54.json',
 'ta1-cadets-1-e5-official-2.bin.54.json.1',
 'ta1-cadets-1-e5-official-2.bin.54.json.2',
 'ta1-cadets-1-e5-official-2.bin.55.json',
 'ta1-cadets-1-e5-official-2.bin.55.json.1',
 'ta1-cadets-1-e5-official-2.bin.55.json.2',
 'ta1-cadets-1-e5-official-2.bin.56.json',
 'ta1-cadets-1-e5-official-2.bin.56.json.1',
 'ta1-cadets-1-e5-official-2.bin.56.json.2',
 'ta1-cadets-1-e5-official-2.bin.57.json',
 'ta1-cadets-1-e5-official-2.bin.57.json.1',
 'ta1-cadets-1-e5-official-2.bin.57.json.2',
 'ta1-cadets-1-e5-official-2.bin.58.json',
 'ta1-cadets-1-e5-official-2.bin.58.json.1',
 'ta1-cadets-1-e5-official-2.bin.58.json.2',
 'ta1-cadets-1-e5-official-2.bin.59.json',
 'ta1-cadets-1-e5-official-2.bin.59.json.1',
 'ta1-cadets-1-e5-official-2.bin.59.json.2',
 'ta1-cadets-1-e5-official-2.bin.5.json',
 'ta1-cadets-1-e5-official-2.bin.5.json.1',
 'ta1-cadets-1-e5-official-2.bin.5.json.2',
 'ta1-cadets-1-e5-official-2.bin.60.json',
 'ta1-cadets-1-e5-official-2.bin.60.json.1',
 'ta1-cadets-1-e5-official-2.bin.60.json.2',
 'ta1-cadets-1-e5-official-2.bin.61.json',
 'ta1-cadets-1-e5-official-2.bin.61.json.1',
 'ta1-cadets-1-e5-official-2.bin.61.json.2',
 'ta1-cadets-1-e5-official-2.bin.62.json',
 'ta1-cadets-1-e5-official-2.bin.62.json.1',
 'ta1-cadets-1-e5-official-2.bin.62.json.2',
 'ta1-cadets-1-e5-official-2.bin.63.json',
 'ta1-cadets-1-e5-official-2.bin.63.json.1',
 'ta1-cadets-1-e5-official-2.bin.63.json.2',
 'ta1-cadets-1-e5-official-2.bin.64.json',
 'ta1-cadets-1-e5-official-2.bin.64.json.1',
 'ta1-cadets-1-e5-official-2.bin.64.json.2',
 'ta1-cadets-1-e5-official-2.bin.65.json',
 'ta1-cadets-1-e5-official-2.bin.65.json.1',
 'ta1-cadets-1-e5-official-2.bin.65.json.2',
 'ta1-cadets-1-e5-official-2.bin.66.json',
 'ta1-cadets-1-e5-official-2.bin.66.json.1',
 'ta1-cadets-1-e5-official-2.bin.66.json.2',
 'ta1-cadets-1-e5-official-2.bin.67.json',
 'ta1-cadets-1-e5-official-2.bin.67.json.1',
 'ta1-cadets-1-e5-official-2.bin.67.json.2',
 'ta1-cadets-1-e5-official-2.bin.68.json',
 'ta1-cadets-1-e5-official-2.bin.68.json.1',
 'ta1-cadets-1-e5-official-2.bin.68.json.2',
 'ta1-cadets-1-e5-official-2.bin.69.json',
 'ta1-cadets-1-e5-official-2.bin.69.json.1',
 'ta1-cadets-1-e5-official-2.bin.69.json.2',
 'ta1-cadets-1-e5-official-2.bin.6.json',
 'ta1-cadets-1-e5-official-2.bin.6.json.1',
 'ta1-cadets-1-e5-official-2.bin.6.json.2',
 'ta1-cadets-1-e5-official-2.bin.70.json',
 'ta1-cadets-1-e5-official-2.bin.70.json.1',
 'ta1-cadets-1-e5-official-2.bin.70.json.2',
 'ta1-cadets-1-e5-official-2.bin.71.json',
 'ta1-cadets-1-e5-official-2.bin.71.json.1',
 'ta1-cadets-1-e5-official-2.bin.71.json.2',
 'ta1-cadets-1-e5-official-2.bin.72.json',
 'ta1-cadets-1-e5-official-2.bin.72.json.1',
 'ta1-cadets-1-e5-official-2.bin.72.json.2',
 'ta1-cadets-1-e5-official-2.bin.73.json',
 'ta1-cadets-1-e5-official-2.bin.73.json.1',
 'ta1-cadets-1-e5-official-2.bin.73.json.2',
 'ta1-cadets-1-e5-official-2.bin.74.json',
 'ta1-cadets-1-e5-official-2.bin.74.json.1',
 'ta1-cadets-1-e5-official-2.bin.74.json.2',
 'ta1-cadets-1-e5-official-2.bin.75.json',
 'ta1-cadets-1-e5-official-2.bin.75.json.1',
 'ta1-cadets-1-e5-official-2.bin.75.json.2',
 'ta1-cadets-1-e5-official-2.bin.76.json',
 'ta1-cadets-1-e5-official-2.bin.76.json.1',
 'ta1-cadets-1-e5-official-2.bin.76.json.2',
 'ta1-cadets-1-e5-official-2.bin.77.json',
 'ta1-cadets-1-e5-official-2.bin.77.json.1',
 'ta1-cadets-1-e5-official-2.bin.77.json.2',
 'ta1-cadets-1-e5-official-2.bin.78.json',
 'ta1-cadets-1-e5-official-2.bin.78.json.1',
 'ta1-cadets-1-e5-official-2.bin.78.json.2',
 'ta1-cadets-1-e5-official-2.bin.79.json',
 'ta1-cadets-1-e5-official-2.bin.79.json.1',
 'ta1-cadets-1-e5-official-2.bin.79.json.2',
 'ta1-cadets-1-e5-official-2.bin.7.json',
 'ta1-cadets-1-e5-official-2.bin.7.json.1',
 'ta1-cadets-1-e5-official-2.bin.7.json.2',
 'ta1-cadets-1-e5-official-2.bin.80.json',
 'ta1-cadets-1-e5-official-2.bin.80.json.1',
 'ta1-cadets-1-e5-official-2.bin.80.json.2',
 'ta1-cadets-1-e5-official-2.bin.81.json',
 'ta1-cadets-1-e5-official-2.bin.81.json.1',
 'ta1-cadets-1-e5-official-2.bin.81.json.2',
 'ta1-cadets-1-e5-official-2.bin.82.json',
 'ta1-cadets-1-e5-official-2.bin.82.json.1',
 'ta1-cadets-1-e5-official-2.bin.82.json.2',
 'ta1-cadets-1-e5-official-2.bin.83.json',
 'ta1-cadets-1-e5-official-2.bin.83.json.1',
 'ta1-cadets-1-e5-official-2.bin.83.json.2',
 'ta1-cadets-1-e5-official-2.bin.84.json',
 'ta1-cadets-1-e5-official-2.bin.84.json.1',
 'ta1-cadets-1-e5-official-2.bin.84.json.2',
 'ta1-cadets-1-e5-official-2.bin.85.json',
 'ta1-cadets-1-e5-official-2.bin.85.json.1',
 'ta1-cadets-1-e5-official-2.bin.85.json.2',
 'ta1-cadets-1-e5-official-2.bin.86.json',
 'ta1-cadets-1-e5-official-2.bin.86.json.1',
 'ta1-cadets-1-e5-official-2.bin.86.json.2',
 'ta1-cadets-1-e5-official-2.bin.87.json',
 'ta1-cadets-1-e5-official-2.bin.87.json.1',
 'ta1-cadets-1-e5-official-2.bin.87.json.2',
 'ta1-cadets-1-e5-official-2.bin.88.json',
 'ta1-cadets-1-e5-official-2.bin.88.json.1',
 'ta1-cadets-1-e5-official-2.bin.88.json.2',
 'ta1-cadets-1-e5-official-2.bin.89.json',
 'ta1-cadets-1-e5-official-2.bin.89.json.1',
 'ta1-cadets-1-e5-official-2.bin.89.json.2',
 'ta1-cadets-1-e5-official-2.bin.8.json',
 'ta1-cadets-1-e5-official-2.bin.8.json.1',
 'ta1-cadets-1-e5-official-2.bin.8.json.2',
 'ta1-cadets-1-e5-official-2.bin.90.json',
 'ta1-cadets-1-e5-official-2.bin.90.json.1',
 'ta1-cadets-1-e5-official-2.bin.90.json.2',
 'ta1-cadets-1-e5-official-2.bin.91.json',
 'ta1-cadets-1-e5-official-2.bin.91.json.1',
 'ta1-cadets-1-e5-official-2.bin.91.json.2',
 'ta1-cadets-1-e5-official-2.bin.92.json',
 'ta1-cadets-1-e5-official-2.bin.92.json.1',
 'ta1-cadets-1-e5-official-2.bin.92.json.2',
 'ta1-cadets-1-e5-official-2.bin.93.json',
 'ta1-cadets-1-e5-official-2.bin.93.json.1',
 'ta1-cadets-1-e5-official-2.bin.93.json.2',
 'ta1-cadets-1-e5-official-2.bin.94.json',
 'ta1-cadets-1-e5-official-2.bin.94.json.1',
 'ta1-cadets-1-e5-official-2.bin.94.json.2',
 'ta1-cadets-1-e5-official-2.bin.95.json',
 'ta1-cadets-1-e5-official-2.bin.95.json.1',
 'ta1-cadets-1-e5-official-2.bin.95.json.2',
 'ta1-cadets-1-e5-official-2.bin.96.json',
 'ta1-cadets-1-e5-official-2.bin.96.json.1',
 'ta1-cadets-1-e5-official-2.bin.96.json.2',
 'ta1-cadets-1-e5-official-2.bin.97.json',
 'ta1-cadets-1-e5-official-2.bin.97.json.1',
 'ta1-cadets-1-e5-official-2.bin.97.json.2',
 'ta1-cadets-1-e5-official-2.bin.98.json',
 'ta1-cadets-1-e5-official-2.bin.98.json.1',
 'ta1-cadets-1-e5-official-2.bin.98.json.2',
 'ta1-cadets-1-e5-official-2.bin.99.json',
 'ta1-cadets-1-e5-official-2.bin.99.json.1',
 'ta1-cadets-1-e5-official-2.bin.99.json.2',
 'ta1-cadets-1-e5-official-2.bin.9.json',
 'ta1-cadets-1-e5-official-2.bin.9.json.1',
 'ta1-cadets-1-e5-official-2.bin.9.json.2',
 'ta1-cadets-1-e5-official-2.bin.json',
 'ta1-cadets-1-e5-official-2.bin.json.1',
 'ta1-cadets-1-e5-official-2.bin.json.2'
]


def stringtomd5(originstr):
    originstr = originstr.encode("utf-8")
    signaturemd5 = hashlib.sha256()
    signaturemd5.update(originstr)
    return signaturemd5.hexdigest()

def store_netflow(file_path, cur, connect):
    # Parse data from logs
    netobjset = set()
    netobj2hash = {}
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in f:
                if "avro.cdm20.NetFlowObject" in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":{"string":"(.*?)"},"localPort":{"int":(.*?)},"remoteAddress":{"string":"(.*?)"},"remotePort":{"int":(.*?)}',
                            line)[0]

                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]

                        nodeproperty = srcaddr + "," + srcport + "," + dstaddr + "," + dstport  # 只关注向哪里发起的网络流 合理么？
                        hashstr = stringtomd5(nodeproperty)
                        netobj2hash[nodeid] = [hashstr, nodeproperty]
                        netobj2hash[hashstr] = nodeid
                        netobjset.add(hashstr)
                    except:
                        pass

    # Store data into database
    datalist = []
    for i in netobj2hash.keys():
        if len(i) != 64:
            datalist.append([i] + [netobj2hash[i][0]] + netobj2hash[i][1].split(","))

    sql = '''insert into netflow_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

def get_subject_file_uuids(file_path):
    subject_uuid2path = {}
    file_uuid2path = {}

    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "schema.avro.cdm20.Subject" in line:
                    pattern = '{"com.bbn.tc.schema.avro.cdm20.Subject":{"uuid":"(.*?)"'
                    match_ans = re.findall(pattern, line)[0]
                    subject_uuid2path[match_ans] = 'none'
                elif "schema.avro.cdm20.FileObject" in line:
                    pattern = '{"com.bbn.tc.schema.avro.cdm20.FileObject":{"uuid":"(.*?)"'
                    match_ans = re.findall(pattern, line)[0]
                    file_uuid2path[match_ans] = 'none'
    return subject_uuid2path, file_uuid2path

def store_subject(file_path, cur, connect, subject_uuid2path):
    # Parse data from logs
    fail_count = 0
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "schema.avro.cdm20.Event" in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type in include_edge_type:
                        # 0: subject uuid  1:object uuid  2 object path name   -1: subject name
                        try:
                            pattern = '"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},(.*?)"exec":"(.*?)",'
                            match_ans = re.findall(pattern, line)
                            if match_ans[0][0] in subject_uuid2path:
                                subject_uuid2path[match_ans[0][0]] = match_ans[0][-1]
                        except:
                            fail_count += 1
    # Store into database
    datalist = []
    for i in subject_uuid2path.keys():
        if subject_uuid2path[i] != 'none':
            datalist.append([i] + [stringtomd5(subject_uuid2path[i]), subject_uuid2path[i]])

    sql = '''insert into subject_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist, page_size=10000)
    connect.commit()

def store_file(file_path, cur, connect, file_uuid2path):
    fail_count = 0
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if "schema.avro.cdm20.Event" in line:
                    relation_type = re.findall('"type":"(.*?)"', line)[0]
                    if relation_type in include_edge_type:
                        try:
                            object_uuid = \
                            re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"},', line)[0]
                            if object_uuid in file_uuid2path:
                                object_path = re.findall('"predicateObjectPath":{"string":"(.*?)"}', line)
                                if len(object_path) == 0:
                                    file_uuid2path[object_uuid] = 'null'
                                else:
                                    file_uuid2path[object_uuid] = object_path[0]
                        except:
                            fail_count += 1

    datalist = []
    for i in file_uuid2path.keys():
        if file_uuid2path[i] != 'none':
            datalist.append([i] + [stringtomd5(file_uuid2path[i]), file_uuid2path[i]])
    datalist_new = []
    for i in datalist:
        if i[-1] != 'null':
            datalist_new.append(i)

    sql = '''insert into file_node_table
                         values %s
            '''
    ex.execute_values(cur, sql, datalist_new, page_size=10000)
    connect.commit()

def create_node_list(cur, connect):
    node_list = {}

    # file
    sql = """
    select * from file_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["file", i[-1]]
    file_uuid2hash = {}
    for i in records:
        file_uuid2hash[i[0]] = i[1]

    # subject
    sql = """
    select * from subject_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["subject", i[-1]]
    subject_uuid2hash = {}
    for i in records:
        subject_uuid2hash[i[0]] = i[1]

    # netflow
    sql = """
    select * from netflow_node_table;
    """
    cur.execute(sql)
    records = cur.fetchall()
    for i in records:
        node_list[i[1]] = ["netflow", i[-2] + ":" + i[-1]]
    net_uuid2hash = {}
    for i in records:
        net_uuid2hash[i[0]] = i[1]

    node_list_database = []
    node_index = 0
    for i in node_list:
        node_list_database.append([i] + node_list[i] + [node_index])
        node_index += 1

    sql = '''insert into node2id
                         values %s
            '''
    ex.execute_values(cur, sql, node_list_database, page_size=10000)
    connect.commit()

    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash

def write_event_in_DB(cur, connect, datalist):
    sql = '''insert into event_table
                         values %s
            '''
    ex.execute_values(cur,sql, datalist,page_size=10000)
    connect.commit()

def store_event(file_path, cur, connect, reverse, nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash):
    datalist = []
    for file in tqdm(filelist):
        with open(file_path + file, "r") as f:
            for line in (f):
                if '{"datum":{"com.bbn.tc.schema.avro.cdm20.Event"' in line:
                    subject_uuid = re.findall('"subject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"}', line)
                    predicateObject_uuid = re.findall('"predicateObject":{"com.bbn.tc.schema.avro.cdm20.UUID":"(.*?)"}',
                                                      line)
                    if len(subject_uuid) > 0 and len(predicateObject_uuid) > 0:
                        if subject_uuid[0] in subject_uuid2hash and (predicateObject_uuid[0] in file_uuid2hash or predicateObject_uuid[0] in net_uuid2hash):
                            relation_type = re.findall('"type":"(.*?)"', line)[0]
                            time_rec = re.findall('"timestampNanos":(.*?),', line)[0]
                            time_rec = int(time_rec)
                            subjectId = subject_uuid2hash[subject_uuid[0]]
                            if predicateObject_uuid[0] in file_uuid2hash:
                                objectId = file_uuid2hash[predicateObject_uuid[0]]
                            else:
                                objectId = net_uuid2hash[predicateObject_uuid[0]]
                            if relation_type in reverse:
                                datalist.append(
                                    [objectId, nodeid2msg[objectId], relation_type, subjectId, nodeid2msg[subjectId],
                                     time_rec])
                            else:
                                datalist.append(
                                    [subjectId, nodeid2msg[subjectId], relation_type, objectId, nodeid2msg[objectId],
                                     time_rec])
                            if len(datalist) == 50000:
                                write_event_in_DB(cur, connect, datalist)
                                datalist.clear()

    write_event_in_DB(cur, connect, datalist)


if __name__ == "__main__":
    cur, connect = init_database_connection()

    # There will be 219384 netflow nodes stored in the table
    print("Processing netflow data")
    store_netflow(file_path=raw_dir, cur=cur, connect=connect)

    print("Extract the uuid of subject and file data")
    subject_uuid2path, file_uuid2path = get_subject_file_uuids(file_path=raw_dir)

    # There will be 6257008 subject nodes stored in the table
    print("Processing subject data")
    store_subject(file_path=raw_dir, cur=cur, connect=connect, subject_uuid2path=subject_uuid2path)

    # There will be 263098 file nodes stored in the table
    print("Processing file data")
    store_file(file_path=raw_dir, cur=cur, connect=connect, file_uuid2path=file_uuid2path)

    # There will be 262625 entities stored in the table
    print("Extracting the node list")
    nodeid2msg, subject_uuid2hash, file_uuid2hash, net_uuid2hash = create_node_list(cur=cur, connect=connect)

    # There will be 37511113 events stored in the table
    print("Processing the events")
    store_event(
        file_path=raw_dir,
        cur=cur,
        connect=connect,
        reverse=edge_reversed,
        nodeid2msg=nodeid2msg,
        subject_uuid2hash=subject_uuid2hash,
        file_uuid2hash=file_uuid2hash,
        net_uuid2hash=net_uuid2hash
    )