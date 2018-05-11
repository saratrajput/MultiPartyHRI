import sys
import time
date_string = time.strftime("%Y-%m-%d-%H:%M")

print date_string

f = open(date_string + '.txt', 'w')
