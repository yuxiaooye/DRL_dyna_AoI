'''
draw visualized trajectories for a group of exp, generate .html for each exp
can be called by post_process.py to draw groups of exp
'''

import os
import argparse
from tools.post.vis import render_HTML

parser = argparse.ArgumentParser()
parser.add_argument('--group_dir', type=str)
args = parser.parse_args()

exps = []
for root, dirs, files in os.walk(args.group_dir):
    if root == args.group_dir:
        for dir in dirs:
            exps.append(os.path.join(root, dir))

for output_dir in exps:
    print(output_dir)
    render_HTML(output_dir)





