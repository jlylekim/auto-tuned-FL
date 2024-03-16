#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
from options import args_parser
from main import run 

if __name__ == '__main__':

    # define paths
    path_project = os.path.abspath('..')

    args = args_parser()

    seeds = [5821]
    # seeds = [5821, 1004, 6078]

    for r, seed in enumerate(seeds):
        run(args, seed)



