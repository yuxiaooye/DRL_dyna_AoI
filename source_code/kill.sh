#!/bin/bash
ps -ef |grep "$1" |grep "$2" |grep -v grep |cut -c 9-15 |xargs kill -9
