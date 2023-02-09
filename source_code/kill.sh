#!/bin/bash
ps -ef |grep "$1" |grep -v grep |cut -c 9-15 |xargs kill -9
