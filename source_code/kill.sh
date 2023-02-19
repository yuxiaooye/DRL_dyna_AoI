#!/bin/bash
ps -ef |grep "wh" |grep -v grep |cut -c 9-15 |xargs kill -9
