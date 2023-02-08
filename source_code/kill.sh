#!/bin/bash

ps -ef |grep "$1" |grep -v grep |cut -c 9-15 |xargs kill -9
# kill $(ps aux | grep "$1" | awk '{print $2}')
