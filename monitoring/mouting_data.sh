#!/bin/bash
echo "--------SCRIPT PROCESSING AT: $(date "+%Y-%m-%d %H:%M:%S")--------"

echo ""
echo "mounting j2 data"
sshfs gdev@10.0.12.30:/home/gdev/tmp/mcmc/ j2/ 

echo ""
echo "mounting j4 data"
sshfs gdev@10.0.12.239:/home/gdev/tmp/mcmc/ j4/ 

echo ""
echo "mounting p2 data"
sshfs gdev@10.0.12.90:/home/gdev/tmp/mcmc/ p2/ 

echo ""
echo "mounting p4 data"
sshfs gdev@10.0.12.21:/home/gdev/tmp/mcmc/ p4/ 

echo ""
echo "mounting p8 data"
sshfs gdev@10.0.12.18:/home/gdev/tmp/mcmc/ p8/ 

echo "---------SCRIPT FINISHED AT: $(date "+%Y-%m-%d %H:%M:%S")---------"
