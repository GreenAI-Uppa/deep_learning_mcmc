#!/bin/bash
echo "--------SCRIPT PROCESSING AT: $(date "+%Y-%m-%d %H:%M:%S")--------"
TEMP=$1
if [ $TEMP ]; then
    echo "pushing to ansa"
    scp -r {j2,j4,p2,p4,p8} mfrancois@10.0.12.25:~/Documents/mas/
    # useful command: 
        # nohup python p8/run.py > logs/p8.logs 2>&1 & \
        # nohup python p2/run.py > logs/p2.logs 2>&1 & \
        # nohup python j4/run.py > logs/j4.logs 2>&1 & \
        # nohup python p4/run.py > logs/p4.logs 2>&1 & \
        # nohup python j2/run.py > logs/j2.logs 2>&1 &


else

    echo ""
    echo "pushing j2"
    scp -r j2/* gdev@10.0.12.30:~/script

    echo ""
    echo "pushing j4"
    scp -r j4/* gdev@10.0.12.239:~/script

    echo ""
    echo "pushing p2"
    scp -r p2/* gdev@10.0.12.90:~/script

    echo ""
    echo "pushing p4"
    scp -r p4/* gdev@10.0.12.21:~/script

    echo ""
    echo "pushing p8"
    scp -r p8/* gdev@10.0.12.18:~/script

fi
echo "---------SCRIPT FINISHED AT: $(date "+%Y-%m-%d %H:%M:%S")---------"

 
