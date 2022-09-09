#!/bin/bash
echo "--------SCRIPT PROCESSING AT: $(date "+%Y-%m-%d %H:%M:%S")--------"

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

echo "---------SCRIPT FINISHED AT: $(date "+%Y-%m-%d %H:%M:%S")---------"
