
python main.py --mode learning --cnn mobilenetv2 --num-step 5 --step-over one_service --net zte.md --wave-num 15 --save-interval 200 --file-prefix "./" --k 1 --workers 8 --steps 10e6 --base-lr 7e-3 --img-height 224 --img-width 224 --cuda "True" --expand-factor 3 --node-size 45.0 
