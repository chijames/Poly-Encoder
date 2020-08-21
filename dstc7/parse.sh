mv ubuntu_train_subtask_1.json train.json
mv ubuntu_dev_subtask_1.json dev.json

python3 parse.py --mode train
python3 parse.py --mode dev
python3 merge.py # combine test file and ans
python3 parse.py --mode test
