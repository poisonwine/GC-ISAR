nohup mpirun -np 3 python train.py --env FetchPush  --method isar --seed 6 --expert_percent 0.5 --random_percent 0.5 --relabel True --threshold 0.05  --wtd_temperature 2.0  --alpha 2.5   --relabel_percent 1.0 >wtd_6.log 2>&1 &
nohup mpirun -np 3 python train.py --env FetchPush --method isar --seed 7 --expert_percent 0.5 --random_percent 0.5 --relabel True  --threshold 0.05 --wtd_temperature 2.0    --alpha 2.5  --relabel_percent 1.0 >wtd_7.log 2>&1 &

