bit=12
python train.py \
--batch-size 32 \
--learning_rate 1e-3 \
--seed 2000 \
--bit $bit

bit=24
python train.py \
--batch-size 32 \
--learning_rate 1e-3 \
--seed 2000 \
--bit $bit

bit=32
python train.py \
--batch-size 32 \
--learning_rate 1e-3 \
--seed 2000 \
--bit $bit

bit=48
python train.py \
--batch-size 32 \
--learning_rate 1e-3 \
--seed 2000 \
--bit $bit
