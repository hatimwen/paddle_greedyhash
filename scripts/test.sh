bit=12
python eval.py \
--batch-size 32 \
--seed 2000 \
--bit $bit

bit=24
python eval.py \
--batch-size 32 \
--seed 2000 \
--bit $bit

bit=32
python eval.py \
--batch-size 32 \
--seed 2000 \
--bit $bit

bit=48
python eval.py \
--batch-size 32 \
--seed 2000 \
--bit $bit
