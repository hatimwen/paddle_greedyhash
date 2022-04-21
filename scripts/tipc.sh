bit=12
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/greedyhash_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=24
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/greedyhash_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=32
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/greedyhash_$bit/train_infer_python.txt \
lite_train_lite_infer

bit=48
bash test_tipc/test_train_inference_python.sh \
test_tipc/configs/greedyhash_$bit/train_infer_python.txt \
lite_train_lite_infer
