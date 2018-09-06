mpirun -n 3 --hostfile hosts_address \
python distributed_nn.py \
--lr=0.01 \
--lr-shrinkage=0.95 \
--momentum=0.0 \
--network=ResNet18 \
--dataset=Cifar10 \
--batch-size=128 \
--test-batch-size=200 \
--comm-type=Bcast \
--num-aggregate=2 \
--eval-freq=200 \
--epochs=10 \
--max-steps=1000000 \
--svd-rank=3 \
--quantization-level=4 \
--bucket-size=512 \
--code=sgd \
--enable-gpu= \
--train-dir=/home/ubuntu/