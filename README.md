# torch_custom_op - trivial saxpy example

## To test

1. Make sure you have Docker and nvidia-docker installed.
2. Start container at the root of the repo:
```
docker run --rm -it --gpus all --ipc host nvcr.io/nvidia/pytorch:20.08-py3
```
3. When in the container:
```
git clone https://github.com/mkolod/torch_custom_op.git
cd torch_custom_op
python setup.py install
cd test
python test.py
```
