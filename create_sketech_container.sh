docker run -t -i --name sketch --gpus all -v /home/viplab/:/data -p 7003:7003 -p 7004:7004 --shm-size 120G sketech_gen /bin/bash
# python -m visdom.server -port 7003