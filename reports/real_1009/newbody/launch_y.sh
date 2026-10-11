#!/bin/bash
cd $(dirname "$0"); until grep -q "NB_TRAIN_DONE seed 1" logs/train_chain_w3s3dnbjw_s1.out; do sleep 10; done
NB_TASK=W3S3dNBjY NB_PARENT=2026-10-11_05-09-59_w3s3dnbje_s1_from_17798 NB_PC=18997 NB_IT=1000 NB_NAME=w3s3dnbjy NB_ENVS=2048 NB_MEM=12000 ./train_task.sh 1 > logs/train_chain_w3s3dnbjy_s1.out 2>&1
