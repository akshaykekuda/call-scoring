#!/bin/bash

source activate cs

env_pt=$CONDA_PREFIX
#export LD_PRELOAD=${env_pt}'/lib/libiomp5.so'
export OMP_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,proclist=[0-15],explicit

MSG="HS2AN with customer channel: csr only"    
#SAVE_DIR='logs/${}rmv_sen_att'
MODEL='best'
BS=4
EP=2
TRAIN_SAMPLES=0
ATTENTION='hs2cross'
TRANSCRIPTS_PATH='/mnt/transcriber/Call_Scoring/transcriptions/csr_ch/train/'
TEST_TRANSCRIPTS_PATH='/mnt/transcriber/Call_Scoring/transcriptions/csr_ch/test/'
TRANSCRIPTS_PATH2='/mnt/transcriber/Call_Scoring/transcriptions/customer_ch/train/'
TEST_TRANSCRIPTS_PATH2='/mnt/transcriber/Call_Scoring/transcriptions/customer_ch/test/'
LOSS_FN='cel'
DIM=256
DP=0.7
WORD_NH=4
SENT_NH=2
LR=1e-4
K=25
NUM_WORKERS=1
NUM_LAYERS=1  
WORD_NUM_LAYERS=1
CALL_TYPE='Sales'
REG=1e-3
ACC_STEP=1
SAVE_DIR="logs/${DIM}_${BS}_${EP}_with_sen_attn"
#mkdir $SAVE_DIR
#echo $MSG > $SAVE_DIR"log"
python MainCalls.py --save_path $SAVE_DIR --subscore "$MODEL" --batch_size $BS \
--epochs $EP --train_samples $TRAIN_SAMPLES --model $ATTENTION --trans_path $TRANSCRIPTS_PATH \
--loss $LOSS_FN --model_size $DIM --dropout $DP --word_nh $WORD_NH --sent_nh $SENT_NH --lr $LR \
--k $K --num_workers $NUM_WORKERS --workgroup $CALL_TYPE --num_layers $NUM_LAYERS \
--word_nlayers $WORD_NUM_LAYERS --reg $REG --acum_step $ACC_STEP --test_path $TEST_TRANSCRIPTS_PATH \
--trans2_path $TRANSCRIPTS_PATH2 --test2_path $TEST_TRANSCRIPTS_PATH2 \
# --new_data
# >> $SAVE_DIR"log" 2>&1 &
