#!/bin/bash

source activate call_scoring

env_pt=$CONDA_PREFIX
#export LD_PRELOAD=${env_pt}'/lib/libiomp5.so'
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,proclist=[0-15],explicit

MSG="Production model CSR Ch LSTM+MHA on Sentence level."    
SAVE_DIR='logs/lstmha_prod/'
MODEL='all'
BS=4
EP=20
TRAIN_SAMPLES=0
ATTENTION='lstmha'
TRANSCRIPTS_PATH='/mnt/transcriber/Call_Scoring/transcriptions/csr_ch/train/'
TEST_TRANSCRIPTS_PATH='/mnt/transcriber/Call_Scoring/transcriptions/csr_ch/test/'
# TRANSCRIPTS_PATH2='/mnt/transcriber/Call_Scoring/transcriptions/customer_ch/train/'
# TEST_TRANSCRIPTS_PATH2='/mnt/transcriber/Call_Scoring/transcriptions/customer_ch/test/'
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

mkdir $SAVE_DIR
echo $MSG > $SAVE_DIR"log"
python MainCalls.py --save_path $SAVE_DIR --subscore "$MODEL" --batch_size $BS \
--epochs $EP --train_samples $TRAIN_SAMPLES --model $ATTENTION --trans_path $TRANSCRIPTS_PATH \
--loss $LOSS_FN --model_size $DIM --dropout $DP --word_nh $WORD_NH --sent_nh $SENT_NH --lr $LR \
--k $K --num_workers $NUM_WORKERS --workgroup $CALL_TYPE --num_layers $NUM_LAYERS \
--word_nlayers $WORD_NUM_LAYERS --reg $REG --acum_step $ACC_STEP --test_path $TEST_TRANSCRIPTS_PATH --new_data\
>> $SAVE_DIR"log" 2>&1 &
