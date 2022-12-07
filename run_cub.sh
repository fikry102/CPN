time1=$(date)
cd scripts
echo "----------START----------"
python 01_CUB_stage1.py
wait
python 01_CUB_stage2_5W1S.py
wait
python 01_CUB_stage2_5W5S.py
wait
echo "----------END----------"
time2=$(date)
echo $time1 "--->" $time2