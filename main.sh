# split the dataset directory based on number of SG* and copy from dataset2 dir to split dir (test_*_dataset2)
# 400 is decided based on #SG

for i in {1..400}; do mkdir test_${i}_dataset2; cp dataset2/SG${i}_* test_${i}_dataset2/; done

# create a script file
# run_dataset2_eczema.sh

for i in {1..400}; do echo "python3.7 bioProject.py markfolder /home/mona/Desktop/Localization/dataset/test_${i}_dataset2/ &" >> run_dataset2_eczema.sh; done

# run the run script
bash run_dataset2_eczema.sh

