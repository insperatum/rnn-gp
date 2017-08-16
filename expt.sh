#!/bin/bash
exptname=`date +%Y%m%d_%H%M%S`
folder=experiments/$exptname
mkdir $folder

# for evoIter in 500 20; do

# layers=0


i=0
novelty=0
# for novelty in 0 0.1; do

for v in 1 2; do
for layers in 0; do
for fitness in solved; do
sgdIter=0
for selection in "tournament(sampleInner=False,select='single',proportion=0.2)" \
				 "tournament(sampleInner=False,select='single',proportion=0.5)" \
				 "tournament(sampleInner=False,select='multiple',proportion=0.2)" \
				 "tournament(sampleInner=False,select='multiple',proportion=0.5)" \
				 "tournament(sampleInner=True,select='single',proportion=1)"; do
# for sgdIter in 1 5; do
		# sbatch -n100 --time=180 ./main.sh --layers=$layers --sgdIter=$sgdIter --novelty $novelty
		# sbatch -n100 --time=180 --output="$folder/slurm-%j.out" ./main.sh --layers=$layers --sgdIter=$sgdIter --novelty $novelty --meta
		# sbatch -n100 --time=180 --output="$folder/slurm-%j.out" ./main.sh --layers=$layers --sgdIter=$sgdIter --novelty $novelty --meta --variational
		# sbatch -n100 --time=180 ./main.sh --layers=$layers --sgdIter=$sgdIter --novelty $novelty --deterministic
		# i=$((i+1))
		# sbatch -n100 --qos=use-everything --time=180 --output="$folder/$i.out" ./main.sh --layers=$layers --fitness=$fitness --sgdIter=$sgdIter --selection="$selection" --novelty=$novelty --deterministic --meta --name="$exptname/$i"
		i=$((i+1))
		sbatch -n100 --qos=use-everything --time=180 --output="$folder/$i.out" ./main.sh --layers=$layers --fitness=$fitness --sgdIter=$sgdIter --selection="$selection" --novelty=$novelty --meta --name="$exptname/$i"
		# i=$((i+1))
		# sbatch -n70 --time=120 --output="$folder/$i.out" ./main.sh --layers=$layers --fitness=$fitness --sgdIter=$sgdIter --selection="$selection" --novelty=$novelty --deterministic --meta --name="$exptname/$i" --normalise

# done
done
done
# done
done
done