export CUDA_VISIBLE_DEVICES=2

python3 synthesize.py --source english_inference/inputs.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
