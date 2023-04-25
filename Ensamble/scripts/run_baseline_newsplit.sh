

# Random Split
CUDA_VISIBLE_DEVICES="0,1" python run_ensemble_no_trainer.py --model_name_or_path microsoft/resnet-18 --output_dir ./output_run_newsplit --with_tracking >> out_terminal_newsplit

# Avoid Shared Classes to fall in same Split
CUDA_VISIBLE_DEVICES="0,1" python run_ensemble_split_superclass.py --model_name_or_path microsoft/resnet-18 --output_dir ./output_run_superclass_newsplit --with_tracking >> out_terminal_superclasssplits