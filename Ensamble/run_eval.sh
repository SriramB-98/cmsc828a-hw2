


python run_ensemble_no_trainer.py --model_name_or_path microsoft/resnet-18 --output_dir ./output_run1 --with_tracking >> out_train_terminal

cp -r ./output_run1 ./output_run2

python run_ensemble_no_trainer.py --model_name_or_path microsoft/resnet-18 --output_dir ./output_run2 --eval_only True --with_tracking >> out_eval_terminal