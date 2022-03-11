# sample trajectories (data/task/sample_paths.json)
python dataset_process/generate_trajectory.py

# generate templates (data/task/R2R_train_templates.tsv)
python dataset_process/extract_noun_phrases.py --source data/task/R2R_train.json --output data/task/R2R_train_noun_phrases.tsv
python dataset_process/gen_templates.py --source data/task/R2R_train_noun_phrases.tsv --output data/task/R2R_train_templates.tsv

# detect rooms and objects by CLIP (data/task/sample_path_predict_noun_phases.json)
ln -s data/connectivity connectivity
python dataset_process/generate_caption_clip.py

# generate directions for sampled paths (data/task/sample_path_directions.json)
python dataset_process/generate_directions.py

# generate instructions for sampled paths (data/task/sample_data.json)
python dataset_process/gen_instructions.py --sample_path data/task/sample_paths.json --template data/task/R2R_train_templates.tsv --caption data/task/sample_path_predict_noun_phases.json --direction data/task/sample_path_directions.json --output data/task/sample_data.json
