import os
import sys
import random
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import yaml
import pickle
import math
import time
import datetime

from dataloader_malscan import ExemplarIncrementalDataset, PureExemplarDataset, MultiExemplarDataset
from exemplar import gen_fixed_random_step_exemplar_set
from trainer import LwFTrainer, FinetuneTrainer, iCaRLTrainer, AFCTrainer, SSILTrainer, RandomTrainer, TIMLTrainer
from metrics import calculate_forgetting_score

trainer_dic = {'lwf': LwFTrainer, 'fine_tune': FinetuneTrainer, 'icarl': iCaRLTrainer,
               'afc': AFCTrainer, 'ssil': SSILTrainer, 'random': RandomTrainer, 'timl': TIMLTrainer}
exemplar_strategy_dic = {'random': gen_fixed_random_step_exemplar_set}

class Logger(object):
    def __init__(self, log_dir):
        self.terminal = sys.stdout
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = open(os.path.join(log_dir, f"training_logs_{current_time}.txt"), "w")

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log_file.write(message)
            self.log_file.flush()
        except Exception as e:
            self.terminal.write(f"Error writing to log file: {e}\n")

    def flush(self):  # Needed for compatibility with Python 3
        pass

    def close(self):  # Close the log file when done
        self.log_file.close()

class TIMLTrainer:
    def __init__(self, config, model, train_data, val_data, families_global_indices, hash_type, log_dir):
        self.config = config
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.families_global_indices = families_global_indices
        self.log_dir = log_dir
        self.hash_type = hash_type

        self.eval_res_dic = {"task_acc_in_each_step": [], "forgetting_score_in_each_step": [],
                             "avg_acc_known_classes": [], "weighted_avg_acc_known_classes": [],
                             "task_best_acc_list": [], "overall_weighted_avg_acc_steps_so_far": [],
                             "training_time_cost": [], "total_training_sample_number": [], 
                             "exemplar_sample_number": [], "family_accs_in_each_step": [],
                             "overall_accuracy": [], "overall_acc_so_far": []}
        
        # self.writer = SummaryWriter(log_dir=log_dir)

    def train(self, num_steps):

        # Redirect standard output to a log file
        # sys.stdout = open(os.path.join(self.log_dir, "training_logs.txt"), "w")
        sys.stdout = Logger(self.log_dir)
        
        mapping_dict = {} #to ensure labels in the training are ordinal  
        inverse_mapping_dict = {} #to retrieve original labels back
        
        incremental_nbr_new_classes = [0] #we assume that before step 0, we had 0 families

        train_dataset = ExemplarIncrementalDataset(self.config, self.train_data, self.config['root_data_path'], families_global_indices, 
                                                   "train", self.hash_type, self.config['mode'], self.config['img_norml'])
        val_dataset   = ExemplarIncrementalDataset(self.config, self.val_data, self.config['root_data_path'], families_global_indices, 
                                                   "test", self.hash_type, "both", self.config['img_norml'])

        if self.config['il_trainer'] == 'ssil':
            if self.config['multi_exemplar']:
                exemplar_dataset = MultiExemplarDataset(self.config, self.train_data, self.config['root_data_path'], families_global_indices, 
                                                   "train", self.hash_type, self.config['mode'], self.config['img_norml'])
            else:
                exemplar_dataset = PureExemplarDataset(self.config, self.train_data, self.config['root_data_path'], families_global_indices, 
                                                   "train", self.hash_type, self.config['mode'], self.config['img_norml'])
        
        exemplars = None
        
        device = torch.device('cuda')
        
        for step in range(num_steps):

            print("="*10+f"step {step+1} started"+"="*10)

            start_time = time.time()

            train_dataset.set_incremental_step(step)
            val_dataset.set_incremental_step(step)
            if self.config['il_trainer'] == 'ssil':
                exemplar_dataset.set_incremental_step(step)
            
            indices_test_end_step = val_dataset.indices_test_end_step

            if int(self.config['exemplar_budget']):
                if self.config['il_trainer'] == 'ssil':
                    exemplar_dataset._update_exemplars(exemplars)
                    self.eval_res_dic["total_training_sample_number"].append(len(train_dataset) + len(exemplar_dataset))
                    self.eval_res_dic["exemplar_sample_number"].append(len(exemplar_dataset))
                else:
                    train_dataset._update_exemplars(exemplars)
                    self.eval_res_dic["total_training_sample_number"].append(len(train_dataset))
                    self.eval_res_dic["exemplar_sample_number"].append(len(train_dataset.exemplars))
            else:
                self.eval_res_dic["total_training_sample_number"].append(len(train_dataset))
                self.eval_res_dic["exemplar_sample_number"].append(0)

            train_dataloader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
            val_dataloader  = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)
            exemplar_dataloader = None
            if self.config['il_trainer'] == 'ssil' and step > 0:
                self.config['exemplar_bs'] = math.ceil(self.config['batch_size'] * len(exemplar_dataset) / len(train_dataset))  # try to align the length the two iterator
                exemplar_dataloader  = DataLoader(exemplar_dataset, 
                                                  batch_size=min(self.config['exemplar_bs'], exemplar_dataset.__len__()), 
                                                  shuffle=True, num_workers=4)
            
            training_classes_current_step = train_dataset.training_classes_current_step #global labels of new families in train
            
            new_classes = [] #to save mapped labels from mapping_dict of new families 
            
            for i in range(len(training_classes_current_step)):
                if training_classes_current_step[i] not in mapping_dict:
                    mapping_dict[training_classes_current_step[i]] = len(mapping_dict)
                    inverse_mapping_dict[len(mapping_dict)-1] = training_classes_current_step[i]
                    new_classes.append(mapping_dict[training_classes_current_step[i]])
                    
            print("new_classes", new_classes)        
            incremental_nbr_new_classes.append(len(new_classes)+incremental_nbr_new_classes[-1])
            
            if not self.config['LSC'] and not (self.config['il_trainer'] == 'fine_tune' and self.config['shift_eval']):
                self.model.incremental_classifier(len(mapping_dict))

            trainer = trainer_dic[self.config['il_trainer']](config, step, self.model, device, self.log_dir)
            if os.path.exists(f"{trainer.save_path}/model_step_{step}.pkl"):
                print("*"*5+f"model_step_{step} exists! Evaluate it directly!"+"*"*5)
                trainer.model = torch.load(f"{trainer.save_path}/model_step_{step}.pkl")  
                trainer.model.to(device)
                self.model = trainer.model
            else:
                trainer.train(train_dataloader, exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes)

            if int(self.config['exemplar_budget']):
                if self.config['il_trainer'] == 'ssil':
                    exemplars = exemplar_strategy_dic[self.config['exemplar_strategy']](step, 
                                                                                        self.config['exemplar_budget'],
                                                                                        exemplar_dataset, exemplars)
                else:
                    exemplars = exemplar_strategy_dic[self.config['exemplar_strategy']](step, 
                                                                                        self.config['exemplar_budget'],
                                                                                        train_dataset, exemplars)
                # print('exemplar size: ', len(exemplars))
            
            end_time = time.time()
            elapsed_time = (end_time - start_time) / 60  # Convert to minutes
            self.eval_res_dic["training_time_cost"].append(elapsed_time)
            
            self.evaluate(trainer, val_dataloader, mapping_dict, inverse_mapping_dict, 
                          step, incremental_nbr_new_classes, indices_test_end_step)

            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"{self.log_dir}/evaluation_results_{step}_{current_time}.json", "w") as f:
                    json.dump(self.eval_res_dic, f, indent=4)

        # Close log file and reset standard output
        sys.stdout.close()
        sys.stdout = sys.__stdout__
    
    def evaluate(self, trainer, val_dataloader, mapping_dict, inverse_mapping_dict, step, incremental_nbr_new_classes, indices_test_end_step):

        def select_samples_of_first_step_families(images, labels, indices, incremental_nbr_new_classes):

            selected_indices = torch.where(labels < incremental_nbr_new_classes[1])[0]
            
            return images[selected_indices], labels[selected_indices], indices[selected_indices]

        def select_samples_of_only_cur_step(images, labels, indices, indices_test_end_step, step):

            selected_indices = np.where(indices >= indices_test_end_step[step-1])[0]
                    
            return images[selected_indices], labels[selected_indices]

        self.model.eval()
        val_total_samples = 0
        
        val_correct_samples = {key: 0 for key in mapping_dict}
        val_class_counts = {key: 0 for key in mapping_dict}
        
        val_preds_list = []
        val_labels_list = []
        val_init_samp_indices_list = []

        with torch.no_grad():
            for i, (images, labels, indices) in enumerate(val_dataloader):

                if self.config['il_trainer'] == 'fine_tune' and self.config['shift_eval']:
                    images, labels, indices = select_samples_of_first_step_families(images, labels, indices, incremental_nbr_new_classes)
                    if step > 0:
                        images, labels = select_samples_of_only_cur_step(images, labels, indices, indices_test_end_step, step)
                    if not len(images):
                        continue
                
                preds = trainer.inference(images, inverse_mapping_dict)

                val_total_samples += images.size(0)
                val_preds_list.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_init_samp_indices_list.extend(indices.cpu().numpy())

                for j in range(labels.size(0)):
                    label = labels[j].item()
                    if label in mapping_dict:
                        val_class_counts[label] += 1
                        if preds[j] == label:
                            val_correct_samples[label] += 1
                            
        overall_accuracy = len([i for i in range(len(labels)) if labels[i] == preds[i]])/len(labels) if len(labels) else np.nan
        
        step_accuracies = {}
        for k in mapping_dict:
            if val_class_counts[k] != 0:
                accuracy = val_correct_samples[k] / val_class_counts[k]
            else:
                # import ipdb; ipdb.set_trace()
                accuracy = 0
            step_accuracies[k] = accuracy
            
        weights = [val_class_counts[k] for k in mapping_dict]
        
        self.eval_res_dic['family_accs_in_each_step'].append(step_accuracies)
        avg_class_acc = np.average(list(step_accuracies.values()))
        weighted_avg_class_acc = np.average(list(step_accuracies.values()), weights=weights)
        
        
        preds_mapped = [mapping_dict[i] for i in val_preds_list]
        labels_mapped = [mapping_dict[i] if i in mapping_dict else -1 for i in val_labels_list]
        

        if self.config['il_trainer'] == 'fine_tune' and self.config['shift_eval']: 
            forgetting_score, task_acc_list = 0, self.eval_res_dic["task_best_acc_list"]
        else:
            forgetting_score, task_acc_list = calculate_forgetting_score(step, preds_mapped, labels_mapped,
                                                                        self.eval_res_dic["task_best_acc_list"],
                                                                        incremental_nbr_new_classes,
                                                                        val_init_samp_indices_list, indices_test_end_step)
        if forgetting_score is not None:
            self.eval_res_dic["forgetting_score_in_each_step"].append(forgetting_score)
        
        self.eval_res_dic["task_acc_in_each_step"].append(task_acc_list)
        self.eval_res_dic["avg_acc_known_classes"].append(avg_class_acc)
        self.eval_res_dic["weighted_avg_acc_known_classes"].append(weighted_avg_class_acc)
        self.eval_res_dic["overall_accuracy"].append(overall_accuracy)
        # self.eval_res_dic["overall_acc_so_far"].append(np.average(self.eval_res_dic["overall_accuracy"]))
        self.eval_res_dic["overall_acc_so_far"].append(np.nanmean(self.eval_res_dic["overall_accuracy"]))
        
        self.eval_res_dic["overall_weighted_avg_acc_steps_so_far"].append(np.average(self.eval_res_dic["weighted_avg_acc_known_classes"]))
        #self.eval_res_dic["task_best_acc_list"].append(task_best_acc_list)
        
        print(f"Average accuracy known classes: {avg_class_acc}")
        print(f"Weighted average accuracy known classes: {weighted_avg_class_acc}")

        print("Overall weighted average accuracy over all history steps so far: ", 
              np.average(self.eval_res_dic["weighted_avg_acc_known_classes"]))
        print("Forgetting_score_in_step", forgetting_score)
        print("Overall average forgetting over all history steps so far: ", 
              np.average(self.eval_res_dic["forgetting_score_in_each_step"]))
        
def set_random_seeds(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment Settings")
    parser.add_argument('--exp_setting', type=str, required=True, help='Path to the YAML configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



if __name__ == "__main__":

    import os
    from model import CILNet, MalscanTIML
    import json
    import shutil
    import glob
    from pathlib import Path

    args = parse_args()
    config_path = args.exp_setting
    config = load_config(config_path)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(config['gpu_id'])

    seed = config['random_seed']
    set_random_seeds(seed)

    with open(config['train_set_json'], 'r') as f:
        train_data = json.load(f)
          
    with open(config['test_set_json'], 'r') as f:
        val_data = json.load(f)

    with open(config['hash_type_dict'], 'rb') as f:
        hash_type = pickle.load(f)

    init_num_classes = len(train_data['step=0'])  # The initial number of classes
    num_steps = len(train_data)
    
    log_dir = f'logs/{config["exp_tag"]}_seed{seed}_init-num{init_num_classes}_bs{config["batch_size"]}_lr{config["learning_rate"]}_step{num_steps}_epoch{config["num_epochs"]}_res18weights-{config["res18_weights"]}'

    if config['img_norml']:
        log_dir += '_w_img_nomrl'

    # Make sure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    shutil.copy(config_path, log_dir)

    # Copy all .py files from the current directory to the destination directory
    for file_path in Path('.').glob('*.py'):
        shutil.copy(file_path, log_dir)

    # Initialize the model
    if config["timl_backbone"]:
        model = MalscanTIML() 
    else:
        model = CILNet(init_num_classes, config['res18_weights'], config['LSC'])

    # Initialize the trainer
    families_global_indices = {}

    trainer = TIMLTrainer(config, model, train_data, val_data, families_global_indices, hash_type, log_dir)

    # Train the model
    trainer.train(num_steps)

    with open(f"{log_dir}/evaluation_results.json", "w") as f:
            json.dump(trainer.eval_res_dic, f, indent=4)

    with open(f"{log_dir}/families_global_indices.json", "w") as f:
            json.dump(families_global_indices, f, indent=4)
