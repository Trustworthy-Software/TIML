import os
import copy
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from datetime import datetime
from tqdm import tqdm
from layers import SplitLSCLinear

def map_labels(original_labels, dict_):
    """Map the 'general' labels of families in training to new labels from dict_. 
       The dictionary is created based only on labels from training. 
       This is important in order to have all the known labels in training incremental (i.e., O, 1, 2, 3, ...)"""
    return [dict_[i.item()] for i in original_labels]

def retrieve_general_labels_back(preds, dict_):
    """Get the general labels back from the predictions so we can compare against the labels in the test"""
    return [dict_[i.item()] for i in preds]

class RandomTrainer():
    def __init__(self, config, step, model, device, log_dir):

        self.step = step
    
    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):

        print('-'*10+'step: ', self.step+1, '-'*10)

    def inference(self, images, inverse_mapping_dict):

        preds = np.random.randint(0, len(inverse_mapping_dict), size=len(images))
        preds = torch.tensor(retrieve_general_labels_back(preds, inverse_mapping_dict))

        return preds
    
class LwFTrainer():
    def __init__(self, config, step, model, device, log_dir):
        self.config = config
        self.step = step
        self.model = model
        self.device = device
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'saved_models')
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):
        
        self.model.to(self.device)
        self.model.train()

        if self.step > 0:
            model_last_step = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            model_last_step.to(self.device)

        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            total_samples = 0

            for i, (images, labels, _) in enumerate(train_dataloader):
                labels = torch.tensor(map_labels(labels, mapping_dict))
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(images)
            
                if self.step == 0:
                    loss = F.cross_entropy(logits, labels)
                else:
                    with torch.no_grad():
                        logits_last_step, _ = model_last_step(images)
                    
                    loss_KD = torch.zeros(self.step).to(self.device)
                    
                    for t in range(self.step):
                        start = incremental_nbr_new_classes[t] ### DOUBLE CHECK 
                        end = incremental_nbr_new_classes[t+1]

                        soft_target = F.softmax(logits_last_step[:, start:end] / self.config['T'], dim=1)
                        output_log = F.log_softmax(logits[:, start:end] / self.config['T'], dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.config['T']**2)
                    
                    loss_KD = loss_KD.sum()

                    if not self.config['adaptation']:
                        selected_indices = torch.where(labels >= incremental_nbr_new_classes[-2])[0]
                        logits = logits[selected_indices]
                        labels = labels[selected_indices]
                        logits = logits[:, incremental_nbr_new_classes[-2]:]
                        labels = labels % incremental_nbr_new_classes[-2]
                    CE_loss = F.cross_entropy(logits, labels)  
        
                    loss = self.config['lamda']*loss_KD + CE_loss
            
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Log the iteration loss
                # self.writer.add_scalar('global_iteration_loss/step_{}'.format(step), loss.item(), global_iter)
                # global_iter += 1

            epoch_loss = running_loss / total_samples ### DOUBLE CHECK if correct
            # self.writer.add_scalar('loss/step_{}'.format(step), epoch_loss, epoch)

            now = datetime.now()
            current_time = now.strftime("%y/%m/%d %H:%M:%S")
            print(f"Step: {self.step+1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}") 

        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)
                    
    
    def inference(self, images, inverse_mapping_dict):
        
        self.model.eval()

        images = images.to(self.device) 
        logits, _ = self.model(images)

        preds = torch.argmax(logits, dim=1)
        preds = torch.tensor(retrieve_general_labels_back(preds.cpu().numpy(), inverse_mapping_dict))
    
        return preds

class FinetuneTrainer(LwFTrainer):
    def __init__(self, config, step, model, device, log_dir):
        super().__init__(config, step, model, device, log_dir)

    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):
        
        self.model.to(self.device)
        self.model.train()

        if self.step == 0 or (self.step > 0 and not self.config['shift_eval']):
            for epoch in range(self.config['num_epochs']):
                running_loss = 0.0
                total_samples = 0

                for i, (images, labels, _) in enumerate(train_dataloader):
                    labels = torch.tensor(map_labels(labels, mapping_dict))
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    logits, _ = self.model(images)
                    
                    loss = F.cross_entropy(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    total_samples += images.size(0)

                epoch_loss = running_loss / total_samples
                # self.writer.add_scalar('loss/step_{}'.format(step), epoch_loss, epoch)

                now = datetime.now()
                current_time = now.strftime("%y/%m/%d %H:%M:%S")
                print(f"Step: {self.step + 1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}")
        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)

class iCaRLTrainer(LwFTrainer):
    def __init__(self, config, step, model, device, log_dir):
        super().__init__(config, step, model, device, log_dir)

    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):
        
        self.model.to(self.device)
        self.model.train()

        if self.step > 0:
            model_last_step = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            model_last_step.to(self.device)

        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            total_samples = 0

            for i, (images, labels, _) in enumerate(train_dataloader):
                labels = torch.tensor(map_labels(labels, mapping_dict))
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(images)
            
                if self.step == 0:
                    loss = F.cross_entropy(logits, labels)
                else:
                    with torch.no_grad():
                        logits_last_step, _ = model_last_step(images)
                    
                    old_target = torch.sigmoid(logits_last_step)
                    old_task_size = old_target.shape[1]
                    loss_KD = F.binary_cross_entropy_with_logits(torch.sigmoid(logits[..., :old_task_size]), old_target) 

                    if not self.config['adaptation']:
                        selected_indices = torch.where(labels >= incremental_nbr_new_classes[-2])[0]
                        logits = logits[selected_indices]
                        labels = labels[selected_indices]
                    CE_loss = F.cross_entropy(logits, labels)  
        
                    loss = self.config['lamda']*loss_KD + CE_loss
            
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Log the iteration loss
                # self.writer.add_scalar('global_iteration_loss/step_{}'.format(step), loss.item(), global_iter)
                # global_iter += 1

            epoch_loss = running_loss / total_samples ### DOUBLE CHECK if correct
            # self.writer.add_scalar('loss/step_{}'.format(step), epoch_loss, epoch)

            now = datetime.now()
            current_time = now.strftime("%y/%m/%d %H:%M:%S")
            print(f"Step: {self.step+1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}") 

        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)

class AFCTrainer(LwFTrainer):
    def __init__(self, config, step, model, device, log_dir):
        super().__init__(config, step, model, device, log_dir)

    def imprint_weights(self, model, train_loader, new_classes, mapping_dict):
        
        model.eval()
        
        class_num_cur_step = len(new_classes)
        start = new_classes[0]

        features_arr = []
        targets_arr = []
        
        with torch.no_grad():
            print('===========================')
            print('Imprint weights')
            print('===========================')
            model = model.to(self.device)

            for images, labels, _ in tqdm(train_loader):
                labels = torch.tensor(map_labels(labels, mapping_dict))
                images = images.to(self.device)
                _, features = model(images)
                features = F.normalize(features)
                
                features = features.detach().cpu()
                
                new_idx = labels >= start

                new_features = features[new_idx]
                new_target_idx = labels[new_idx] - start
                
                features_arr.append(new_features)
                targets_arr.append(new_target_idx)
            
            model = model.to('cpu')
            weights_norm = model.classifier.fc1.weight.data.norm(dim=1, keepdim=True)
            avg_weights_norm = torch.mean(weights_norm, dim=0)

            features = torch.cat(features_arr, dim=0)
            targets = torch.cat(targets_arr, dim=0)
            
            new_weights = []
            for c in range(class_num_cur_step):
                class_features = features[targets == c]
                clusterizer = KMeans(n_clusters=10)
                clusterizer.fit(class_features.numpy())
                
                for center in clusterizer.cluster_centers_:
                    new_weights.append(torch.tensor(center) * avg_weights_norm)
            
            new_weights = torch.stack(new_weights)
            model.classifier.fc2.weight.data = new_weights
        
        return model

    def nca(self, similarities, targets, scale=1, margin=0.6):
        '''
        Neighborhood Component Analysis (NCA) is a distance metric learning method that aims to improve the accuracy of nearest neighbor classification.
        '''
        margins = torch.zeros_like(similarities)
        margins[torch.arange(margins.shape[0]), targets] = margin
        
        similarities = scale * (similarities - margin)

        similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

        disable_pos = torch.zeros_like(similarities)
        disable_pos[torch.arange(len(similarities)),
                    targets] = similarities[torch.arange(len(similarities)), targets]

        numerator = similarities[torch.arange(similarities.shape[0]), targets]
        denominator = similarities - disable_pos

        losses = numerator - torch.log(torch.exp(denominator).sum(-1))
        
        losses = -losses
        loss = torch.mean(losses)
        
        return loss

    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):
        
        self.model.to(self.device)
        self.model.train()

        if self.step == 1:
            model = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            in_features = model.classifier.in_features
            out_features = model.classifier.out_features

            new_classifier = SplitLSCLinear(in_features, out_features, len(new_classes))
            new_classifier.fc1.weight.data = model.classifier.weight.data
            model.classifier = new_classifier

            old_model = copy.deepcopy(model)
            self.model = model

        elif self.step > 1:
            model = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            in_features = model.classifier.in_features

            out_features1 = model.classifier.fc1.out_features
            out_features2 = model.classifier.fc2.out_features
            K = model.classifier.fc2.K

            new_fc = SplitLSCLinear(in_features, out_features1 + out_features2, len(new_classes))

            new_fc.fc1.weight.data[:out_features1*K] = model.classifier.fc1.weight.data
            new_fc.fc1.weight.data[out_features1*K:] = model.classifier.fc2.weight.data
            model.classifier = new_fc

            old_model = copy.deepcopy(model)
            self.model = model
            
        if self.step > 0:
            self.model = self.imprint_weights(self.model, train_dataloader, new_classes, mapping_dict)

        self.model = self.model.to(self.device)
        if self.step > 0:
            old_model = old_model.to(self.device)
            old_model.eval()
        
        if self.step > 0:
            ignored_params = list(map(id, self.model.classifier.fc1.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
            params =[{'params': base_params, 'lr': float(self.config['learning_rate']), 'weight_decay': float(self.config['weight_decay'])}, \
                    {'params': self.model.classifier.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
        else:
            params = self.model.parameters()
        self.optimizer = torch.optim.Adam(params, lr=float(self.config['learning_rate']), weight_decay=float(self.config['weight_decay']))

        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            total_samples = 0

            for i, (images, labels, _) in enumerate(train_dataloader):
                labels = torch.tensor(map_labels(labels, mapping_dict))
                images, labels = images.to(self.device), labels.to(self.device)
                logits, feature = self.model(images, AFC_train_out=True)
                if self.step > 0:
                    old_logits, old_feature = old_model(images, AFC_train_out=True)
                    
                    prev_loss_cls = self.nca(old_logits, labels, scale=old_model.classifier.factor)
                    prev_loss_cls.backward(retain_graph=True)
                    
                    loss_disc = torch.mean(torch.norm(F.normalize(feature) - F.normalize(old_feature), p='fro', dim=-1))
                    
                    loss_cls = self.nca(logits, labels, scale=self.model.classifier.factor)
                    lam_t = (self.step + 1) ** 0.5
                    lam_disc = self.config['lam_disc']
                    loss = loss_cls + lam_disc * lam_t * loss_disc
                else:
                    loss = self.nca(logits, labels, scale=self.model.classifier.factor)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Log the iteration loss
                # self.writer.add_scalar('global_iteration_loss/step_{}'.format(step), loss.item(), global_iter)
                # global_iter += 1

            epoch_loss = running_loss / total_samples #
            now = datetime.now()
            current_time = now.strftime("%y/%m/%d %H:%M:%S")
            print(f"Step: {self.step+1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}") 

        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)

class SSILTrainer(LwFTrainer):
    def __init__(self, config, step, model, device, log_dir):
        super().__init__(config, step, model, device, log_dir)

    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):

        self.model.to(self.device)
        self.model.train()

        if self.step > 0:
            model_last_step = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            model_last_step.to(self.device)

        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            total_samples = 0

            if self.step == 0:
                iterator = tqdm(train_dataloader)
                pbar = iterator
            else:
                iterator = zip(train_dataloader, ssil_exemplar_dataloader)
                pbar = tqdm(iterator, total=min(len(train_dataloader), len(ssil_exemplar_dataloader)))
            
            for samples in pbar:
                if self.step == 0:
                    images, labels, _ = samples
                    labels = torch.tensor(map_labels(labels, mapping_dict))
                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    logits, _ = self.model(images)
                    loss = F.cross_entropy(logits, labels)
                else:
                    curr, prev = samples
                    images, labels, _ = curr
                    labels = torch.tensor(map_labels(labels, mapping_dict))
                    if self.config['multi_exemplar']:
                        labels = labels % incremental_nbr_new_classes[-2]  # may do not need this as normal data in current step still contains samples from old classes
                    labels = labels.to(self.device)

                    exemplar_images, exemplar_labels, _ = prev
                    exemplar_labels = torch.tensor(map_labels(exemplar_labels, mapping_dict))
                    exemplar_labels = exemplar_labels.to(self.device)

                    data_batch_size = labels.shape[0]
                    exemplar_data_batch_size = exemplar_labels.shape[0]

                    total_images = torch.cat((images, exemplar_images))
                    total_images = total_images.to(self.device)
                    logits, _ = self.model(total_images)
                    
                    with torch.no_grad():
                        logits_last_step, _ = model_last_step(total_images)
                    
                    logits_last_step = logits_last_step[:, :incremental_nbr_new_classes[-2]]
                    if self.config['multi_exemplar']:
                        logits_cur = logits[:data_batch_size, incremental_nbr_new_classes[-2]:]
                    else:
                        logits_cur = logits[:data_batch_size, :]  # normal data in current step still contains samples from old classes

                    loss_curr = F.cross_entropy(logits_cur, labels)

                    logits_prev = logits[data_batch_size:data_batch_size+exemplar_data_batch_size, :incremental_nbr_new_classes[-2]]
                    loss_prev = F.cross_entropy(logits_prev, exemplar_labels)

                    loss_CE = (loss_curr * data_batch_size + loss_prev * exemplar_data_batch_size) / (data_batch_size + exemplar_data_batch_size)

                    loss_KD = torch.zeros(self.step).to(self.device)
                    
                    for t in range(self.step):
                        start = incremental_nbr_new_classes[t]
                        end = incremental_nbr_new_classes[t+1]

                        soft_target = F.softmax(logits_last_step[:, start:end] / self.config['T'], dim=1)
                        output_log = F.log_softmax(logits[:, start:end] / self.config['T'], dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.config['T']**2)

                    loss_KD = loss_KD.sum()
                    loss = loss_CE + self.config['lamda']*loss_KD
            
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # Log the iteration loss
                # self.writer.add_scalar('global_iteration_loss/step_{}'.format(step), loss.item(), global_iter)
                # global_iter += 1

            epoch_loss = running_loss / total_samples ### DOUBLE CHECK if correct
            # self.writer.add_scalar('loss/step_{}'.format(step), epoch_loss, epoch)

            now = datetime.now()
            current_time = now.strftime("%y/%m/%d %H:%M:%S")
            print(f"Step: {self.step+1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}") 

        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)


class TIMLTrainer(LwFTrainer):
    def __init__(self, config, step, model, device, log_dir):
        super().__init__(config, step, model, device, log_dir)

        self.mse_loss = torch.nn.MSELoss()

    def train(self, train_dataloader, ssil_exemplar_dataloader, mapping_dict, incremental_nbr_new_classes, num_steps, new_classes=[]):

        self.model.to(self.device)
        self.model.train()

        if self.step > 0:
            model_last_step = torch.load(f"{self.save_path}/model_step_{self.step - 1}.pkl")
            model_last_step.to(self.device)
            model_last_step.eval()

        for epoch in range(self.config['num_epochs']):
            running_loss = 0.0
            total_samples = 0

            for i, (malscan_images, labels, _) in enumerate(train_dataloader):
                labels = torch.tensor(map_labels(labels, mapping_dict))
                malscan_images, labels = malscan_images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logits, malscan_features = self.model(vector_input=malscan_images)
            
                if self.step == 0:
                    loss = F.cross_entropy(logits, labels)
                else:
                    with torch.no_grad():
                        logits_last_step, malscan_features_last_step = \
                            model_last_step(vector_input=malscan_images)
                    
                    loss_KD = torch.zeros(self.step).to(self.device)
                    
                    for t in range(self.step):
                        start = incremental_nbr_new_classes[t] ### DOUBLE CHECK 
                        end = incremental_nbr_new_classes[t+1]

                        soft_target = F.softmax(logits_last_step[:, start:end] / self.config['T'], dim=1)
                        output_log = F.log_softmax(logits[:, start:end] / self.config['T'], dim=1)
                        loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.config['T']**2)
                    
                    loss_KD = loss_KD.sum()

                    loss = F.cross_entropy(logits, labels) + self.config['lamda']*loss_KD


                    if self.config['feat_dist_loss']:
                        malscan_dist_loss = self.mse_loss(malscan_features, malscan_features_last_step.detach())
                        loss += self.config['lamda_feat_malscan']*malscan_dist_loss

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * malscan_images.size(0)
                total_samples += malscan_images.size(0)

            epoch_loss = running_loss / total_samples ### DOUBLE CHECK if correct
            # self.writer.add_scalar('loss/step_{}'.format(step), epoch_loss, epoch)

            now = datetime.now()
            current_time = now.strftime("%y/%m/%d %H:%M:%S")
            print(f"Step: {self.step+1}/{num_steps}, Epoch: {epoch + 1}/{self.config['num_epochs']}, Loss: {epoch_loss:.4f}, time: {current_time}") 

        # Save the model at each step
        torch.save(self.model, f"{self.save_path}/model_step_{self.step}.pkl")
        print("-"*40)
                    
    
    def inference(self, malscan_images, inverse_mapping_dict):
        
        self.model.eval()

        malscan_images = malscan_images.to(self.device) 
        logits, _ = self.model(malscan_images)

        preds = torch.argmax(logits, dim=1)
        preds = torch.tensor(retrieve_general_labels_back(preds.cpu().numpy(), inverse_mapping_dict))
    
        return preds