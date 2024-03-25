import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import itertools
    

class IncrementalDataset(Dataset):
    def __init__(self, config, data, root_path, step, families_global_indices, train_test, hash_type, mode='new', img_nomrl=True):
        self.config = config
        self.data = data
        self.step = step
        self.hash_type = hash_type
        self.families_global_indices = families_global_indices
        self.mode = mode
        self.train_test = train_test
        transform_list = [
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[9.28691073e-05, 1.65401002e-01, 6.80466368e-02], std=[0.00598309, 0.13839283, 0.14697313]),  
        ]
        if not img_nomrl:
            transform_list.pop(-1)
        self.transform = transforms.Compose(transform_list)
        self.indices_test_end_step = []
        self.new_cls_ids_in_current_step = []
        self.training_cls_number_each_step = []

        self.path_images = root_path
        self.update_data_indices()

    def update_data_indices(self):
        indices = []
        cls_id_list = []
        # training_classes_current_step = []
        
        if self.mode == 'both': 
            for step in range(self.step+1):
                for class_name in self.data[f'step={step}']:
                    self.families_global_indices.setdefault(class_name, len(self.families_global_indices)) #give the class an index if new
                    indices.extend([(os.path.join(self.path_images, self.hash_type[sha], class_name, sha+'.png'),
                                     self.families_global_indices[class_name]) for sha in self.data[f'step={step}'][class_name]])
                    cls_id_list.append(self.families_global_indices[class_name])
                
                if self.train_test=="test":
                    self.indices_test_end_step.append(len(indices))
                
            cls_id_list = list(set(cls_id_list))
                    
        elif self.mode == 'new':
            for class_name in self.data[f'step={self.step}']:
                self.families_global_indices.setdefault(class_name, len(self.families_global_indices))
                if self.families_global_indices[class_name] in self.cls_id_list_so_far: 
                    if self.config['il_trainer'] == 'ssil' and self.config['multi_exemplar'] and self.step > 0:
                        continue
                else:
                    self.new_cls_ids_in_current_step.append(self.families_global_indices[class_name])
                indices.extend([(os.path.join(self.path_images, self.hash_type[sha], class_name, sha+'.png'), 
                                 self.families_global_indices[class_name]) for sha in self.data[f'step={self.step}'][class_name]])
                cls_id_list.append(self.families_global_indices[class_name])
                
        self.cls_id_list_so_far.update(set(cls_id_list))
        if self.train_test=="train":
            self.training_classes_current_step = list(set(cls_id_list))
            self.training_cls_number_each_step.append(len(self.training_classes_current_step))
                
                    
        self.indices = indices
        # self.training_classes_current_step = training_classes_current_step
        
        # print(f"Initialized {self.train_test} dataset with {len(self.indices)} samples across {len(cls_id_list)} classes: {' '.join(map(str, cls_id_list))}.")
        print(f"Initialized {self.train_test} dataset with {len(self.indices)} samples across {len(cls_id_list)} classes.")

    def set_incremental_step(self, step):
        self.step = step
        self.new_cls_ids_in_current_step = []
        self.update_data_indices()

    def __getitem__(self, index):
        img_path, label = self.indices[index]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.indices)

class ExemplarIncrementalDataset(IncrementalDataset):
    def __init__(self, config, data, root_path, families_global_indices, train_test, hash_type, mode='new', img_nomrl=True):
        self.config = config
        self.data = data
        self.step = 0
        self.hash_type = hash_type
        self.families_global_indices = families_global_indices
        self.mode = mode
        self.train_test = train_test
        self.path_images = root_path
        transform_list = [
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[9.28691073e-05, 1.65401002e-01, 6.80466368e-02], std=[0.00598309, 0.13839283, 0.14697313]),  
        ]
        if not img_nomrl:
            transform_list.pop(-1)
        self.transform = transforms.Compose(transform_list)
        self.exemplars = []
        self.indices_test_end_step = []
        self.cls_id_list_so_far = set()
        self.training_cls_number_each_step = []

    def _update_exemplars(self, exemplars):
        if exemplars is None:
            return
        self.exemplars = list(itertools.chain(*exemplars))
        print('exemplar size: ', len(self.exemplars))
        self._conbine_exemplar()

    def _conbine_exemplar(self):
        self.indices += self.exemplars

class PureExemplarDataset(ExemplarIncrementalDataset):

    def __init__(self, config, data, root_path, families_global_indices, train_test, hash_type, mode='new', img_nomrl=True):
        super().__init__(config, data, root_path, families_global_indices, train_test, hash_type, mode, img_nomrl)

        # self.exemplars = []

    def __getitem__(self, index):
        index_new = index % len(self.exemplars)
        img_path, label = self.exemplars[index_new]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            img = self.transform(img)

        return img, label, index

    def __len__(self):
        return len(self.exemplars)

class MultiExemplarDataset(PureExemplarDataset):
    '''
    The exemplars come from two sources: 
        
        1) random selection from old samples of old families from previous steps
        
        2) all new samples of old families in current step
    '''
    
    def __init__(self, config, data, root_path, families_global_indices, train_test, hash_type, mode='new', img_nomrl=True):
        super().__init__(config, data, root_path, families_global_indices, train_test, hash_type, mode, img_nomrl)

        self.new_samps_of_old_families = []

    def update_data_indices(self):
        indices = []
        cls_id_list = []
        training_classes_current_step = []
        
        for class_name in self.data[f'step={self.step}']:

            self.families_global_indices.setdefault(class_name, len(self.families_global_indices)) 
            
            if self.step > 0 and self.families_global_indices[class_name] in self.cls_id_list_so_far:
                self.new_samps_of_old_families.extend([(os.path.join(self.path_images, self.hash_type[sha], class_name, sha+'.png'), 
                                 self.families_global_indices[class_name]) for sha in self.data[f'step={self.step}'][class_name]])
            else:
                indices.extend([(os.path.join(self.path_images, self.hash_type[sha], class_name, sha+'.png'), 
                                 self.families_global_indices[class_name]) for sha in self.data[f'step={self.step}'][class_name]])
                    
            cls_id_list.append(self.families_global_indices[class_name])
                
        self.cls_id_list_so_far.update(set(cls_id_list))

        self.training_classes_current_step = list(set(cls_id_list))
        self.training_cls_number_each_step.append(len(self.training_classes_current_step))
                    
        self.indices = indices
        # self.training_classes_current_step = training_classes_current_step
        
        # print(f"Initialized {self.train_test} dataset with {len(self.indices)} samples across {len(cls_id_list)} classes: {' '.join(map(str, cls_id_list))}.")
        print(f"Initialized exemplar dataset with {len(self.exemplars)} samples across {len(cls_id_list)} classes.")

    def _update_exemplars(self, exemplars):
        if exemplars is None:
            return
        self.exemplars = list(itertools.chain(*exemplars)) + self.new_samps_of_old_families

    def set_incremental_step(self, step):
        self.step = step
        self.new_samps_of_old_families = []
        self.update_data_indices()


if __name__ == "__main__":

    import json
    from exemplar import gen_random_step_exemplar_set
    
    root_data_path = 'DATASET/MalNet/malnet-images-tiny/images_all/'

    with open('../data_info/tiny/steps_family_samples_train.json', 'r') as f:
        train_data = json.load(f)
            
    with open('../data_info/tiny/steps_family_samples_test.json', 'r') as f:
        val_data = json.load(f)

    # Set hyperparameters
    mode = "new"
    init_num_classes = len(train_data['step=0'])  # The initial number of classes
    families_global_indices = {}
    batch_size = 256
    num_steps = len(train_data)
    exemplar_numper_per_class_per_step = 3

    mapping_dict = {} #to ensure labels in the training are ordinal  
    inverse_mapping_dict = {} #to retrieve original labels back
    
    incremental_nbr_new_classes = [0] #we assume that before step 0, we had 0 families

    train_dataset = ExemplarIncrementalDataset(train_data, root_data_path, families_global_indices, "train", mode)
    val_dataset   = ExemplarIncrementalDataset(val_data, root_data_path, families_global_indices, "test", 'both')
    
    exemplars = None
    
    for step in range(num_steps):

        train_dataset.set_incremental_step(step)
        val_dataset.set_incremental_step(step)

        train_dataset._update_exemplars(exemplars)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_dataloader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        training_classes_current_step = train_dataset.training_classes_current_step #global labels of new families in train
        
        new_classes = [] #to save mapped labels from mapping_dict of new families 

        for i in range(len(training_classes_current_step)):
            
            if training_classes_current_step[i] not in mapping_dict:
                mapping_dict[training_classes_current_step[i]] = len(mapping_dict)
                inverse_mapping_dict[len(mapping_dict)-1] = training_classes_current_step[i]
                new_classes.append(mapping_dict[training_classes_current_step[i]])
                
        # print("new_classes", new_classes)        
        incremental_nbr_new_classes.append(len(new_classes)+incremental_nbr_new_classes[-1])

        '''
        [Training Process]
        '''
        exemplars = gen_random_step_exemplar_set(step, exemplar_numper_per_class_per_step, train_dataset, exemplars)
        print('train_class_number', len(train_dataset.training_classes_current_step))
        print('exemplar size: ', len(exemplars))