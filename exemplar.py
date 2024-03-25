import random
from collections import defaultdict
# import itertools

def gen_random_step_exemplar_set(step, exemplar_numper_per_class_per_step, train_dataset, old_sublists):

    training_classes_current_step = train_dataset.training_classes_current_step

    categories = defaultdict(list)
    for data_tensor, class_id in train_dataset.indices:
        categories[class_id].append((data_tensor, class_id))

    # Select n tuples from each category in the range N
    selected_sublists = [random.sample(categories[i], min(exemplar_numper_per_class_per_step, len(categories[i]))) for i in training_classes_current_step]
    
    # # Concatenate all the selected sublists into one list using itertools.chain
    # new_exemplars = list(itertools.chain(*selected_sublists))

    if old_sublists is not None:
        return selected_sublists + old_sublists
    else:
        return selected_sublists
    
def trim_sublists(sublists, N):
    for i, sublist in enumerate(sublists):
        sublists[i] = sublist[:N]
    return sublists
    
def gen_fixed_random_step_exemplar_set(step, total_memory_budget, train_dataset, old_sublists):
    
    training_cls_number_each_step = train_dataset.training_cls_number_each_step
    training_classes_current_step = train_dataset.training_classes_current_step
    # new_classes_current_step = train_dataset.new_cls_ids_in_current_step
    # training_classes_so_far = train_dataset.cls_id_list_so_far

    categories = defaultdict(list)
    for data_tensor, class_id in train_dataset.indices:
        categories[class_id].append((data_tensor, class_id))

    # exemplar_numper_per_class_per_step = int(total_memory_budget / len(training_classes_so_far))
    exemplar_numper_per_class_per_step = int(total_memory_budget / sum(training_cls_number_each_step))

    # Select n tuples from each category in the range N
    # selected_sublists = [random.sample(categories[i], min(exemplar_numper_per_class_per_step, len(categories[i]))) for i in new_classes_current_step]  # this version would miss exemplars for new samples of old families
    selected_sublists = [random.sample(categories[i], min(exemplar_numper_per_class_per_step, len(categories[i]))) for i in training_classes_current_step]

    if old_sublists is not None:
        old_sublists = trim_sublists(old_sublists, exemplar_numper_per_class_per_step)
        return selected_sublists + old_sublists
    else:
        return selected_sublists
    