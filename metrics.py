import numpy as np

def top_1_accuracy(predictions, true_labels):
    """
    Calculate the Top-1 accuracy.
    
    :param predictions: List of predicted labels
    :param true_labels: List of true labels
    :return: Top-1 accuracy
    """
    assert len(predictions) == len(true_labels), "Predictions and true_labels must have the same length"
    
    # Count the number of correct predictions
    correct_predictions = sum(p == t for p, t in zip(predictions, true_labels))
    
    # Calculate the Top-1 accuracy
    top_1_accuracy = correct_predictions / len(predictions)
    
    return top_1_accuracy

def select_samples_of_only_previous_steps(images, labels, indices, indices_test_end_step, step):

    selected_indices = np.where(indices < indices_test_end_step[step-1])[0]
            
    return images[selected_indices], labels[selected_indices]

def calculate_forgetting_score(step, preds_mapped, labels_mapped, task_best_acc_list, incremental_nbr_new_classes, val_init_samp_indices_list, indices_test_end_step):
    
    predictions = np.array(preds_mapped)
    true_labels = np.array(labels_mapped)
    init_samp_indices = np.array(val_init_samp_indices_list)

    if step > 0:
        predictions, true_labels = select_samples_of_only_previous_steps(predictions, true_labels, init_samp_indices, indices_test_end_step, step)

    old_task_acc_list = []
    for i in range(step+1):
        step_class_list = range(incremental_nbr_new_classes[i], incremental_nbr_new_classes[i+1])
        
        step_class_idxs = []
        for c in step_class_list:
            idxs = np.where(true_labels == c)[0].tolist()
            step_class_idxs += idxs
        step_class_idxs = np.array(step_class_idxs)
        
        if len(step_class_idxs):
            i_labels = true_labels[step_class_idxs]
            i_logits = predictions[step_class_idxs]
        else:
            i_labels = true_labels
            i_logits = predictions

        i_acc = top_1_accuracy(i_logits, i_labels)
        if i == step:
            curren_step_acc = i_acc
        else:
            old_task_acc_list.append(i_acc)
            
    if step > 0:
        forgetting = np.mean(np.array(task_best_acc_list) - np.array(old_task_acc_list))
        for i in range(len(task_best_acc_list)):
            task_best_acc_list[i] = max(task_best_acc_list[i], old_task_acc_list[i])
    else:
        forgetting = None

    task_best_acc_list.append(curren_step_acc)

    return forgetting, old_task_acc_list + [curren_step_acc]
