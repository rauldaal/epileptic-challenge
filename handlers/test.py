import cv2
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
import seaborn as sns
import logging
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score
)
from torchvision.utils import make_grid


#  use gpu if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = .5


def test(model, test_data_loader, criterion):

    test_loss = 0
    model.eval()
    logging.info("++++++++"*10)
    correct_predictions = 0
    total_samples = 0
    predicciones_lista = []
    etiquetas_verdaderas_lista = []
    for window, cls in tqdm.tqdm(test_data_loader):
        
        window = window.to(DEVICE, dtype=torch.float)
        cls = cls.to(DEVICE, dtype=torch.float)
        with torch.no_grad():
            outputs = model(window)
            predictions = (outputs >= THRESHOLD).float()
            correct_predictions += (predictions == cls.view(-1, 1)).sum().item()
            total_samples += cls.size(0)
            loss = criterion(outputs, cls.view(-1, 1))
            test_loss += loss.item()
            predicciones_lista.append(predictions.cpu().numpy())
            etiquetas_verdaderas_lista.append(cls.cpu().numpy())
            wandb.log({"test_loss": test_loss})
            logging.info(f"Test Loss {test_loss}")
            logging.info("++++++++"*10)
    accuracy = correct_predictions / total_samples
    logging.info(f"ACCURACY {accuracy}")
    y_pred = np.concatenate(predicciones_lista, axis=0)
    y_true = np.concatenate(etiquetas_verdaderas_lista, axis=0)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Tipo 0', 'Tipo 1'], yticklabels=['Tipo 0', 'Tipo 1'])
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.title('Matriz de Confusión')
    plt.show()

    # compute the epoch test loss
    test_loss = test_loss / len(test_data_loader)
    return


def test_lstm(model, test_data_loader, criterion):

    test_loss = 0
    model.eval()
    logging.info("++++++++"*10)
    correct_predictions = 0
    total_samples = 0
    predicciones_lista = []
    etiquetas_verdaderas_lista = []
    for window, cls in tqdm.tqdm(test_data_loader):
        
        window = window.to(DEVICE, dtype=torch.float)
        cls = cls.to(DEVICE, dtype=torch.float)

        batch = window.shape[0]

        with torch.no_grad():
            model.hidden_state = model.init_hidden(hidden_size=64, num_layers=8, batch_size=batch)
            outputs = model(window)
            predictions = (outputs >= THRESHOLD).float()
            correct_predictions += (predictions == cls.view(-1, 1)).sum().item()
            total_samples += cls.size(0)
            loss = criterion(outputs, cls.view(-1, 1))
            test_loss += loss.item()
            predicciones_lista.append(predictions.cpu().numpy())
            etiquetas_verdaderas_lista.append(cls.cpu().numpy())
            wandb.log({"test_loss": test_loss})
            logging.info(f"Test Loss {test_loss}")
            logging.info("++++++++"*10)
    accuracy = correct_predictions / total_samples
    logging.info(f"ACCURACY {accuracy}")
    y_pred = np.concatenate(predicciones_lista, axis=0)
    y_true = np.concatenate(etiquetas_verdaderas_lista, axis=0)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Tipo 0', 'Tipo 1'], yticklabels=['Tipo 0', 'Tipo 1'])
    plt.xlabel('Predicciones')
    plt.ylabel('Etiquetas Verdaderas')
    plt.title('Matriz de Confusión')
    plt.show()

    # compute the epoch test loss
    test_loss = test_loss / len(test_data_loader)
    return

def convertir_a_hsv(input, output):
    input_results = []
    output_results = []
    input = input.permute(0, 2, 3, 1)
    input_canal_h = np.zeros_like(input[:, :, :, 0].to("cpu"), dtype=np.float32)

    output = output.permute(0, 2, 3, 1)
    output_canal_h = np.zeros_like(output[:, :, :, 0].to("cpu"), dtype=np.float32)

    # Itera sobre cada imagen en el batch
    for i in range(input.shape[0]):
        input_imagen_hsv = cv2.cvtColor(input[i].to("cpu").numpy(), cv2.COLOR_RGB2HSV)
        input_canal_h[i] = input_imagen_hsv[:, :, 0]
        f_red_input = np.sum(np.logical_or(input_canal_h[i] >= 340, input_canal_h[i] <= 20))
        #logging.info("++++++"*5)
        #logging.info(f"f_red_input:     {f_red_input}")

        output_imagen_hsv = cv2.cvtColor(output[i].to("cpu").numpy(), cv2.COLOR_RGB2HSV)
        output_canal_h[i] = output_imagen_hsv[:, :, 0]
        f_red_output = np.sum(np.logical_or(output_canal_h[i] >= 340, output_canal_h[i] <= 20))
        #logging.info(f"f_red_output:     {f_red_output}")       
        #logging.info("++++++"*5)
        input_results.append(f_red_input)
        output_results.append(f_red_output)

    return input_results, output_results


def classifier(input, output):
    
    batch_results = []
    division_results = []

    fred_input, fred_output = convertir_a_hsv(input, output)
    for fi, fo in zip(fred_input, fred_output):
        f = fi/fo if fo != 0 else fi
        division_results.append(f)
        if f > THRESHOLD:
            batch_results.append(1)
        else:
            batch_results.append(0)
    return batch_results, division_results


def analyzer(results, true_labels, project_path, name=None):
    fpr, tpr, thresholds = roc_curve(true_labels, results)
    roc_auc = auc(fpr, tpr)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]

    logging.info(f'Umbral óptimo: {optimal_threshold}')
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    name = name if name else "roc.png"
    plt.savefig(project_path+"/plots/"+name)
    plt.show()
    return optimal_threshold


def compute_confussion_matrix(true, pred, project_path, name=None):
    plt.figure(figsize=(8, 8))
    conf_matrix = confusion_matrix(true, pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel('Etiquetas Predichas')
    plt.ylabel('Etiquetas Reales')
    plt.title('Matriz de Confusión')
    name = name if name else "confussion.png"
    plt.savefig(project_path+"/plots/"+name)
    plt.show()
    acc = accuracy_score(true, pred)
    logging.info(f"ACCURACY SCORE: {acc}")


def compute_classification(dataloader, patients_idx, labels, model, project_path):
    generated_labels = []

    for imgs in dataloader:
        imgs = imgs.to(DEVICE, dtype=torch.float)
        with torch.no_grad():
            outputs = model(imgs)
            ret, _ = classifier(imgs, outputs)

            generated_labels.extend(ret)
    
    labels_per_patient = {}
    logging.info(len(list(generated_labels)))
    logging.info(len(list(patients_idx.keys())))
    for indice, id_valor in patients_idx.items():
        if id_valor not in list(labels_per_patient.keys()):
            labels_per_patient[id_valor] = []
        labels_per_patient[id_valor].append(generated_labels[indice-1])
    
    probabilities = []
    actual_label = []

    for patient in labels_per_patient.keys():
        prob = sum(labels_per_patient[patient])/len(labels_per_patient[patient])
        probabilities.append(prob)
        actual_label.append(labels[patient])
    
    optimal = analyzer(results=probabilities, true_labels=actual_label, project_path=project_path, name="final_roc.png")


    final_results = []
    for patient in labels_per_patient.keys():
        prob = sum(labels_per_patient[patient])/len(labels_per_patient[patient])
        if prob > optimal:
            final_results.append(1)
        else:
            final_results.append(0)
    
    compute_confussion_matrix(true=actual_label, pred=final_results, project_path=project_path, name="final_cm.png")

