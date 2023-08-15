from optimization import optimization_loop
import numpy as np
import os
import SimpleITK as sitk
from create_test_dataset import fn_scan_to_array, fn_segm_mask_to_array, createtest_datasets
from sklearn.metrics import accuracy_score, classification_report


def get_xw_trainspaces(test_scan_no):
    W = np.empty((193,27))
    X = np.empty((150,27))
    i = 0
    for scan_no in os.listdir('global_descriptors'):
        scan_no = scan_no.split('_')[1].split('.')[0]
        if scan_no != test_scan_no:
            print(scan_no)
            X[:, i] = np.load("crisp_desc2/gd_"+scan_no+".npy")
            W[:,i] = np.load("weight_vectors/w_"+scan_no+".npy")
            i+=1
    return X, W

def transfer_weight_vector(xi, Dx, Dw, M): #, W_normalizer):
    c_1= Dx.T @ Dx + 0.0001 * np.identity(Dx.shape[1])
    L_xi= np.linalg.inv(c_1) @ Dx.T  @ xi
    L_wi= M @ L_xi
    wi = Dw @ L_wi
    # wi*=47.18468020040863
    return wi


def evaluate_model(x_test, wi, y_test):
    y_pred=[]
    w = wi[:192]
    b= wi[-1]
    for descriptor in x_test:
        # print(descriptor.shape)
        
        y_pred.append((np.sign(np.dot(w, descriptor) + b) + 1) // 2)
    y_pred=np.array(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

with open('readme.txt', 'w') as f:
    f.write('')

    for scan in os.listdir('global_descriptors'):
        scan_no=scan.split('_')[1].split('.')[0]
        f.write(scan_no+ '--')
        scan = fn_scan_to_array(scan_no)
        mask = fn_segm_mask_to_array(scan_no)
        X, W = get_xw_trainspaces(scan_no)
        Dx, Dw, Lx, Lw, M = optimization_loop(X, W, 20)

        x_test, y_test = createtest_datasets(scan_no)
        xi = np.load('crisp_desc2/gd_'+ scan_no + '.npy')

        wi = transfer_weight_vector(xi, Dx, Dw, M)
        wi=np.load('weight_vectors/w_'+ scan_no + '.npy')
        accuracy, report = evaluate_model(x_test, wi, y_test)
        f.write("Accuracy: "+str(accuracy)+ '\n' + report)
        f.write('====================================================================================')