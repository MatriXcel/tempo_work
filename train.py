from netCDF4 import Dataset
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import numpy as np
import torch
from torch import utils

import NN_layer
import torch.nn.functional as F

import tqdm
import wandb

import argparse
import os
import random

import pickle


random.seed(42)

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def parse_args():
    parser = argparse.ArgumentParser(description="Train machine translation transformer model")

    #required argument
    parser.add_argument(
        "--map_file",
        type=str,
        required=True,
        help=("geographic map containing data of different atmospheric variables"
        ),
    )

    args = parser.parse_args()

    return args


def evaluate_model(model, learning_data, feature_map, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    
    with torch.no_grad():
        x, y= learning_data.tensors[0].float().to(device), learning_data.tensors[1].float().to(device)

        pred_AMF = model(x)
        test_loss = torch.sqrt(F.mse_loss(torch.flatten(pred_AMF), y))

   # plt.plot(np.arange(len(learning_data.tensors[0])), pred_AMF, color='red')
    #plt.plot(np.arange(len(learning_data.tensors[0])), y, color='green')

  #  plt.imshow(pred_AMF.reshape(feature_map.shape[0], feature_map.shape[1]), vmin=0.5, vmax=1.5)
    #plt.colorbar()
    #plt.show()
    
    return test_loss


def main():
    args = parse_args()

    # Define path to netCDF file to open and read
    hcho_fp = './TEMPO_20130704_17h_LA1_HCHO.nc'
    o3_fp = './TEMPO_20130704_17h_LA1_O3.nc'
    #la_rtm_fp = './TEMPO_20130704_17h_LA1_rtm.nc'

    map_folder = args.map_file
    file_names = os.listdir(map_folder)

    print("out of ", len(file_names))

    # Open the file. I like to do it using with to guarantee
    # that it will be closed after we are done reading
    # netCDF files are organized in group (which can have subgroups)
    # and data fields. Each group or data field can have attributes.
    # The file as a whole can also have attributes that in this case
    # are called global attributes. Following this example and reading
    # https://unidata.github.io/netcdf4-python/ you should get some
    # familiarity about how to work with them

    if torch.cuda.is_available():
        device = 'gpu'
        print("GPU is available, using gpu")
    else:
        device = 'cpu'
        print("GPU unaivailable, using CPU")

    with Dataset(hcho_fp,'r') as hcho_src, Dataset(o3_fp, 'r') as o3_src:
        # Read geolocation variables
        common_src = hcho_src

        lat = common_src['geolocation']['latitude'][:]
        lon = common_src['geolocation']['longitude'][:]
        terrain_height = common_src['geolocation']['terrain_height'][:]
    
        solar_zenith_angle = common_src['geolocation']['solar_zenith_angle'][:]
        viewing_zenith_angle = common_src['geolocation']['viewing_zenith_angle'][:]
        relative_azimuth_angle = common_src['geolocation']['relative_azimuth_angle'][:]
        
        surface_pressure = common_src['geolocation']['surface_pressure'][:]
        tropopause_pressure = common_src['geolocation']['tropopause_pressure'][:]

        albedo = common_src['support_data']['albedo'][:]

        hcho_gas_profile = hcho_src['support_data']['gas_profile'][:]
        o3_gas_profile = o3_src['support_data']['gas_profile'][:]

    all_training_examples = []
    all_training_labels = []

    successes = 0
    filenames_used = []
    
    random.shuffle(file_names)

    for filename in file_names:
        with Dataset(os.path.join(map_folder, filename),'r') as la_src:
            print(filename, "is running")
            air_partial_col = la_src['Profile']['AirPartialColumn'][:]


            Ar_gas_mixing_ratio = la_src['Profile']['Ar_GasMixingRatio'][:]
            BrO_gas_mixing_ratio = la_src['Profile']['BrO_GasMixingRatio'][:]
            CO2_gas_mixing_ratio = la_src['Profile']['CO2_GasMixingRatio'][:]
            GLYX_gas_mixing_ratio = la_src['Profile']['GLYX_GasMixingRatio'][:]
            H2O_gas_mixing_ratio = la_src['Profile']['H2O_GasMixingRatio'][:]
            HCHO_gas_mixing_ratio = la_src['Profile']['HCHO_GasMixingRatio'][:]
            N2_gas_mixing_ratio = la_src['Profile']['N2_GasMixingRatio'][:]
            NO2_gas_mixing_ratio = la_src['Profile']['NO2_GasMixingRatio'][:]
            O2_gas_mixing_ratio = la_src['Profile']['O2_GasMixingRatio'][:]
            O3_gas_mixing_ratio = la_src['Profile']['O3_GasMixingRatio'][:]
            SO2_gas_mixing_ratio = la_src['Profile']['SO2_GasMixingRatio'][:]

            Ar_amf = la_src['RTM_Band1']['Ar_AMF'][:]
            BrO_amf = la_src['RTM_Band1']['BrO_AMF'][:]
            CO2_amf = la_src['RTM_Band1']['CO2_AMF'][:]
            GLYX_amf = la_src['RTM_Band1']['GLYX_AMF'][:]
            H2O_amf = la_src['RTM_Band1']['H2O_AMF'][:]
            HCHO_amf = la_src['RTM_Band1']['HCHO_AMF'][:]
            N2_amf = la_src['RTM_Band1']['N2_AMF'][:]
            NO2_amf = la_src['RTM_Band1']['NO2_AMF'][:]
            O2_amf = la_src['RTM_Band1']['O2_AMF'][:]
            O3_amf = la_src['RTM_Band1']['O3_AMF'][:]
            SO2_amf = la_src['RTM_Band1']['SO2_AMF'][:]

            #add filter/flag that selects pixels for which there is proper data
            #expected range is around 0-20, start with that, just look at AMF for formaldehyde
            
        
        data_type = np.float64

        Ar_slant_col = (np.sum(Ar_gas_mixing_ratio * air_partial_col, axis=0) * Ar_amf).astype(data_type)
        BrO_slant_col = (np.sum(BrO_gas_mixing_ratio * air_partial_col, axis=0) * BrO_amf).astype(data_type)
        GLYX_slant_col = (np.sum(GLYX_gas_mixing_ratio * air_partial_col, axis=0) * GLYX_amf).astype(data_type)
        H2O_slant_col = (np.sum(H2O_gas_mixing_ratio * air_partial_col, axis=0) * H2O_amf).astype(data_type)
        HCHO_slant_col = (np.sum(HCHO_gas_mixing_ratio * air_partial_col, axis=0) * HCHO_amf).astype(data_type)
        N2_slant_col = (np.sum(N2_gas_mixing_ratio * air_partial_col, axis=0) * N2_amf).astype(data_type)
        NO2_slant_col = (np.sum(NO2_gas_mixing_ratio * air_partial_col, axis=0) * NO2_amf).astype(data_type)
        O2_slant_col = (np.sum(O2_gas_mixing_ratio * air_partial_col, axis=0) * O2_amf).astype(data_type)
        O3_slant_col = (np.sum(O3_gas_mixing_ratio * air_partial_col, axis=0) * O3_amf).astype(data_type)
        SO2_slant_col = (np.sum(SO2_gas_mixing_ratio * air_partial_col, axis=0) * SO2_amf).astype(data_type)


        transform_func = np.log
        #np.seterr(all = 'ignore') 
       
        try:
             with np.errstate(divide='ignore', over='ignore'):
                feature_map = np.stack((
                    np.deg2rad(solar_zenith_angle), #normal
                    np.deg2rad(viewing_zenith_angle), #normal
                    np.deg2rad(relative_azimuth_angle), #normal
                    #transform_func(surface_pressure),  #skewed
                    #transform_func(tropopause_pressure), #skewed
                    #terrain_height,   #skewed
                    albedo,  #skewed

                    #transform_func(Ar_slant_col[0]), #skewed
                    #transform_func(BrO_slant_col[0]), #skewed
                    #transform_func(GLYX_slant_col[0]), #skewed
                    #transform_func(H2O_slant_col[0]), #skewed
                    #transform_func(HCHO_slant_col[0]), #skewed
                    #transform_func(N2_slant_col[0]), #skewe
                    #transform_func(NO2_slant_col[0]), #skewed
                    #transform_func(O2_slant_col[0]), #skewed
                    #transform_func(O3_slant_col[0]), #skewed
                    #transform_func(SO2_slant_col[0]) #skewed
                ), axis=-1)
             print("this one passed")

             training_examples = feature_map.reshape(feature_map.shape[0] * feature_map.shape[1], feature_map.shape[2])
             HCHO_amf_labels = HCHO_amf[0].reshape(HCHO_amf[0].shape[0] * HCHO_amf[0].shape[1])

             all_training_examples.append(training_examples)
             all_training_labels.append(HCHO_amf_labels)
             successes += 1

             filenames_used.append(filename)

             if successes == 3:
                break
        except:
            return 1

    
    print("total number of successes ", successes)
    print("out of ", len(file_names))

  
    training_tensor = torch.from_numpy(np.array(all_training_examples))
    training_tensor = training_tensor.flatten(0, 1)

    label_tensor = torch.from_numpy(all_training_labels).flatten()

    torch.save(training_tensor, 'training_tensor.pt')
    torch.save(label_tensor, 'label_tensor.pt')

    with open('filenames_used.pkl', 'wb') as f:
        pickle.dump(filenames_used, f)

    return
    
    all_feature_maps.append(feature_map)


    learning_data = utils.data.TensorDataset(training_examples, HCHO_amf_labels)



    run = wandb.init(project=f"tempo_ml_training")

    max_epochs = 60000


    input_size = len(learning_data[0][0])

    model = NN_layer.AMFNet(input_size, 10)
    adamWOptim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    previous_loss = 0



    run = wandb.init(project=f"tempo_ml_training")

    max_epochs = 120000
    
    input_size = len(learning_data[0][0])

    model = NN_layer.AMFNet(input_size, 10)
    adamWOptim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    previous_loss = 0

    epsilon = 1e-50

    for epoch in range(max_epochs):

        x, y = learning_data.tensors[0].float().to(device), learning_data.tensors[1].float().to(device)
        pred_AMF = model(x)

        #print(model.fc1.weight)
        
        loss = torch.sqrt(F.mse_loss(torch.flatten(pred_AMF), y))

        if torch.abs(loss - previous_loss) < epsilon:
            break

        previous_loss = loss
        # zero out gradients
        adamWOptim.zero_grad() 
        # 4. Backpropagate the loss
        loss.backward()
        # 5. Update the parameters
        adamWOptim.step()

        wandb.log(
            {
                "train_loss": loss,
                "learning_rate": adamWOptim.param_groups[0]["lr"],
                "epoch": epoch,
            },
            step=epoch,
        )

        if epoch % 200 == 0:
            print(loss)

        # if epoch > 10000 and (loss - previous_loss) <= 1e-20:
        #     break
        
        #previous_loss = loss


    test_loss = evaluate_model(model, learning_data, feature_map)

    print(loss, test_loss)
    run.finish()


if __name__ == "__main__":
    main()
