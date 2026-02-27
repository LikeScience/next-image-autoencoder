import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def metrics_predict_map(env,model,config,device,scaling_factor,input_test_processed,target_test,tracked_losses_train, tracked_losses_test,outdir=None,allo_width=None,allo_height=None,ego_len=None):
    input_test_processed = input_test_processed.to(device)
    output = model(input_test_processed)  # Model output (logits or probabilities)
    output = output.cpu() 

    if config["observation_space"] == "allocentric":
        allocentric=True
    else:
        allocentric=False
        
    # Initialize a list to hold the decoded outputs for each observation
    decoded_outputs = []
    if allocentric:
        incorrect_agent_num=0
        incorrect_agent_location=0
    count_by_type_output=[0]*11
    count_by_type_target=[0]*11
    FN_cell_by_type=[0]*11
    FP_cell_by_type=[0]*11
    incorrect_cell=0
    not_perfect=0

    for i in range(output.size(0)):  # Loop over each sample in the batch
        # Extract the blocks of the output
        decoded_map = output[i]
        resized_decoded_map = decoded_map*scaling_factor
        rounded_decoded_map = torch.round(resized_decoded_map).long()
        if allocentric:
            num_agents = 0
        maybe_perfect=1

        for j, val in enumerate(rounded_decoded_map):
            if allocentric and val== 10:
                num_agents +=1
                if (target_test[i, j] != 10):
                    incorrect_agent_location=1
            if 0 <= val and val <= 10:
                count_by_type_output[val]+=1
            count_by_type_target[target_test[i, j]]+=1
            if (target_test[i, j] != val):
                maybe_perfect=0
                incorrect_cell +=1
                FN_cell_by_type[target_test[i, j]]+=1
                if 0 <= val and val <= 10:
                    FP_cell_by_type[val]+=1
        if allocentric:
            if num_agents != 1:
                maybe_perfect=0
                incorrect_agent_num +=1

        if not maybe_perfect:
            not_perfect +=1
        
        # Append the decoded sample to the final list
        decoded_outputs.append(rounded_decoded_map)
    if env is not None:
        allo_width, allo_height, ego_len =env.width, env.height ,env.unwrapped.agent_view_size
    if allocentric:
        acc_cell = 1.0-(incorrect_cell/((allo_width*allo_height)*output.size(0)))
        acc_agentloc = 1.0-(incorrect_agent_location/output.size(0))
        acc_agentnum = 1.0-(incorrect_agent_num/output.size(0))
    else:
        acc_cell = 1.0-(incorrect_cell/((ego_len**2)*output.size(0)))
    acc_cell_type = [((1.0-FN_cell_by_type[i]/count_by_type_target[i]) if count_by_type_target[i] >0 else -1) for i in range(11)]
    prec_cell_type = [((1.0-FP_cell_by_type[i]/count_by_type_output[i]) if count_by_type_output[i] >0 else -1) for i in range(11)]
    acc_perfect = 1.0-(not_perfect/output.size(0))

    # Convert to tensor if needed
    decoded_outputs = torch.stack(decoded_outputs)
    print(output[0])

    print ("MSE Loss at the end of all the epochs, test:", tracked_losses_test[-1])
    print ("Cell accuracy:", acc_cell)
    print ("Perfectness:", acc_perfect)
    if allocentric:
        print ("Number of agents accuracy:", acc_agentnum)
        print ("Agent location accuracy:", acc_agentloc)
    print ("Cell accuracy by type:", acc_cell_type)
    print ("Cell precision by type:", prec_cell_type)
    
    results = {"tracked_losses_train": tracked_losses_train,
               "tracked_losses_test": tracked_losses_test,
               "cell_acc":acc_cell,
               "perfect":acc_perfect,
               "cell_acc_by_type":acc_cell_type,
               "cell_prec_by_type":prec_cell_type}
    
    if allocentric:
        results["num_agents_accuracy"] = acc_agentnum
        results["agent_loc_accuracy"] = acc_agentloc

    if outdir == None: 
        out_dir = f"outputs/exploring_latent_space/model_evaluation/{config['config_name']}.json"
    else: 
        out_dir = outdir
    with open(out_dir, "w") as f:
        json.dump(results, f, indent=4)

    return decoded_outputs

def print_metrics(config, metrics):
    print ("MSE Loss end of epochs, train:", metrics["tracked_losses_train"])
    print ("MSE Loss end of epochs, test:", metrics["tracked_losses_test"])
    print ("Cell accuracy:", metrics["cell_acc"])
    print ("Perfectness:", metrics["perfect"])
    if config["observation_space"] == "allocentric":
        print ("Number of agents accuracy:", metrics["num_agents_accuracy"])
        print ("Agent location accuracy:", metrics["agent_loc_accuracy"])
    print ("Cell accuracy by type:", metrics["cell_acc_by_type"])
    print ("Cell precision by type:", metrics["cell_prec_by_type"])


def metrics_predict_rgb_map(model,config,device,scaling_factor,input_test_processed,target_test,tracked_losses_train, tracked_losses_test,outdir=None,allo_width=None,allo_height=None,ego_len=None):
    input_test_processed = input_test_processed.to(device)
    output = model(input_test_processed)  # Model output (logits or probabilities)
    output = output.cpu() 
        
    # Initialize a list to hold the decoded outputs for each observation
    decoded_outputs = []
    
    for i in range(output.size(0)):
        decoded_map = output[i]
        resized_decoded_map = decoded_map*scaling_factor
        rounded_decoded_map = torch.round(resized_decoded_map).long()
        decoded_outputs.append(rounded_decoded_map)

    # Convert to tensor if needed
    decoded_outputs = torch.stack(decoded_outputs)

    print ("MSE Loss at the end of all the epochs, test:", tracked_losses_test[-1])
    
    results = {"tracked_losses_train": tracked_losses_train,
               "tracked_losses_test": tracked_losses_test,
               }

    if outdir == None: 
        out_dir = f"outputs/exploring_latent_space/model_evaluation/{config['config_name']}.json"
    else: 
        out_dir = outdir
    with open(out_dir, "w") as f:
        json.dump(results, f, indent=4)

    return decoded_outputs

RGBtitles = ['Red Channel', 'Green Channel', 'Blue Channel']
RGBcmaps = ['Reds', 'Greens', 'Blues']

def metrics_predict_rgb_with_decoding(model,config,device,scaling_factor,input_test_processed,target_test,tracked_losses_train, tracked_losses_test,outdir=None,nbins=8,ego_len=None):
    input_test_processed = input_test_processed.to(device)
    output = model(input_test_processed)  # Model output (logits or probabilities)
    output = output.cpu() 
        
    decoded_outputs = []
    
    corr = torch.zeros((nbins, nbins, 3), dtype=torch.long)

    total_pixels = 3*(ego_len**2)*output.size(0)
    for i in range(output.size(0)):
        decoded_map = output[i]
        resized_decoded_map = decoded_map*scaling_factor
        rounded_decoded_map = torch.round(resized_decoded_map).long()
        decoded_outputs.append(rounded_decoded_map)

        pred = rounded_decoded_map.reshape(ego_len, ego_len, 3)
        target = target_test[i].reshape(ego_len, ego_len,3).int()
        pred_bins = (pred.float() / 256 * nbins).long().clamp(0, nbins - 1)
        target_bins = (target.float() / 256 * nbins).long().clamp(0, nbins - 1)
        for ch in range(3):
            p_flat = pred_bins[:, :, ch].reshape(-1)
            t_flat = target_bins[:, :, ch].reshape(-1)
            
            combined_indices = t_flat * nbins + p_flat
            counts = torch.bincount(combined_indices, minlength=nbins**2)
            corr[:, :, ch] += counts.reshape(nbins, nbins)

    decoded_outputs = torch.stack(decoded_outputs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ch in range(3):
        out_path_ch_heatmap = outdir+f"/confusion_heatmap.png"
        data = corr.numpy()
        im = axes[ch].imshow(data[:, :, ch], cmap=RGBcmaps[ch], origin='lower')
        axes[ch].set_title(RGBtitles[ch])
        axes[ch].set_xlabel('Predicted Bin')
        axes[ch].set_ylabel('Target Bin')
        fig.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path_ch_heatmap)
    plt.show()

    range_img = config["range_images_of_predictions_to_save"]
    imgt = target_test.reshape(-1,ego_len, ego_len, 3).astype(np.uint8)
    imgo = decoded_outputs.cpu().numpy().astype(np.uint8).reshape(-1,ego_len, ego_len, 3)
    for i in range(range_img[0],range_img[1]):
        fig, ax = plt.subplots(1, 2, figsize=(5, 2))
        ax[0].imshow(imgt[i], interpolation='nearest')
        ax[0].set_title("Target")
        ax[0].axis('off')
        
        ax[1].imshow(imgo[i], interpolation='nearest')
        ax[1].set_title("Output")
        ax[1].axis('off')
        plt.savefig(outdir+f"/agent_view_{i}.png")
        plt.show()

    print ("MSE Loss at the end of all the epochs, test:", tracked_losses_test[-1])
    corr_to_save = corr.permute(2, 0, 1).tolist()
    results = {"tracked_losses_train": tracked_losses_train,
               "tracked_losses_test": tracked_losses_test,
               "pixel_acc": (corr.diagonal(dim1=0, dim2=1).sum().item() / total_pixels),
                    "confusion_matrices": {
                        "red": corr_to_save[0],
                        "green": corr_to_save[1],
                        "blue": corr_to_save[2]
                    }
               }

    if outdir == None: 
        out_dir = f"metrics.json"
    else: 
        out_dir = outdir+ "/metrics.json"
    with open(out_dir, "w") as f:
        json.dump(results, f, indent=4)

    return decoded_outputs



def latent_space_PCA(latent_spaces,config,pos_list_train, dir_list_train,img_width,img_height,outdir=None):
    top_components = []
    for latent_space, label in latent_spaces:
        U, S, V = torch.pca_lowrank(latent_space, q=None, center=True, niter=2)
        mean = latent_space.mean(dim=0)
        input_centered = latent_space - mean

        top_2_components = V[:, :2]

        input_transformed_2D = input_centered @ top_2_components

        input_2D_np = input_transformed_2D.detach().cpu().numpy()

        top_3_components = V[:, :3]

        input_transformed_3D = input_centered @ top_3_components

        input_3D_np = input_transformed_3D.detach().cpu().numpy()

        explained_variance = (S**2) / (latent_space.size(0) - 1) / torch.var(latent_space, dim=0).sum()
        cumulative_variance = torch.cumsum(explained_variance, dim=0)
        variance_dict = {"explained_variance":explained_variance.tolist(),"cumulative_variance":cumulative_variance.tolist()}
        if outdir == None:
            OUT_DIR = f"outputs/exploring_latent_space/latent_space_PCA/{config['config_name']}"
        else:
            OUT_DIR = outdir
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{OUT_DIR}/variance_{label}.json", "w") as f:
            json.dump(variance_dict, f, indent=4)


        for i in config["variables_for_viewing_latent_space_representations"]:
            Path(f"{OUT_DIR}/{i}").mkdir(parents=True, exist_ok=True)
            if (i == "x_position"):
                pos_list_train_np = np.array(pos_list_train)
                scatter = plt.scatter(input_2D_np[:, 0], input_2D_np[:, 1], c=pos_list_train_np[:,0], cmap='viridis', s=5)
                plt.colorbar(scatter, shrink=0.4, pad=0.05, label="X")
                plt.xlabel("1st principal component")
                plt.ylabel("2nd principal component")
                plt.savefig(f"{OUT_DIR}/{i}/2D_{label}.png")
                plt.clf()
                ax = plt.figure().add_subplot(projection='3d')
                ax.set_xlabel("1st principal component")
                ax.set_ylabel("2nd principal component")
                ax.set_zlabel("3rd principal component")
                scatter = ax.scatter(input_3D_np[:, 0], input_3D_np[:, 1], input_3D_np[:, 2], c=pos_list_train_np[:,0], cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='X',shrink=0.6, pad=0.1)
                plt.savefig(f"{OUT_DIR}/{i}/3D_{label}.png")
                plt.clf()
            elif (i == "y_position"):
                pos_list_train_np = np.array(pos_list_train)
                scatter = plt.scatter(input_2D_np[:, 0], input_2D_np[:, 1], c=pos_list_train_np[:,1], cmap='viridis', s=5)
                plt.colorbar(scatter, shrink=0.4, pad=0.05, label="Y")
                plt.xlabel("1st principal component")
                plt.ylabel("2nd principal component")
                plt.savefig(f"{OUT_DIR}/{i}/2D_{label}.png")
                plt.clf()
                ax = plt.figure().add_subplot(projection='3d')
                ax.set_xlabel("1st principal component")
                ax.set_ylabel("2nd principal component")
                ax.set_zlabel("3rd principal component")
                scatter = ax.scatter(input_3D_np[:, 0], input_3D_np[:, 1], input_3D_np[:, 2], c=pos_list_train_np[:,1], cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='Y',shrink=0.6, pad=0.1)
                plt.savefig(f"{OUT_DIR}/{i}/3D_{label}.png")
                plt.clf()
            elif (i== "L2_dist_center"):
                dist_list = np.sqrt((np.array(pos_list_train)[:,0]-(img_width-1)/2)**2+ (np.array(pos_list_train)[:,1]-(img_height-1)/2)**2)
                scatter = plt.scatter(input_2D_np[:, 0], input_2D_np[:, 1], c=dist_list, cmap='viridis', s=5)
                plt.colorbar(scatter, shrink=0.4, pad=0.05, label="L2 distance to center")
                plt.xlabel("1st principal component")
                plt.ylabel("2nd principal component")
                plt.savefig(f"{OUT_DIR}/{i}/2D_{label}.png")
                plt.clf()
                ax = plt.figure().add_subplot(projection='3d')
                ax.set_xlabel("1st principal component")
                ax.set_ylabel("2nd principal component")
                ax.set_zlabel("3rd principal component")
                scatter = ax.scatter(input_3D_np[:, 0], input_3D_np[:, 1], input_3D_np[:, 2], c=dist_list, cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='L2 distance to center',shrink=0.6, pad=0.1)
                plt.savefig(f"{OUT_DIR}/{i}/3D_{label}.png")
                plt.clf()   
            elif (i == "head_direction"):
                dir_list_train_np = np.array(dir_list_train)
                scatter = plt.scatter(input_2D_np[:, 0], input_2D_np[:, 1], c=dir_list_train_np[:], cmap='viridis', s=5)
                plt.colorbar(scatter, shrink=0.4, pad=0.05, label="Head direction")
                plt.xlabel("1st principal component")
                plt.ylabel("2nd principal component")
                plt.savefig(f"{OUT_DIR}/{i}/2D_{label}.png")
                plt.clf()
                ax = plt.figure().add_subplot(projection='3d')
                ax.set_xlabel("1st principal component")
                ax.set_ylabel("2nd principal component")
                ax.set_zlabel("3rd principal component")
                scatter = ax.scatter(input_3D_np[:, 0], input_3D_np[:, 1], input_3D_np[:, 2], c=dir_list_train_np[:], cmap='viridis')
                plt.colorbar(scatter, ax=ax, label='Head direction',shrink=0.6, pad=0.1)
                plt.savefig(f"{OUT_DIR}/{i}/3D_{label}.png")
                plt.clf()
        top_components.append(top_3_components)
    return top_components

def accuracy(a,b):
    if a.ndim == 1:
        return (a == b).float().mean().item()
    return (a == b).all(dim=-1).float().mean().item()

def mse (a, b):
    return (a-b).pow(2).mean().item()

def round(a):
    return a.round()

def identity(a):
    return a

def argmax(a):
    return torch.argmax(a,dim=1)