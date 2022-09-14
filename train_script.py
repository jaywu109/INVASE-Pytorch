# %%

import os
from tqdm import tqdm


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid" )

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import torch

from invase_utils import CCE_loss, Actor_loss, Actor, Critic_or_Baseline, bernoulli_sampling

# %%
train = pd.read_csv('/home/Jay/feature_selection/20220105_task/20220105_data_processed/train_1.csv')
train.PHENOTYPE = (train.PHENOTYPE.values - 1).astype(int)

### filter out columns with same value 
var_result = train[train.columns[6:]].var()
useful_cols = var_result[var_result!=0].index.to_numpy()
###

train_feature = train[useful_cols].values
scaler = StandardScaler()
train_x = scaler.fit_transform(train_feature)

test = pd.read_csv('/home/Jay/feature_selection/20220105_task/20220105_data_processed/test_1.csv')
test.PHENOTYPE = (test.PHENOTYPE.values - 1).astype(int)
test_feature = test[useful_cols].values
test_x = scaler.transform(test_feature)

x_train, x_val, y_train, y_val = train_test_split(train_x, train.PHENOTYPE.values,
                                                    test_size=0.1,
                                                    random_state=26,
                                                    stratify=train.PHENOTYPE.values)

x_test = test_x
y_test = test.PHENOTYPE.values

enc = OneHotEncoder()
y_train = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
y_val = enc.transform(y_val.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

# %%
# np.random.seed(0)
# actor_out = torch.tensor(np.random.randint(low=0,high=2,size=(1000, 532)).astype(np.float32))

# np.random.seed(0)
# prob_axis_0 = np.random.rand(1000,1)
# prob_axis_1 = 1 - prob_axis_0
# critic_out = torch.tensor(np.concatenate((prob_axis_0, prob_axis_1), axis=1))

# np.random.seed(1)
# prob_axis_0 = np.random.rand(1000,1)
# prob_axis_1 = 1 - prob_axis_0
# baseline_out = torch.tensor(np.concatenate((prob_axis_0, prob_axis_1), axis=1))

# np.random.seed(0)
# y_out = torch.tensor(np.random.randint(low=0,high=2,size=(1000, 2)).astype(np.float32))

# np.random.seed(0)
# y_pred = torch.tensor(np.random.rand(1000,532))


# %%
model_parameters = {'lamda': 0.1,
                    'actor_h_dim': 300, 
                    'critic_h_dim': 200,
                    'selector_n_layer': 5,
                    'predictor_n_layer': 4,
                    'batch_size': 3000,
                    'epoch': 20000, 
                    'activation': 'relu', 
                    'learning_rate': 0.00002,
                    'dim': x_train.shape[1],
                    'label_dim': y_train.shape[1]}

# %%
# Training hyperparameters
batch_size = model_parameters['batch_size']
lr = model_parameters['learning_rate']

n_epoch = model_parameters['epoch']
# n_critic = 1 

workspace_dir = '/home/Jay/feature_selection/INVASE'
log_dir = os.path.join(workspace_dir, 'logs')
ckpt_dir = os.path.join(workspace_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Model
actor = Actor(model_parameters).cuda()
critic = Critic_or_Baseline(model_parameters).cuda()
baseline = Critic_or_Baseline(model_parameters).cuda()

# Loss
cce_loss = CCE_loss()
actor_loss = Actor_loss(lamda = model_parameters['lamda'])

# Optimizer
opt_actor = torch.optim.Adam(actor.parameters(), lr=lr, weight_decay=1e-3)
opt_critic = torch.optim.Adam(critic.parameters(), lr=lr, weight_decay=1e-3)
opt_baseline = torch.optim.Adam(baseline.parameters(), lr=lr, weight_decay=1e-3)

x_train_T = torch.tensor(x_train).cuda().float()
x_val_T = torch.tensor(x_val).cuda().float()
x_test_T = torch.tensor(x_test).cuda().float()
y_train_T = torch.tensor(y_train).cuda().float()
y_val_T = torch.tensor(y_val).cuda().float()
y_test_T = torch.tensor(y_test).cuda().float()

# # DataLoader
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)



# %%
best_val_acc = 0
best_epoch = 0
critic_loss_list = []
actor_loss_list = []
baseline_loss_list = []
val_acc_list = []
for iter_idx in range(n_epoch):
    ## Train critic
    # Select a random batch of samples
    # x_train_batch_T = x_train_T
    # y_train_batch_T = y_train_T
    idx = np.random.randint(0, x_train_T.shape[0], batch_size)
    x_train_batch_T = x_train_T[idx, :].float()
    y_train_batch_T = y_train_T[idx, :].float()

    # Generate a batch of selection probability
    actor.eval()
    with torch.no_grad():
        selection_probability = actor(x_train_batch_T)
    # Sampling the features based on the selection_probability
    selection = bernoulli_sampling(selection_probability)
    # Critic loss
    critic.train()
    critic_pred = critic((x_train_batch_T * selection))
    critic_loss = cce_loss(critic_pred, y_train_batch_T)
    # Model backwarding
    critic.zero_grad()
    critic_loss.backward()
    # Update the critic
    opt_critic.step()    

    
    # Baseline loss
    baseline.train()
    baseline_pred = baseline(x_train_batch_T)
    baseline_loss = cce_loss(baseline_pred, y_train_batch_T)
    # Model backwarding
    baseline.zero_grad()
    baseline_loss.backward()
    # Update the baseline
    opt_baseline.step()    

    ## Train actor
    # Use multiple things as the y_true:
    # - selection, critic_out, baseline_out, and ground truth (y_batch)
    critic.eval()
    baseline.eval()
    with torch.no_grad():
        critic_out = critic((x_train_batch_T * selection))
        baseline_out = baseline(x_train_batch_T)
    y_batch_final = {'selection':selection, 'critic_out':critic_out, 'baseline_out':baseline_out, 'ground_truth': y_train_batch_T}
    # Train the actor
    actor.train()
    actor_pred = actor(x_train_batch_T)
    act_loss = actor_loss(actor_pred, y_batch_final)
    # Model backwarding
    actor.zero_grad()
    act_loss.backward()
    # Update the actor
    opt_actor.step()    

    actor.eval()
    critic.eval()
    with torch.no_grad():
        selection_val = actor(x_val_T)
        selection_val[selection_val > 0.5] = 1
        selection_val[selection_val <= 0.5] = 0
        y_val_hat = critic((x_val_T * selection_val))

        y_val_hat[y_val_hat > 0.5] = 1
        y_val_hat[y_val_hat <= 0.5] = 0
        val_acc = accuracy_score(y_val[:, 0], y_val_hat[:, 0].cpu().numpy())

    actor_loss_list.append(act_loss.item())
    critic_loss_list.append(critic_loss.item())
    baseline_loss_list.append(baseline_loss.item())
    val_acc_list.append(val_acc)

    if iter_idx == 0:
        best_val_acc = val_acc
        torch.save(actor.state_dict(), os.path.join(ckpt_dir,  'Actor.pth'))
        torch.save(critic.state_dict(), os.path.join(ckpt_dir,  'Critic.pth'))
        torch.save(baseline.state_dict(), os.path.join(ckpt_dir,  'Baseline.pth'))
        # head_name = iter_idx + '_' + str(np.round(val_acc, 3) *100) + "_"
        # torch.save(actor.state_dict(), os.path.join(ckpt_dir, head_name + 'Actor.pth'))
        # torch.save(critic.state_dict(), os.path.join(ckpt_dir, head_name + 'Critic.pth'))
        # torch.save(baseline.state_dict(), os.path.join(ckpt_dir, head_name + 'Baseline.pth'))
    else:
        if  val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = iter_idx
            torch.save(actor.state_dict(), os.path.join(ckpt_dir,  'Actor.pth'))
            torch.save(critic.state_dict(), os.path.join(ckpt_dir,  'Critic.pth'))
            torch.save(baseline.state_dict(), os.path.join(ckpt_dir,  'Baseline.pth'))

    dialog = (
        "Epoch: "
        + str(iter_idx)
        + ", actor loss: "
        + str(act_loss.detach().cpu().item())
        + ", critic loss: "
        + str(critic_loss.detach().cpu().item())
        + ", baseline loss: "
        + str(baseline_loss.detach().cpu().item())
        + ", val acc: "
        + str(np.round(val_acc, 4))
        + ", best epoch: "
        + str(best_epoch)
    )
    print(dialog)



# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
sns.lineplot(ax=axs[0, 0], x=range(len(actor_loss_list)), y=actor_loss_list)
axs[0, 0].set_title('Selector Loss')
sns.lineplot(ax=axs[0, 1], x=range(len(critic_loss_list)), y=critic_loss_list)
axs[0, 1].set_title('Predictor Loss')
sns.lineplot(ax=axs[1, 0], x=range(len(baseline_loss_list)), y=baseline_loss_list)
axs[1, 0].set_title('Baseline Loss')
sns.lineplot(ax=axs[1, 1], x=range(len(val_acc_list)), y=val_acc_list)
axs[1, 1].set_title('Validation Accuracy')
fig.savefig('result_all.png')

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
sns.lineplot(ax=axs[0, 0], x=range(len(actor_loss_list[:6000])), y=actor_loss_list[:6000])
axs[0, 0].set_title('Selector Loss')
sns.lineplot(ax=axs[0, 1], x=range(len(critic_loss_list[:6000])), y=critic_loss_list[:6000])
axs[0, 1].set_title('Predictor Loss')
sns.lineplot(ax=axs[1, 0], x=range(len(baseline_loss_list[:6000])), y=baseline_loss_list[:6000])
axs[1, 0].set_title('Baseline Loss')
sns.lineplot(ax=axs[1, 1], x=range(len(val_acc_list[:6000])), y=val_acc_list[:6000])
axs[1, 1].set_title('Validation Accuracy')
fig.savefig('result_all_6000.png')
# sns.lineplot(x=range(len(actor_loss_list)), y=actor_loss_list).set_title('Actor Loss')
# sns.lineplot(x=range(len(critic_loss_list)), y=critic_loss_list).set_title('Critic Loss')
# sns.lineplot(x=range(len(baseline_loss_list)), y=baseline_loss_list).set_title('Baseline Loss')
# sns.lineplot(x=range(len(val_acc_list)), y=val_acc_list).set_title('Validation Accuracy')



