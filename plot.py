import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import itertools,math
from sklearn.metrics import confusion_matrix
from matplotlib.mlab import bivariate_normal
from matplotlib.colors import LogNorm


plt.tight_layout()
#sns.set_theme()
sns.set_style("darkgrid")
OUTPUT_DIR = 'results'

def plot_vipi_history(metric_dictionary,title,cat = 'n_state', output_file=None,logscale=False):
    plt.figure()
    for key, value in metric_dictionary.items():
        plt.plot(value,'-', label='{}-'.format(cat)+str(key),alpha=0.7)
    plt.legend()
    if logscale == True:
        plt.yscale('log')
    plt.xlabel('number of episodes')
    plt.ylabel('Value Difference')
    plt.title(title)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)

def plot_part1_df(df):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    vi = df[(df.alg=='VI') &(df.gamma==0.95)]
    sns.barplot(x="n_state", y="iter", data=vi, ax=axes[0])\
        .set(title='VI-Iterations to Converge',ylabel='iterations',xlabel='State Size')
    sns.barplot(x="n_state", y="time", data=vi, ax=axes[1])\
        .set(title='VI-Time to Converge',ylabel='Time',xlabel='State Size')
    f.savefig('part1-vi.png')

    f, axes = plt.subplots(1, 2, figsize=(10,5))
    pi = df[(df.alg=='PI')&(df.gamma==0.95)]
    sns.barplot(x="n_state", y="iter", data=pi, ax=axes[0])\
        .set(title='PI-Iterations to Converge',ylabel='iterations',xlabel='State Size')
    sns.barplot(x="n_state", y="time", data=pi, ax=axes[1])\
        .set(title='PI-Time to Converge',ylabel='Time',xlabel='State Size')
    f.savefig('part1-pi.png')

def plot_data_heatmap(df, output_file):
    plt.figure()
    x = 'n_state'
    y='gamma'
    sns.set(font_scale=1.4)
    fig, axs = plt.subplots(ncols=2, figsize=(15,15))
    z1 = 'iter'
    pivot_tab = df.pivot(x,y,z1)
    Z1 = bivariate_normal(df[x], df[y], 0.1, 0.2, 1.0, 1.0) + 0.1 * bivariate_normal(df[x], df[y], 1.0, 1.0, 0.0, 0.0)
    cbar_ticks = [math.pow(10, i) for i in range(math.floor(math.log10(df[z1].min())), 1+math.ceil(math.log10(df[z1].max())))]
    sns.heatmap(
        pivot_tab, 
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        norm=LogNorm(vmin=Z1.min(), vmax=Z1.max()),
        cbar_kws={"ticks": cbar_ticks,"shrink": .3},
        ax=axs[0],
        cbar=True,
        )
    axs[0].set_title('Iterations to Converge')
    z2 = 'time'
    pivot_tab = df.pivot(x,y,z2)
    sns.heatmap(
        pivot_tab, 
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
#        norm=LogNorm(vmin=Z1.min(), vmax=Z1.max()),
#        cbar_kws={"ticks": cbar_ticks},
        cbar=True,
        cbar_kws={"shrink": .3},
        ax=axs[1]
        )
    
    axs[0].figure.tight_layout()
#    axs[0].set_xticklabels(axs[0].get_xticklabels(),rotation=45, horizontalalignment='right')
    axs[1].set_title('Run Time')
    
    plt.savefig(output_file)


def plot_cluster_metrics(metrics_df, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    metrics_df['n_cluster'] = metrics_df['n_cluster'].astype(int)
    
    credit_df = metrics_df[metrics_df['data']=='creditcard']
    sns.lineplot(y="silh", x= "n_cluster", data=credit_df,label='silh_score',marker='o', ax=axes[0])
    ax = sns.lineplot(y="cluster_acc", x= "n_cluster", data=credit_df,label='cluster_accuracy',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')
#    plt.ylim(0,1)
    
    cancer_df = metrics_df[metrics_df['data']=='cancer']
    sns.lineplot(y="silh", x= "n_cluster", data=cancer_df,label='silh_score',marker='o', ax=axes[1])
    sns.lineplot(y="cluster_acc", x= "n_cluster", data=cancer_df,label='cluster_accuracy',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
#    plt.ylim(0,1)
    
    f.savefig(output_file)

    
    
def plot_cumsum(df, title, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
#    df['n_cluster'] = df['n_cluster'].astype(int)
    
    credit_df = df[df['data']=='creditcard']
    sns.lineplot(y="cumsum", x= "n_cluster", data=credit_df,label='variance(cumsum)',marker='o', ax=axes[0])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=credit_df,label='reconstruction_error',marker='o', ax=axes[0]).set(title='CreditCard',ylabel='score')
    axes[0].set_ylim(0,1)

    cancer_df = df[df['data']=='cancer']
#    print(cancer_df)
    sns.lineplot(y="cumsum", x= "n_cluster", data=cancer_df,label='variance(cumsum)',marker='o', ax=axes[1])
    sns.lineplot(y="reconstruction_error", x= "n_cluster", data=cancer_df,label='reconstruction_error',marker='o', ax=axes[1]).set(title='Cancer',ylabel='score')
    axes[1].set_ylim(0,1)
    
    f.savefig(output_file)
    
def plot_part3(df, output_file):
    df.n_cluster = df.n_cluster.astype(int)
    f, axes = plt.subplots(2, 2, figsize=(10,10))
    
    alg_list = ['PCA','ICA','RP','RFE','original']
    data_list = ['creditcard','cancer']
    for i, data in enumerate(data_list):
        for alg in alg_list:
            sub_df = df[df['data']== data]
            axes[0][i].plot('n_cluster','silh','o-',data=sub_df[sub_df['alg']==alg], label=alg)
            axes[1][i].plot('n_cluster','cluster_acc','o-',data=sub_df[sub_df['alg']==alg], label=alg)
            
        axes[0][i].legend()
        axes[0][i].set_xlabel('n_cluster')
        axes[0][i].set_ylabel('silh score')
        axes[0][i].set_title(data+'_silh')
        axes[1][i].legend()
        axes[1][i].set_xlabel('n_cluster')
        axes[1][i].set_ylabel('accuracy')
        axes[1][i].set_title(data+'_accuracy') 
    f.savefig(output_file)
    
def plot_part4_test_acc(df, output_file):
#    plt.figure()
#    plt.bar('alg','acc',data=df)
#    plt.savefig(output_file)
    plt.figure()
    plt.plot('acc','auc','o', data=df)
    for i, row in df.iterrows():
        plt.annotate(row['alg'], (row['acc'], row['auc']), fontsize=15)
    plt.xlabel('test accuracy')
    plt.ylabel('test AUC')
    plt.title('NN Performance on test data')
    plt.savefig(output_file)
    
    
def plot_part4_history(history_dict, output_file):
    f, axes = plt.subplots(1, 2, figsize=(10,5))
    
    for key, history in history_dict.items():
        axes[0].plot(history['val_acc'],label=key)
    axes[0].legend()
    axes[0].set_title('Validation Accuracy during training')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('accuracy')
    
    for key, history in history_dict.items():
        axes[1].plot(history['val_loss'],label=key)
    axes[1].legend()
    axes[1].set_title('Validation loss during training')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')    
    f.savefig(output_file)
    
def plot_confusion_matrix(cm, classes,
                        normalize=True,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    sns.heatmap(cm, cmap=cmap)
#    sns.heatmap(cm, cmap=cmap,annot=True)
    
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm.round(3)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.savefig('test.png')
