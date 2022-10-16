import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import json
import dgl
from RegionLevelPre_parser import Parser
from RegionLevelPre_model import GIN
import os
import random
import pdb
from sklearn.metrics import f1_score,roc_curve,accuracy_score, hinge_loss,auc,roc_auc_score,recall_score,cohen_kappa_score,hamming_loss

os.environ['CUDA_VISIBLE_DEVICES']='0'
import setproctitle
setproctitle.setproctitle('GCN@xinshiduo')

def train(args, net, train_mask, optimizer, criterion, epoch,g,regionkeylist,regionlabeldict,h0):
    net.train()


    # bar = tqdm(range(1), unit='batch', position=0, file=sys.stdout)
    graphs = g.to(args.device)
    graphoutputs = net(graphs,h0)
    batchkeys=random.sample(list(regionkeylist),args.batch_size)
    regionloss=0
    for i in range(args.batch_size):

        regionmask=torch.tensor(regionlabeldict[batchkeys[i]]['streets']).cuda()
        selectgraphoutput=torch.sum(graphoutputs[regionmask],dim=0).unsqueeze(0)
        lab=torch.tensor(regionlabeldict[batchkeys[i]]['GT']).cuda().unsqueeze(0)
        regionloss+=criterion(selectgraphoutput,lab)/args.batch_size
    
    loss=regionloss
    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # report
    # bar.set_description('epoch-{}'.format(epoch))
    # bar.close()

    return 0,regionloss.item()


def eval_net(args, net, test_mask, criterion,g,regionkeylist,regionlabeldict,h0):
    net.eval()


    graphs = g.to(args.device)
    graphoutputs = net(graphs, h0)

    regionloss=0
    regiontotal = len(list(regionkeylist))
    regiontotal_correct = 0
    regionmask=torch.tensor(regionlabeldict[regionkeylist[0]]['streets']).cuda()
    outputs=torch.sum(graphoutputs[regionmask],dim=0).unsqueeze(0)
    labels=torch.tensor(regionlabeldict[regionkeylist[0]]['GT']).cuda().unsqueeze(0)
    regionloss+=criterion(outputs,labels)
    _, predicted = torch.max(outputs.data, 1)
    if (predicted == labels):
        regiontotal_correct+=1
    for i in range(1,len(list(regionkeylist))):
        regionmask=torch.tensor(regionlabeldict[regionkeylist[i]]['streets']).cuda()
        selectgraphoutput=torch.sum(graphoutputs[regionmask],dim=0).unsqueeze(0)
        lab=torch.tensor(regionlabeldict[regionkeylist[i]]['GT']).cuda().unsqueeze(0)
        labels=torch.cat((labels,lab),0)
        outputs=torch.cat((outputs,selectgraphoutput),0)
        regionloss+=criterion(selectgraphoutput,lab)
        _, regionpredicted = torch.max(selectgraphoutput.data, 1)
        predicted=torch.cat((predicted,regionpredicted),0)
        if (regionpredicted == lab):
            regiontotal_correct+=1
    yscore_test=outputs.data.cpu().numpy()
    testlab_onehot=np.eye(4)[labels.cpu().numpy()]
    f1score=f1_score(labels.cpu().numpy(),predicted.cpu().numpy(),average='weighted')
    n_classes = 4
    fprt = dict()
    tprt = dict()
    roc_auct = dict()
    for i in range(n_classes):
        fprt[i], tprt[i], _ = roc_curve(testlab_onehot[:, i], yscore_test[:, i])
        roc_auct[i] = auc(fprt[i], tprt[i])
    fprt["micro"], tprt["micro"], _ = roc_curve(testlab_onehot.ravel(), yscore_test.ravel())
    roc_auct["micro"] = auc(fprt["micro"], tprt["micro"])
    regionacc = 1.0*regiontotal_correct / regiontotal
    rescore=recall_score(labels.cpu().numpy(),predicted.cpu().numpy(), average='macro')
    kappa = cohen_kappa_score(labels.cpu().numpy(),predicted.cpu().numpy())
    ham_distance = hamming_loss(labels.cpu().numpy(),predicted.cpu().numpy())
    net.train()

    return 0, 0, regionloss.item()/regiontotal, regionacc,f1score,roc_auct["micro"],kappa,ham_distance,rescore

def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        args.dim_nfeats, args.hidden_dim, args.gclasses,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type).to(args.device)

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    (g,), _ = dgl.load_graphs("./data/graph_poi_num.dgl")
    labels = g.ndata['GT']
    rd=np.random.rand(len(labels))
    rd2=np.random.rand(len(labels))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    
    zero_disab=torch.tensor(np.logical_and(rd2<args.ABORT_ZERO ,(np.array(labels))==0))
    train_mask = torch.tensor(np.logical_and(rd< args.TRAIN_SIZE , (~zero_disab)))
    test_mask = torch.tensor(np.logical_and(~(train_mask) , (~zero_disab)))
    test_mask=test_mask.to(torch.bool).cuda()
    train_mask=train_mask.to(torch.bool).cuda()
    h0 = torch.load('./data/h0_pretrain.pt').cuda()
    file = open('./data/MUTregionlabel_'+args.dataset+'.json',encoding='utf-8')
    regionlabeldict=json.load(file)
    keylist=list(regionlabeldict.keys())
    rd3=np.random.rand(len(keylist))
    keylist=np.array(keylist)
    train_keylist=keylist[rd3<args.TRAIN_SIZE*0.2]
    test_keylist=keylist[rd3>args.TRAIN_SIZE]
    print('train_keylist length:',len(list(train_keylist)))
    print('test_keylist length:',len(list(test_keylist)))
    tbar = tqdm(range(args.epochs), unit="epoch", position=0, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=1, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=2, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, train_mask, optimizer, criterion, epoch,g,train_keylist,regionlabeldict,h0)
        scheduler.step()
        if epoch%100==0:
            nodetrain_loss, nodetrain_acc, regiontrain_loss, regiontrain_acc,_,_,_,_,_ = eval_net(
                args, model, train_mask, criterion,g,train_keylist,regionlabeldict,h0)
            tbar.set_description(
                'train set - region loss: {:.4f}, region accuracy: {:.0f}%'
                .format(regiontrain_loss, 100. * regiontrain_acc))

            nodevalid_loss, nodevalid_acc, regionvalid_loss, regionvalid_acc,f1score,auc,kappa,ham,rescore = eval_net(
                args, model, test_mask, criterion,g,test_keylist,regionlabeldict,h0)
            vbar.set_description(
                'valid set - region loss: {:.4f}, region accuracy: {:.0f}%'
                .format(regionvalid_loss, 100. * regionvalid_acc))

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write('%s %s %s %s' % (
                        args.dataset,
                        args.learn_eps,
                        args.neighbor_pooling_type,
                        args.graph_pooling_type
                    ))
                    f.write("\n")
                    f.write("train: %f %f valid:%f acc:%f f1:%f \n auc%f kappa:%f ham:%f rescore:%f" % (
                        regiontrain_loss, 100. * regiontrain_acc, regionvalid_loss, 100. * regionvalid_acc,f1score,auc,kappa,ham,rescore
                    ))

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)