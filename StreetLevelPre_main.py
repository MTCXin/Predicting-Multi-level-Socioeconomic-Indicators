import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score,roc_curve,accuracy_score, hinge_loss,auc,roc_auc_score,recall_score,cohen_kappa_score,hamming_loss
import dgl
from StreetLevelPre_parser import Parser
from StreetLevelPre_model import GIN


def train(args, net, train_mask, optimizer, criterion, epoch,g,h0):
    net.train()

    # bar = tqdm(range(1), unit='batch', position=2, file=sys.stdout)
    labels = g.ndata['GT'].to(args.device)
    graphs = g.to(args.device)

    outputs = net(graphs, h0)

    loss = criterion(outputs[train_mask], labels[train_mask])

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # report
    # bar.set_description('epoch-{}'.format(epoch))
    # bar.close()

    return loss.item()


def eval_net(args, net, test_mask, criterion,g,h0):
    net.eval()

    total = 0
    total_correct = 0

    labels = g.ndata['GT'].to(args.device)
    graphs = g.to(args.device)

    total = len(labels[test_mask])
    outputs = net(graphs, h0)
    _, predicted = torch.max(outputs.data, 1)

    total_correct += (predicted[test_mask] == labels[test_mask].data).sum().item()
    loss = criterion(outputs[test_mask], labels[test_mask])
    yscore_test=outputs[test_mask].data.cpu().numpy()
    testlab_onehot=np.eye(4)[labels[test_mask].cpu().numpy()]
    f1score=f1_score(labels[test_mask].cpu().numpy(),predicted[test_mask].cpu().numpy(),average='weighted')
    n_classes = 4
    fprt = dict()
    tprt = dict()
    roc_auct = dict()
    for i in range(n_classes):
        fprt[i], tprt[i], _ = roc_curve(testlab_onehot[:, i], yscore_test[:, i])
        roc_auct[i] = auc(fprt[i], tprt[i])
    fprt["micro"], tprt["micro"], _ = roc_curve(testlab_onehot.ravel(), yscore_test.ravel())
    roc_auct["micro"] = auc(fprt["micro"], tprt["micro"])
    acc = 1.0*total_correct / total
    rescore=recall_score(labels[test_mask].cpu().numpy(),predicted[test_mask].cpu().numpy(), average='macro')
    kappa = cohen_kappa_score(labels[test_mask].cpu().numpy(),predicted[test_mask].cpu().numpy())
    net.train()

    return loss.item(), acc,f1score,roc_auct["micro"],rescore,kappa


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
    (g,), _ = dgl.load_graphs('./data/graph.dgl')
    
    labels = g.ndata['GT']
    rd=np.random.rand(len(labels))
    rd2=np.random.rand(len(labels))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    h0 = torch.load('./data/pretrain_model.pt').cuda()
    zero_disab=torch.tensor(np.logical_and(rd2<args.ABORT_ZERO ,(np.array(labels))==0))
    train_mask = torch.tensor(np.logical_and(rd< args.TRAIN_SIZE*0.2 , (~zero_disab)))
    test_mask = torch.tensor(np.logical_and(rd>= args.TRAIN_SIZE  , (~zero_disab)))
    test_mask=test_mask.to(torch.bool).cuda()
    train_mask=train_mask.to(torch.bool).cuda()

    tbar = tqdm(range(args.epochs), unit="epoch", position=0, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=1, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=2, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, train_mask, optimizer, criterion, epoch,g,h0)
        scheduler.step()

        train_loss, train_acc,_,_,_,_ = eval_net(
            args, model, train_mask, criterion,g,h0)
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))

        valid_loss, valid_acc,f1score,auc,rescore,kappa = eval_net(
            args, model, test_mask, criterion,g,h0)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(valid_loss, 100. * valid_acc))

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("%f %f %f %f %f %f \n rescore:%f  kappa:%f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    f1score,
                    auc,rescore,kappa
                ))
                f.write("\n")

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    args = Parser(description='GIN').args
    print('show all arguments configuration...')
    print(args)

    main(args)