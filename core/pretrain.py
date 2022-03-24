"""Pre-train encoder and classifier for source dataset."""

import torch.nn as nn
import torch.optim as optim
import numpy as np

import params
from utils import make_variable, save_model, EarlyStopping


def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    valid_accuracies = []
    train_accuracies = []
    # initialize the early_stopping object
    patience = 50
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        train_acc = float(0)
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            features = encoder(images)
            features = features.reshape([1, 1024])
            preds = classifier(features)
            loss = criterion(preds, labels)

            pred_cls = preds.data.max(1)[1]
            train_acc += pred_cls.eq(labels.data).cpu().sum()

            # optimize source classifier
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_acc /= len(data_loader)
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: train_loss={}, train_Acc={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.data.item(),
                              train_acc))


        # eval model on test set
        valid_loss, valid_acc = eval_src(encoder, classifier, data_loader)

        # checking early stopping
        early_stopping(valid_loss, encoder)
        early_stopping(valid_loss, classifier)

        # recording train/val accuracies
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        # recording train/val losses
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        if early_stopping.early_stop:
            save_model(encoder, "KimiaNet-ADDA-source-encoder-ES-{}.pt".format(epoch + 1))
            save_model(classifier, "KimiaNet-ADDA-source-classifier-ES-{}.pt".format(epoch + 1))
            print("Early stopping")
            break

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, "KimiaNet-ADDA-source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, "KimiaNet-ADDA-source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, "KimiaNet-ADDA-source-encoder-final.pt")
    save_model(classifier, "KimiaNet-ADDA-source-classifier-final.pt")

    return encoder, classifier, avg_train_losses, avg_valid_losses, train_accuracies, valid_accuracies


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = float(0)

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).data.item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()
        # record validation loss

    loss /= len(data_loader)
    acc /= float(len(data_loader.dataset))

    print("val_loss = {}, val_Accuracy = {:2%}".format(loss, acc))
    return loss, acc
