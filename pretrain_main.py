"""Main script for ADDA."""
import torch
from torchvision import datasets, transforms
# import bachdataset from bachdata_binary

import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed, load_kimiaNet

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    data_transforms = {
        'train': transforms.Compose([
            # transforms.Resize(1000),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    """
    train_dataset = bachdataset(path='/home/ravi/Desktop/Token_mixing/Training/train.csv', transforms=data_transforms['train'])
    test_dataset = bachdataset(path='/home/ravi/Desktop/Token_mixing/Training/val.csv', transforms=data_transforms['val'])

    src_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    src_data_loader_eval = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    """

    # load models
    # loading KimiaNet
    print("=== Loading KimiaNet Encoder and Classifier ===")
    src_encoder, src_classifier = load_kimiaNet(
        pt_model_path='./KimiaNetPyTorchWeights.pth',
        input_size=1024,
        num_classes=2)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    # if not (src_encoder.restored and src_classifier.restored and
    #         params.src_model_trained):
    #     src_encoder, src_classifier = train_src(
    #         src_encoder, src_classifier, src_data_loader)

    # # eval source model
    # print("=== Evaluating classifier for source domain ===")
    # eval_src(src_encoder, src_classifier, src_data_loader_eval)


    # # eval target encoder on test set of target dataset
    # print("=== Evaluating classifier for encoded target domain ===")
    # print(">>> source only <<<")
    # eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
